"""Spark-TTS reference-audio inference adapter.

The upstream runtime only exposes scalar inference. Sequence inputs are
executed serially for API compatibility, so this backend is intended for smoke
tests and short probes, not long-running production materialization.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, Unpack, overload

import torch

from anytrain.tts import (
    AudioReference,
    EncodedAudioReference,
    PathAudioReference,
    TTSCapabilities,
    TTSOptions,
    TTSOutput,
    TTSVoiceMode,
    WaveformReference,
    validate_text,
    waveform_duration,
)
from anytrain.tts._common import config_hash, hashable_mapping, validate_mapping

DEFAULT_MODEL = "SparkAudio/Spark-TTS-0.5B"


class SparkLoadKwargs(TypedDict, total=False):
    cache_dir: str | PathLike[str] | None
    local_files_only: bool
    revision: str | None
    device: str | torch.device


@dataclass(frozen=True)
class SparkTTSConfig:
    """Serializable Spark-TTS loading and generation defaults."""

    model: str = DEFAULT_MODEL
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)
    load_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("model must be a string.")
        if not self.model:
            raise ValueError("model must be a non-empty string.")
        validate_mapping(self.runtime_kwargs, "runtime_kwargs")
        validate_mapping(self.load_kwargs, "load_kwargs")

    def options(self) -> TTSOptions:
        runtime = dict(self.runtime_kwargs)
        temperature = _optional_float(runtime.pop("temperature", None), "temperature")
        top_p = _optional_float(runtime.pop("top_p", None), "top_p")
        return TTSOptions(
            temperature=temperature,
            top_p=top_p,
            extra=runtime,
        )


@dataclass
class SparkTTS:
    """Run Spark-TTS voice cloning from explicit reference audio and text."""

    model: Any
    config: SparkTTSConfig = field(default_factory=SparkTTSConfig)
    model_loaded_from: str = "spark_tts"

    name: str = "spark-tts-0.5b"

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_kwargs: Unpack[SparkLoadKwargs],
    ) -> SparkTTS:
        try:
            from cli.SparkTTS import (  # pyright: ignore[reportMissingImports]
                SparkTTS as SparkRuntime,
            )
        except ImportError as exc:
            raise ImportError(
                "anytrain.tts.spark requires Spark-TTS. Install it with "
                "python -m pip install 'anytrain[spark-tts]'."
            ) from exc

        cache_dir = load_kwargs.pop("cache_dir", None)
        local_files_only = load_kwargs.pop("local_files_only", False)
        revision = load_kwargs.pop("revision", None)
        device = load_kwargs.pop("device", "cuda:0")
        source = _pretrained_source(
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        loaded = SparkRuntime(Path(source), device=torch.device(device))
        config = SparkTTSConfig(
            model=str(model),
            runtime_kwargs={} if runtime_kwargs is None else dict(runtime_kwargs),
            load_kwargs={
                "revision": revision,
                **load_kwargs,
            },
        )
        return cls(loaded, config=config)

    @property
    def sample_rate(self) -> int:
        value = getattr(self.model, "sample_rate", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError("Spark-TTS runtime returned an invalid sample rate.")
        return value

    @property
    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            voice_modes=frozenset((TTSVoiceMode.REFERENCE_AUDIO,)),
        )

    def config_hash(self) -> str:
        return config_hash(
            {
                "backend": self.name,
                "model": self.config.model,
                "runtime_kwargs": hashable_mapping(self.config.runtime_kwargs),
                "load_kwargs": hashable_mapping(self.config.load_kwargs),
            }
        )

    @overload
    def synthesize(
        self,
        text: str,
        options: TTSOptions | None = None,
        *,
        reference_audio: AudioReference | None = None,
        reference_audio_path: PathAudioReference | None = None,
    ) -> TTSOutput: ...

    @overload
    def synthesize(
        self,
        text: Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audios: Sequence[AudioReference] | None = None,
        reference_audio_paths: Sequence[PathAudioReference] | None = None,
    ) -> list[TTSOutput]: ...

    @torch.no_grad()
    def synthesize(
        self,
        text: str | Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audio: AudioReference | None = None,
        reference_audios: Sequence[AudioReference] | None = None,
        reference_audio_path: PathAudioReference | None = None,
        reference_audio_paths: Sequence[PathAudioReference] | None = None,
    ) -> TTSOutput | list[TTSOutput]:
        resolved = self.config.options().merged(options)
        _validate_options(resolved)
        reference_audio, reference_audios = _resolve_reference_aliases(
            reference_audio=reference_audio,
            reference_audios=reference_audios,
            reference_audio_path=reference_audio_path,
            reference_audio_paths=reference_audio_paths,
        )
        if isinstance(text, str):
            if reference_audios is not None:
                raise TypeError("single text requires reference_audio.")
            prompt_text = _single_prompt_text(resolved)
            reference = _required_reference(reference_audio)
            return self._synthesize_one(
                validate_text(text),
                reference,
                prompt_text,
                resolved,
            )

        if isinstance(text, bytes) or not isinstance(text, Sequence):
            raise TypeError("text must be a string or sequence of strings.")
        if reference_audio is not None:
            raise TypeError("text batches require reference_audios.")
        texts = [validate_text(value) for value in text]
        if not texts:
            raise ValueError("text batch must not be empty.")
        references = _required_references(reference_audios, len(texts))
        prompt_texts = _batch_prompt_texts(resolved, len(texts))
        with _materialized_audio_references(references) as paths:
            return [
                self._inference(target, path, prompt, resolved)
                for target, path, prompt in zip(
                    texts,
                    paths,
                    prompt_texts,
                    strict=True,
                )
            ]

    def _synthesize_one(
        self,
        text: str,
        reference: AudioReference,
        prompt_text: str,
        options: TTSOptions,
    ) -> TTSOutput:
        with _materialized_audio_references((reference,)) as paths:
            return self._inference(text, paths[0], prompt_text, options)

    def _inference(
        self,
        text: str,
        reference_path: str,
        prompt_text: str,
        options: TTSOptions,
    ) -> TTSOutput:
        value = self.model.inference(
            text,
            prompt_speech_path=Path(reference_path),
            prompt_text=prompt_text,
            **_runtime_kwargs(options),
        )
        waveform = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise RuntimeError("Spark-TTS returned an invalid waveform shape.")
        if not torch.isfinite(waveform).all():
            raise RuntimeError("Spark-TTS returned a non-finite waveform.")
        sample_rate = self.sample_rate
        return TTSOutput(
            waveform=waveform,
            sample_rate=sample_rate,
            duration=waveform_duration(waveform, sample_rate),
            meta={"prompt_text": prompt_text},
        )


def _validate_options(options: TTSOptions) -> None:
    if options.speaker is not None:
        raise ValueError("Spark-TTS reference cloning does not support speaker ids.")
    if options.language is not None:
        raise ValueError("Spark-TTS does not accept a language override.")
    if options.sample_rate is not None:
        raise ValueError("Spark-TTS does not support sample_rate overrides.")
    if options.max_new_tokens is not None:
        raise ValueError("Spark-TTS runtime does not expose max_new_tokens.")
    if options.seed is not None:
        raise ValueError("Spark-TTS runtime does not expose deterministic seeds.")
    if options.instruction is not None:
        raise ValueError("Spark-TTS does not support instruction prompts.")
    unknown = set(options.extra) - {"top_k"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unsupported Spark-TTS runtime options: {names}.")


def _runtime_kwargs(options: TTSOptions) -> dict[str, object]:
    runtime: dict[str, object] = {}
    if options.temperature is not None:
        runtime["temperature"] = options.temperature
    if options.top_p is not None:
        runtime["top_p"] = options.top_p
    top_k = options.extra.get("top_k")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("Spark-TTS top_k must be a positive integer.")
        if top_k <= 0:
            raise ValueError("Spark-TTS top_k must be a positive integer.")
        runtime["top_k"] = top_k
    return runtime


def _single_prompt_text(options: TTSOptions) -> str:
    value = options.prompt_text
    if not isinstance(value, str):
        raise TypeError("Spark-TTS single synthesis requires prompt_text.")
    return validate_text(value)


def _batch_prompt_texts(options: TTSOptions, count: int) -> list[str]:
    values = options.prompt_text
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("Spark-TTS text batches require a prompt_text sequence.")
    prompt_texts = [
        validate_text(value) if isinstance(value, str) else _invalid_prompt_text()
        for value in values
    ]
    if len(prompt_texts) != count:
        raise ValueError("prompt_text length must match text batch length.")
    return prompt_texts


def _invalid_prompt_text() -> str:
    raise TypeError("prompt_text must contain strings.")


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Spark-TTS {name} must be numeric when set.")
    return float(value)


def _required_reference(value: AudioReference | None) -> AudioReference:
    if value is None:
        raise ValueError("Spark-TTS voice cloning requires reference_audio.")
    return value


def _required_references(
    values: Sequence[AudioReference] | None,
    count: int,
) -> list[AudioReference]:
    if values is None:
        raise ValueError("Spark-TTS voice cloning requires reference_audios.")
    if isinstance(values, (str, bytes, PathLike)):
        raise TypeError("reference_audios must be a sequence of audio references.")
    references = list(values)
    if len(references) != count:
        raise ValueError("reference_audios length must match text batch length.")
    return references


def _resolve_reference_aliases(
    *,
    reference_audio: AudioReference | None,
    reference_audios: Sequence[AudioReference] | None,
    reference_audio_path: PathAudioReference | None,
    reference_audio_paths: Sequence[PathAudioReference] | None,
) -> tuple[AudioReference | None, Sequence[AudioReference] | None]:
    if reference_audio is not None and reference_audio_path is not None:
        raise TypeError("reference_audio and reference_audio_path cannot both be passed.")
    if reference_audios is not None and reference_audio_paths is not None:
        raise TypeError("reference_audios and reference_audio_paths cannot both be passed.")
    if reference_audio_path is not None:
        reference_audio = reference_audio_path
    if reference_audio_paths is not None:
        reference_audios = reference_audio_paths
    return reference_audio, reference_audios


@contextmanager
def _materialized_audio_references(
    values: Sequence[AudioReference],
) -> Iterator[list[str]]:
    temporary: TemporaryDirectory[str] | None = None
    paths: list[str] = []
    try:
        for index, value in enumerate(values):
            if isinstance(value, (str, PathLike)):
                paths.append(str(value))
                continue
            if temporary is None:
                temporary = TemporaryDirectory(prefix="anytrain-spark-reference-")
            root = Path(temporary.name)
            if isinstance(value, WaveformReference):
                path = root / f"reference-{index}.wav"
                _save_waveform_reference(value, path)
            elif isinstance(value, EncodedAudioReference):
                path = root / f"reference-{index}{value.suffix}"
                path.write_bytes(value.data)
            else:
                raise TypeError(
                    "reference audio must be a path, WaveformReference, or EncodedAudioReference."
                )
            paths.append(str(path))
        yield paths
    finally:
        if temporary is not None:
            temporary.cleanup()


def _save_waveform_reference(value: WaveformReference, path: Path) -> None:
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "WaveformReference requires torchaudio; install anytrain[spark-tts]."
        ) from exc
    torchaudio.save(
        str(path),
        value.waveform.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0),
        value.sample_rate,
    )


def _pretrained_source(
    model: str | Path,
    *,
    cache_dir: str | PathLike[str] | None,
    local_files_only: bool,
    revision: str | None,
) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Resolving Spark-TTS checkpoints requires huggingface-hub.") from exc
    return snapshot_download(
        repo_id=str(model),
        revision=revision,
        cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser()),
        local_files_only=local_files_only,
    )
