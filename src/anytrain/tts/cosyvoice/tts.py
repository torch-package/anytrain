"""Unified CosyVoice 2/3 text-to-waveform inference adapter.

The adapter owns upstream ``AutoModel`` loading, reference-audio conversion and
generator aggregation. Dataset roles, prompt selection and materialized stores
belong to AnyDataset and downstream workspace resources.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import auto
from functools import cache
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, Unpack, cast, overload
from warnings import warn

import torch
from torch import Tensor

from anytrain._compat import StrEnum
from anytrain.tts import (
    AudioReference,
    EncodedAudioReference,
    PathAudioReference,
    TTSAcousticEngine,
    TTSCapabilities,
    TTSEngineCapabilities,
    TTSInferenceConfig,
    TTSOptions,
    TTSOutput,
    TTSTransformerEngine,
    TTSVoiceMode,
    WaveformReference,
    validate_text,
    waveform_duration,
)
from anytrain.tts._common import (
    config_hash,
    hashable_mapping,
    positive_float,
    validate_mapping,
)
from anytrain.tts.environments import COSYVOICE_VLLM_ENVIRONMENT

DEFAULT_MODEL_V2 = "FunAudioLLM/CosyVoice2-0.5B"
DEFAULT_MODEL_V3 = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
_V3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_RUNTIME_KEYS = frozenset(("speed", "text_frontend"))
ENGINE_CAPABILITIES = TTSEngineCapabilities(
    transformer_engines=frozenset(
        (TTSTransformerEngine.TORCH, TTSTransformerEngine.VLLM)
    ),
    acoustic_engines=frozenset((TTSAcousticEngine.TORCH, TTSAcousticEngine.TENSORRT)),
)


class CosyVoiceVersion(StrEnum):
    """Supported upstream CosyVoice model generations."""

    V2 = auto()
    V3 = auto()


class CosyVoiceLoadKwargs(TypedDict, total=False):
    load_jit: bool
    load_trt: bool
    load_vllm: bool
    fp16: bool
    trt_concurrent: int


@dataclass(frozen=True)
class CosyVoiceTTSConfig:
    """Serializable semantic configuration for one CosyVoice backend."""

    version: CosyVoiceVersion = CosyVoiceVersion.V3
    model: str = DEFAULT_MODEL_V3
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)
    inference: TTSInferenceConfig | None = None
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, CosyVoiceVersion):
            raise TypeError("version must be a CosyVoiceVersion.")
        if not isinstance(self.model, str):
            raise TypeError("model must be a string.")
        if not self.model:
            raise ValueError("model must be a non-empty string.")
        validate_mapping(self.load_kwargs, "load_kwargs")
        validate_mapping(self.runtime_kwargs, "runtime_kwargs")
        _runtime_kwargs(self.runtime_kwargs)
        if self.inference is not None and not isinstance(
            self.inference, TTSInferenceConfig
        ):
            raise TypeError("inference must be a TTSInferenceConfig or None.")
        if self.max_concurrency is not None and (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
        ):
            raise TypeError("max_concurrency must be a positive integer.")
        if self.max_concurrency is not None and self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        if self.version is CosyVoiceVersion.V3 and "load_jit" in self.load_kwargs:
            raise ValueError("CosyVoice 3 does not accept load_jit.")
        resolved_inference = self.resolved_inference()
        if (
            self.inference is None
            and resolved_inference.max_concurrency > 1
            and resolved_inference.transformer_engine is TTSTransformerEngine.TORCH
        ):
            raise ValueError("max_concurrency greater than 1 requires load_vllm=True.")
        ENGINE_CAPABILITIES.validate(resolved_inference, "CosyVoice")

    def options(self) -> TTSOptions:
        return TTSOptions(extra=self.runtime_kwargs)

    def hash_dict(self) -> dict[str, object]:
        load_kwargs = {
            key: value
            for key, value in self.load_kwargs.items()
            if key not in {"load_vllm", "load_trt", "trt_concurrent"}
        }
        return {
            "version": self.version.value,
            "model": self.model,
            "inference": self.resolved_inference().hash_dict(),
            "load_kwargs": hashable_mapping(load_kwargs),
            "runtime_kwargs": hashable_mapping(self.runtime_kwargs),
        }

    def resolved_inference(self) -> TTSInferenceConfig:
        legacy = _legacy_inference(self.load_kwargs, self.max_concurrency)
        if self.inference is None:
            return legacy
        if self.max_concurrency is not None and (
            self.max_concurrency != self.inference.max_concurrency
        ):
            raise ValueError(
                "CosyVoice max_concurrency conflicts with inference.max_concurrency."
            )
        _validate_legacy_engine_flags(self.load_kwargs, self.inference)
        trt_concurrent = self.load_kwargs.get("trt_concurrent")
        if (
            trt_concurrent is not None
            and trt_concurrent != self.inference.max_concurrency
        ):
            raise ValueError(
                "CosyVoice trt_concurrent conflicts with inference.max_concurrency."
            )
        return self.inference


@dataclass
class CosyVoiceTTS:
    """CosyVoice 2/3 backend implementing :class:`anytrain.tts.TTSBackend`."""

    model: Any
    config: CosyVoiceTTSConfig = field(default_factory=CosyVoiceTTSConfig)
    model_loaded_from: str = "custom"

    name: str = "cosyvoice"

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path | None = None,
        *,
        version: CosyVoiceVersion | str = CosyVoiceVersion.V3,
        device: str | torch.device | None = "auto",
        inference: TTSInferenceConfig | None = None,
        runtime_kwargs: Mapping[str, object] | None = None,
        max_concurrency: int | None = None,
        **load_kwargs: Unpack[CosyVoiceLoadKwargs],
    ) -> CosyVoiceTTS:
        resolved_version = CosyVoiceVersion(version)
        resolved_model = str(model or _default_model(resolved_version))
        config = CosyVoiceTTSConfig(
            version=resolved_version,
            model=resolved_model,
            load_kwargs=load_kwargs,
            runtime_kwargs=dict(runtime_kwargs or {}),
            inference=inference,
            max_concurrency=max_concurrency,
        )
        resolved_load_kwargs = _load_kwargs(config.resolved_inference(), load_kwargs)
        if config.resolved_inference().transformer_engine is TTSTransformerEngine.VLLM:
            COSYVOICE_VLLM_ENVIRONMENT.validate()
        try:
            from cosyvoice.cli.cosyvoice import AutoModel  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "anytrain.tts.cosyvoice requires the official CosyVoice source package "
                "and its Matcha-TTS submodule. Install the validated engine runtime "
                "with `anytrain[cosyvoice-vllm]` when load_vllm is enabled, then make "
                "the pinned CosyVoice checkout importable."
            ) from exc
        if config.resolved_inference().transformer_engine is TTSTransformerEngine.VLLM:
            _register_vllm_model()
        try:
            with _device_context(device):
                loaded = AutoModel(model_dir=resolved_model, **resolved_load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load CosyVoice {resolved_version.value} model {resolved_model!r}"
            ) from exc
        return cls(loaded, config=config, model_loaded_from="cosyvoice")

    @property
    def sample_rate(self) -> int:
        value = getattr(self.model, "sample_rate", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError("CosyVoice model returned an invalid sample_rate.")
        return value

    @property
    def capabilities(self) -> TTSCapabilities:
        speakers = _available_speakers(self.model)
        modes = {TTSVoiceMode.REFERENCE_AUDIO}
        if speakers:
            modes.add(TTSVoiceMode.SPEAKER_ID)
        return TTSCapabilities(
            voice_modes=frozenset(modes),
            speaker_ids=speakers or None,
            engines=ENGINE_CAPABILITIES,
        )

    def config_hash(self) -> str:
        return config_hash(self.config.hash_dict())

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
        reference_audio, reference_audios = _resolve_reference_aliases(
            reference_audio=reference_audio,
            reference_audios=reference_audios,
            reference_audio_path=reference_audio_path,
            reference_audio_paths=reference_audio_paths,
        )
        resolved = self.config.options().merged(options)
        _validate_options(resolved)
        single = isinstance(text, str)
        texts = [validate_text(text)] if single else _text_batch(text)
        references = _reference_batch(
            reference_audio=reference_audio,
            reference_audios=reference_audios,
            count=len(texts),
            single=single,
        )
        speakers = _optional_text_batch(resolved.speaker, len(texts), "speaker")
        prompt_texts = _optional_text_batch(
            resolved.prompt_text,
            len(texts),
            "prompt_text",
        )
        instructions = _optional_text_batch(
            resolved.instruction,
            len(texts),
            "instruction",
        )
        runtime = _runtime_kwargs(resolved.extra)
        requests = [
            {
                "text": current_text,
                "reference": references[index],
                "speaker": None if speakers is None else speakers[index],
                "prompt_text": None if prompt_texts is None else prompt_texts[index],
                "instruction": None if instructions is None else instructions[index],
            }
            for index, current_text in enumerate(texts)
        ]
        max_concurrency = self.config.resolved_inference().max_concurrency
        if len(requests) == 1 or max_concurrency == 1:
            outputs = [self._synthesize_one(**request, runtime=runtime) for request in requests]
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(requests), max_concurrency),
                thread_name_prefix="cosyvoice",
            ) as executor:
                futures = [
                    executor.submit(self._synthesize_one, **request, runtime=runtime)
                    for request in requests
                ]
                outputs = [future.result() for future in futures]
        return outputs[0] if single else outputs

    @torch.no_grad()
    def _synthesize_one(
        self,
        text: str,
        *,
        reference: AudioReference | None,
        speaker: str | None,
        prompt_text: str | None,
        instruction: str | None,
        runtime: Mapping[str, object],
    ) -> TTSOutput:
        if reference is None:
            if prompt_text is not None or instruction is not None:
                raise ValueError("CosyVoice prompt_text and instruction require reference audio.")
            if speaker is None:
                raise ValueError("CosyVoice synthesis requires reference audio or a speaker id.")
            generated = self.model.inference_sft(
                text,
                speaker,
                stream=False,
                **runtime,
            )
            mode = "sft"
            waveform = _waveform(generated)
        else:
            with _prompt_audio(reference) as prompt_audio:
                zero_shot_spk_id = "" if speaker is None else speaker
                if instruction is not None:
                    generated = self.model.inference_instruct2(
                        text,
                        instruction,
                        prompt_audio,
                        zero_shot_spk_id=zero_shot_spk_id,
                        stream=False,
                        **runtime,
                    )
                    mode = "instruct2"
                elif prompt_text is not None:
                    generated = self.model.inference_zero_shot(
                        text,
                        _prompt_text(prompt_text, self.config.version),
                        prompt_audio,
                        zero_shot_spk_id=zero_shot_spk_id,
                        stream=False,
                        **runtime,
                    )
                    mode = "zero_shot"
                else:
                    generated = self.model.inference_cross_lingual(
                        text,
                        prompt_audio,
                        zero_shot_spk_id=zero_shot_spk_id,
                        stream=False,
                        **runtime,
                    )
                    mode = "cross_lingual"
                waveform = _waveform(generated)
        meta: dict[str, object] = {
            "backend": self.name,
            "model": self.config.model,
            "version": self.config.version.value,
            "mode": mode,
        }
        if speaker is not None:
            meta["speaker"] = speaker
        return TTSOutput(
            waveform=waveform,
            sample_rate=self.sample_rate,
            duration=waveform_duration(waveform, self.sample_rate),
            meta=meta,
        )


def _default_model(version: CosyVoiceVersion) -> str:
    return DEFAULT_MODEL_V2 if version is CosyVoiceVersion.V2 else DEFAULT_MODEL_V3


def _legacy_inference(
    load_kwargs: Mapping[str, object],
    max_concurrency: int | None,
) -> TTSInferenceConfig:
    load_vllm = load_kwargs.get("load_vllm", False)
    load_trt = load_kwargs.get("load_trt", False)
    if not isinstance(load_vllm, bool):
        raise TypeError("load_vllm must be a bool when set.")
    if not isinstance(load_trt, bool):
        raise TypeError("load_trt must be a bool when set.")
    return TTSInferenceConfig(
        transformer_engine=(
            TTSTransformerEngine.VLLM if load_vllm else TTSTransformerEngine.TORCH
        ),
        acoustic_engine=(
            TTSAcousticEngine.TENSORRT if load_trt else TTSAcousticEngine.TORCH
        ),
        max_concurrency=1 if max_concurrency is None else max_concurrency,
    )


def _validate_legacy_engine_flags(
    load_kwargs: Mapping[str, object],
    inference: TTSInferenceConfig,
) -> None:
    expected = {
        "load_vllm": inference.transformer_engine is TTSTransformerEngine.VLLM,
        "load_trt": inference.acoustic_engine is TTSAcousticEngine.TENSORRT,
    }
    for key, value in expected.items():
        if key in load_kwargs and load_kwargs[key] is not value:
            raise ValueError(f"CosyVoice {key} conflicts with inference engine selection.")


def _load_kwargs(
    inference: TTSInferenceConfig,
    load_kwargs: Mapping[str, object],
) -> dict[str, object]:
    result = dict(load_kwargs)
    if inference.transformer_engine is TTSTransformerEngine.VLLM:
        result["load_vllm"] = True
    if inference.acoustic_engine is TTSAcousticEngine.TENSORRT:
        result["load_trt"] = True
        result["trt_concurrent"] = inference.max_concurrency
    return result


@cache
def _register_vllm_model() -> None:
    from anytrain.tts.cosyvoice._vllm import register

    try:
        register()
    except ImportError as exc:
        raise ImportError("CosyVoice vLLM loading requires vLLM.") from exc


def _device_context(device: str | torch.device | None):
    if device is None or str(device) == "auto":
        return nullcontext()
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CosyVoice requested {resolved}, but CUDA is unavailable.")
        return torch.cuda.device(resolved)
    if resolved.type == "cpu":
        if torch.cuda.is_available():
            raise ValueError(
                "The upstream CosyVoice runtime selects CUDA whenever it is available "
                "and cannot honor an explicit CPU placement."
            )
        return nullcontext()
    raise ValueError(f"CosyVoice does not support device type {resolved.type!r}.")


def _available_speakers(model: object) -> tuple[str, ...]:
    getter = getattr(model, "list_avaliable_spks", None)
    if not callable(getter):
        return ()
    values = tuple(cast(Iterable[object], getter()))
    if any(not isinstance(value, str) or not value for value in values):
        raise RuntimeError("CosyVoice returned invalid speaker ids.")
    return cast(tuple[str, ...], values)


def _validate_options(options: TTSOptions) -> None:
    unsupported = {
        "language": options.language,
        "sample_rate": options.sample_rate,
        "max_new_tokens": options.max_new_tokens,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "seed": options.seed,
    }
    names = sorted(name for name, value in unsupported.items() if value is not None)
    if names:
        raise ValueError(f"CosyVoice does not support TTS options: {', '.join(names)}.")
    if options.prompt_text is not None and options.instruction is not None:
        raise ValueError("CosyVoice prompt_text and instruction are mutually exclusive.")


def _runtime_kwargs(value: Mapping[str, object]) -> dict[str, object]:
    validate_mapping(value, "CosyVoice runtime options")
    unknown = set(value) - _RUNTIME_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unsupported CosyVoice runtime options: {names}.")
    result = dict(value)
    speed = result.get("speed")
    if speed is not None:
        result["speed"] = positive_float(speed, "CosyVoice speed")
    text_frontend = result.get("text_frontend")
    if text_frontend is not None and not isinstance(text_frontend, bool):
        raise TypeError("CosyVoice text_frontend must be a bool when set.")
    return result


def _text_batch(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("text batch must be a sequence of strings.")
    texts = [validate_text(text) for text in value]
    if not texts:
        raise ValueError("text batch must not be empty.")
    return texts


def _optional_text_batch(
    value: str | Sequence[str] | None,
    count: int,
    name: str,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise ValueError(f"{name} length must match text batch length.")
    return values


def _reference_batch(
    *,
    reference_audio: AudioReference | None,
    reference_audios: Sequence[AudioReference] | None,
    count: int,
    single: bool,
) -> Sequence[AudioReference | None]:
    if single:
        if reference_audios is not None:
            raise TypeError("single text requires reference_audio.")
        return [reference_audio]
    if reference_audio is not None:
        raise TypeError("text batches require reference_audios.")
    if reference_audios is None:
        return [None] * count
    values = list(reference_audios)
    if len(values) != count:
        raise ValueError("reference_audios length must match text batch length.")
    return values


def _resolve_reference_aliases(
    *,
    reference_audio: AudioReference | None,
    reference_audios: Sequence[AudioReference] | None,
    reference_audio_path: PathAudioReference | None,
    reference_audio_paths: Sequence[PathAudioReference] | None,
) -> tuple[AudioReference | None, Sequence[AudioReference] | None]:
    if reference_audio_path is not None:
        if reference_audio is not None:
            raise TypeError("reference_audio and reference_audio_path are mutually exclusive.")
        warn(
            "reference_audio_path is deprecated; use reference_audio.",
            DeprecationWarning,
            stacklevel=3,
        )
        reference_audio = reference_audio_path
    if reference_audio_paths is not None:
        if reference_audios is not None:
            raise TypeError("reference_audios and reference_audio_paths are mutually exclusive.")
        warn(
            "reference_audio_paths is deprecated; use reference_audios.",
            DeprecationWarning,
            stacklevel=3,
        )
        reference_audios = reference_audio_paths
    return reference_audio, reference_audios


@contextmanager
def _prompt_audio(reference: AudioReference) -> Iterator[str]:
    if isinstance(reference, (str, PathLike)):
        yield str(Path(reference).expanduser())
        return
    with TemporaryDirectory(prefix="anytrain-cosyvoice-") as directory:
        if isinstance(reference, EncodedAudioReference):
            path = Path(directory, f"reference{reference.suffix}")
            path.write_bytes(reference.data)
        elif isinstance(reference, WaveformReference):
            path = Path(directory, "reference.wav")
            waveform = _mono_waveform(reference.waveform)
            _torchaudio().save(str(path), waveform, reference.sample_rate)
        else:
            raise TypeError("unsupported CosyVoice reference audio type.")
        yield str(path)


def _mono_waveform(waveform: Tensor) -> Tensor:
    tensor = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[-1] <= 0:
        raise ValueError("CosyVoice reference waveform must have shape [channels, time].")
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    return tensor.contiguous()


def _prompt_text(value: str, version: CosyVoiceVersion) -> str:
    if version is CosyVoiceVersion.V3 and "<|endofprompt|>" not in value:
        return f"{_V3_PROMPT_PREFIX}{value}"
    return value


def _waveform(value: object) -> Tensor:
    chunks: list[Tensor] = []
    outputs: Iterable[object] = (value,) if isinstance(value, Mapping) else cast(Iterable[object], value)
    try:
        iterator = iter(outputs)
    except TypeError as exc:
        raise TypeError("CosyVoice inference must return an iterable of mappings.") from exc
    for output in iterator:
        if not isinstance(output, Mapping):
            raise TypeError("CosyVoice inference output must be a mapping.")
        speech = output.get("tts_speech")
        if speech is None:
            raise ValueError("CosyVoice inference output is missing 'tts_speech'.")
        tensor = torch.as_tensor(speech, dtype=torch.float32).detach().cpu()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[-1] <= 0:
            raise ValueError("CosyVoice output waveform must have shape [channels, time].")
        chunks.append(tensor)
    if not chunks:
        raise RuntimeError("CosyVoice inference returned no waveform chunks.")
    channels = chunks[0].shape[0]
    if any(chunk.shape[0] != channels for chunk in chunks[1:]):
        raise ValueError("CosyVoice output chunks must have the same channel count.")
    return torch.cat(chunks, dim=-1).contiguous()


def _torchaudio() -> Any:
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "CosyVoice in-memory waveform references require torchaudio."
        ) from exc
    return torchaudio


__all__ = [
    "DEFAULT_MODEL_V2",
    "DEFAULT_MODEL_V3",
    "CosyVoiceLoadKwargs",
    "CosyVoiceTTS",
    "CosyVoiceTTSConfig",
    "CosyVoiceVersion",
]
