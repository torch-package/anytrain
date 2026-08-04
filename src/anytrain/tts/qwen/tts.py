"""Qwen3-TTS CustomVoice inference adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, overload

import torch
from torch import Tensor
from typing_extensions import Unpack

from anytrain.tts import (
    AudioReference,
    PathAudioReference,
    TTSOptions,
    TTSOutput,
    validate_text,
    waveform_duration,
)
from anytrain.tts._common import config_hash, hashable_mapping, positive_int, validate_mapping

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SAMPLE_RATE = 24000


class QwenLoadKwargs(TypedDict, total=False):
    attn_implementation: object
    device: object
    device_map: object
    dtype: object
    torch_dtype: object


@dataclass(frozen=True)
class QwenCustomVoiceTTSConfig:
    model: str = DEFAULT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    language: str = "Auto"
    speaker: str | None = None
    instruct: str | None = None
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty.")
        positive_int(self.sample_rate, "sample_rate")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string.")
        if self.speaker is not None and not self.speaker:
            raise ValueError("speaker must be non-empty when set.")
        validate_mapping(self.load_kwargs, "load_kwargs")
        validate_mapping(self.runtime_kwargs, "runtime_kwargs")

    def options(self) -> TTSOptions:
        kwargs: dict[str, object] = {
            "language": self.language,
            "sample_rate": self.sample_rate,
            "extra": self.runtime_kwargs,
        }
        if self.speaker is not None:
            kwargs["speaker"] = self.speaker
        return TTSOptions(**kwargs)

    def hash_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "speaker": self.speaker,
            "instruct": self.instruct,
            "load_kwargs": hashable_mapping(self.load_kwargs),
            "runtime_kwargs": hashable_mapping(self.runtime_kwargs),
        }


@dataclass
class QwenCustomVoiceTTS:
    model: Any
    config: QwenCustomVoiceTTSConfig = field(default_factory=QwenCustomVoiceTTSConfig)
    model_loaded_from: str = "custom"

    name: str = "qwen3-tts-customvoice"

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: str = "Auto",
        speaker: str | None = None,
        instruct: str | None = None,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_kwargs: Unpack[QwenLoadKwargs],
    ) -> QwenCustomVoiceTTS:
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "anytrain.tts.qwen requires Qwen3-TTS. "
                "Install Qwen3-TTS with "
                "`python -m pip install git+https://github.com/QwenLM/Qwen3-TTS.git`."
            ) from exc

        runtime = dict(runtime_kwargs or {})
        config = QwenCustomVoiceTTSConfig(
            model=str(model),
            sample_rate=sample_rate,
            language=language,
            speaker=speaker,
            instruct=instruct,
            load_kwargs=load_kwargs,
            runtime_kwargs=runtime,
        )
        try:
            loaded = Qwen3TTSModel.from_pretrained(str(model), **load_kwargs)
        except Exception as exc:
            raise RuntimeError(f"failed to load Qwen3-TTS checkpoint {model!s}") from exc
        return cls(loaded, config=config, model_loaded_from="qwen_tts")

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def config_hash(self) -> str:
        return config_hash(self.config.hash_dict())

    def supported_speakers(self) -> tuple[str, ...]:
        getter = getattr(self.model, "get_supported_speakers", None)
        if not callable(getter):
            return ()
        speakers = tuple(getter())
        if any(not isinstance(speaker, str) or not speaker for speaker in speakers):
            raise RuntimeError("Qwen3-TTS returned invalid supported speaker ids.")
        return speakers

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
        if any(
            value is not None
            for value in (
                reference_audio,
                reference_audios,
                reference_audio_path,
                reference_audio_paths,
            )
        ):
            raise ValueError("Qwen CustomVoice TTS does not accept reference audio.")
        self._validate_options(options)
        merged = self._options(options)
        if merged.speaker is None:
            raise ValueError("Qwen CustomVoice TTS requires a speaker id.")
        return self._synthesize_custom_voice(
            text,
            speakers=merged.speaker,
            languages=merged.language or self.config.language,
            instructs=self.config.instruct,
            options=merged,
        )

    def synthesize_custom_voice(
        self,
        text: str | Sequence[str],
        *,
        speakers: str | Sequence[str] | None = None,
        languages: str | Sequence[str] | None = None,
        instructs: str | Sequence[str] | None = None,
        options: TTSOptions | None = None,
    ) -> TTSOutput | list[TTSOutput]:
        self._validate_options(options)
        return self._synthesize_custom_voice(
            text,
            speakers=speakers,
            languages=languages,
            instructs=instructs,
            options=self._options(options),
        )

    def _synthesize_custom_voice(
        self,
        text: str | Sequence[str],
        *,
        speakers: str | Sequence[str] | None,
        languages: str | Sequence[str] | None,
        instructs: str | Sequence[str] | None,
        options: TTSOptions,
    ) -> TTSOutput | list[TTSOutput]:
        single = isinstance(text, str)
        texts = [validate_text(text)] if single else _text_batch(text)
        resolved_speakers = _value_batch(
            speakers if speakers is not None else self.config.speaker,
            count=len(texts),
            name="speakers",
        )
        resolved_languages = _value_batch(
            languages if languages is not None else self.config.language,
            count=len(texts),
            name="languages",
        )
        resolved_instructs = _optional_value_batch(
            instructs if instructs is not None else self.config.instruct,
            count=len(texts),
            name="instructs",
        )
        self._validate_speakers(resolved_speakers)

        runtime = _runtime_kwargs(options)
        generated = self.model.generate_custom_voice(
            text=text if single else texts,
            language=resolved_languages[0] if single else resolved_languages,
            speaker=resolved_speakers[0] if single else resolved_speakers,
            instruct=None
            if resolved_instructs is None
            else resolved_instructs[0]
            if single
            else resolved_instructs,
            **runtime,
        )
        outputs = _outputs(
            generated,
            speakers=resolved_speakers,
            languages=resolved_languages,
            default_sample_rate=self.config.sample_rate,
        )
        return outputs[0] if single else outputs

    def _options(self, options: TTSOptions | None) -> TTSOptions:
        return self.config.options().merged(options)

    @staticmethod
    def _validate_options(options: TTSOptions | None) -> None:
        if options is None:
            return
        if options.seed is not None:
            raise ValueError("Qwen CustomVoice TTS does not support seed.")
        if options.sample_rate is not None:
            raise ValueError("Qwen CustomVoice TTS does not support sample_rate overrides.")

    def _validate_speakers(self, speakers: Sequence[str]) -> None:
        supported = self.supported_speakers()
        if not supported:
            return
        missing = sorted(set(speakers) - set(supported))
        if missing:
            available = ", ".join(supported)
            raise ValueError(
                f"unsupported Qwen speaker ids: {', '.join(missing)}; "
                f"available speakers: {available}."
            )


def _text_batch(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("text batch must be a sequence of strings.")
    texts = [validate_text(text) for text in value]
    if not texts:
        raise ValueError("text batch must not be empty.")
    return texts


def _value_batch(
    value: str | Sequence[str] | None,
    *,
    count: int,
    name: str,
) -> list[str]:
    if value is None:
        raise ValueError(f"Qwen CustomVoice TTS requires {name}.")
    if isinstance(value, str):
        values = [value] * count
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise TypeError(f"{name} must be a string or sequence of strings.")
    if len(values) != count:
        raise ValueError(f"{name} length must match text batch length.")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    return values


def _optional_value_batch(
    value: str | Sequence[str] | None,
    *,
    count: int,
    name: str,
) -> list[str] | None:
    if value is None:
        return None
    return _value_batch(value, count=count, name=name)


def _runtime_kwargs(options: TTSOptions) -> dict[str, object]:
    runtime = dict(options.extra)
    if options.max_new_tokens is not None:
        runtime["max_new_tokens"] = options.max_new_tokens
    if options.temperature is not None:
        runtime["temperature"] = options.temperature
    if options.top_p is not None:
        runtime["top_p"] = options.top_p
    return runtime


def _outputs(
    generated: object,
    *,
    speakers: Sequence[str],
    languages: Sequence[str],
    default_sample_rate: int,
) -> list[TTSOutput]:
    waveforms, sample_rate = _generated_parts(generated, default_sample_rate)
    values = [waveforms] if _is_waveform_value(waveforms) else list(waveforms)
    if len(values) != len(speakers):
        raise RuntimeError("Qwen3-TTS output count does not match text batch length.")
    return [
        _output(value, sample_rate, speaker=speaker, language=language)
        for value, speaker, language in zip(values, speakers, languages)
    ]


def _generated_parts(
    generated: object,
    default_sample_rate: int,
) -> tuple[object, int]:
    if isinstance(generated, tuple) and len(generated) == 2:
        waveforms, sample_rate = generated
        positive_int(sample_rate, "sample_rate")
        return waveforms, int(sample_rate)
    return generated, default_sample_rate


def _is_waveform_value(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _output(value: object, sample_rate: int, *, speaker: str, language: str) -> TTSOutput:
    waveform = _waveform(value)
    return TTSOutput(
        waveform=waveform,
        sample_rate=sample_rate,
        duration=waveform_duration(waveform, sample_rate),
        meta={
            "backend": DEFAULT_MODEL,
            "speaker": speaker,
            "language": language,
        },
    )


def _waveform(value: object) -> Tensor:
    waveform = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise RuntimeError("Qwen3-TTS returned an invalid waveform shape.")
    if not torch.isfinite(waveform).all():
        raise RuntimeError("Qwen3-TTS returned a non-finite waveform.")
    return waveform
