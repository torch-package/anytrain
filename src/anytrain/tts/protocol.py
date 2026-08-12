from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import auto
from os import PathLike
from typing import Protocol, TypedDict, Unpack, overload

from torch import Tensor

from anytrain._compat import StrEnum

from ._common import (
    optional_non_empty_string,
    positive_float,
    positive_int,
    probability,
    validate_mapping,
)

PathAudioReference = str | PathLike[str]


@dataclass(frozen=True, eq=False)
class WaveformReference:
    """In-memory reference audio with an explicit sample rate."""

    waveform: Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, Tensor):
            raise TypeError("waveform must be a torch.Tensor.")
        if self.waveform.ndim != 2:
            raise ValueError("waveform must have shape [channels, time].")
        if self.waveform.shape[0] <= 0 or self.waveform.shape[-1] <= 0:
            raise ValueError("waveform must not be empty.")
        positive_int(self.sample_rate, "sample_rate")


@dataclass(frozen=True, eq=False)
class EncodedAudioReference:
    """Encoded reference audio bytes and their file suffix."""

    data: bytes = field(repr=False)
    suffix: str

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("data must be bytes.")
        if not self.data:
            raise ValueError("data must not be empty.")
        if not isinstance(self.suffix, str):
            raise TypeError("suffix must be a string.")
        if (
            not self.suffix.startswith(".")
            or self.suffix == "."
            or "." in self.suffix[1:]
            or "/" in self.suffix
            or "\\" in self.suffix
        ):
            raise ValueError("suffix must be a file suffix such as '.wav' or '.flac'.")


AudioReference = (
    PathAudioReference
    | WaveformReference
    | EncodedAudioReference
)


class TTSVoiceMode(StrEnum):
    """Voice conditioning modes a TTS backend can guarantee at its boundary."""

    SPEAKER_ID = auto()
    REFERENCE_AUDIO = auto()


class TTSTransformerEngine(StrEnum):
    """Transformer execution engines used by TTS backends."""

    TORCH = auto()
    VLLM = auto()
    VLLM_OMNI = auto()


class TTSAcousticEngine(StrEnum):
    """Acoustic/decoder execution engines used after token generation."""

    TORCH = auto()
    TENSORRT = auto()


@dataclass(frozen=True)
class TTSInferenceConfig:
    """Engine selection plus execution-only request concurrency."""

    transformer_engine: TTSTransformerEngine = TTSTransformerEngine.TORCH
    acoustic_engine: TTSAcousticEngine = TTSAcousticEngine.TORCH
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.transformer_engine, TTSTransformerEngine):
            raise TypeError("transformer_engine must be a TTSTransformerEngine.")
        if not isinstance(self.acoustic_engine, TTSAcousticEngine):
            raise TypeError("acoustic_engine must be a TTSAcousticEngine.")
        positive_int(self.max_concurrency, "max_concurrency")

    def hash_dict(self) -> dict[str, str]:
        """Return semantic engine identity, excluding request concurrency."""

        return {
            "transformer_engine": self.transformer_engine.value,
            "acoustic_engine": self.acoustic_engine.value,
        }


@dataclass(frozen=True)
class TTSEngineCapabilities:
    """Engine combinations implemented by one backend family."""

    transformer_engines: frozenset[TTSTransformerEngine]
    acoustic_engines: frozenset[TTSAcousticEngine]

    def __post_init__(self) -> None:
        if not isinstance(self.transformer_engines, frozenset) or any(
            not isinstance(engine, TTSTransformerEngine)
            for engine in self.transformer_engines
        ):
            raise TypeError(
                "transformer_engines must be a frozenset of TTSTransformerEngine values."
            )
        if not self.transformer_engines:
            raise ValueError("transformer_engines must not be empty.")
        if not isinstance(self.acoustic_engines, frozenset) or any(
            not isinstance(engine, TTSAcousticEngine) for engine in self.acoustic_engines
        ):
            raise TypeError(
                "acoustic_engines must be a frozenset of TTSAcousticEngine values."
            )
        if not self.acoustic_engines:
            raise ValueError("acoustic_engines must not be empty.")

    def validate(self, config: TTSInferenceConfig, backend: str) -> None:
        """Reject an engine combination the backend does not implement."""

        if not isinstance(config, TTSInferenceConfig):
            raise TypeError("inference must be a TTSInferenceConfig.")
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string.")
        if config.transformer_engine not in self.transformer_engines:
            supported = ", ".join(sorted(engine.value for engine in self.transformer_engines))
            raise ValueError(
                f"{backend} does not support transformer engine "
                f"{config.transformer_engine.value!r}; supported engines: {supported}."
            )
        if config.acoustic_engine not in self.acoustic_engines:
            supported = ", ".join(sorted(engine.value for engine in self.acoustic_engines))
            raise ValueError(
                f"{backend} does not support acoustic engine "
                f"{config.acoustic_engine.value!r}; supported engines: {supported}."
            )
        if (
            config.max_concurrency > 1
            and config.transformer_engine is TTSTransformerEngine.TORCH
        ):
            raise ValueError(
                f"{backend} max_concurrency greater than 1 requires a vLLM engine."
            )


TORCH_TTS_ENGINES = TTSEngineCapabilities(
    transformer_engines=frozenset((TTSTransformerEngine.TORCH,)),
    acoustic_engines=frozenset((TTSAcousticEngine.TORCH,)),
)


@dataclass(frozen=True)
class TTSCapabilities:
    """Static voice-conditioning and engine capabilities of one backend."""

    voice_modes: frozenset[TTSVoiceMode]
    speaker_ids: tuple[str, ...] | None = None
    engines: TTSEngineCapabilities = TORCH_TTS_ENGINES

    def __post_init__(self) -> None:
        if not isinstance(self.voice_modes, frozenset) or any(
            not isinstance(mode, TTSVoiceMode) for mode in self.voice_modes
        ):
            raise TypeError("voice_modes must be a frozenset of TTSVoiceMode values.")
        if not isinstance(self.engines, TTSEngineCapabilities):
            raise TypeError("engines must be TTSEngineCapabilities.")
        if self.speaker_ids is None:
            return
        if TTSVoiceMode.SPEAKER_ID not in self.voice_modes:
            raise ValueError("speaker_ids require TTSVoiceMode.SPEAKER_ID.")
        if not isinstance(self.speaker_ids, tuple):
            raise TypeError("speaker_ids must be a tuple of strings or None.")
        for index, speaker in enumerate(self.speaker_ids):
            _validate_speaker_id(speaker, f"speaker_ids[{index}]")
        if len(set(self.speaker_ids)) != len(self.speaker_ids):
            raise ValueError("speaker_ids must be unique.")

    def supports(self, mode: TTSVoiceMode) -> bool:
        if not isinstance(mode, TTSVoiceMode):
            raise TypeError("mode must be a TTSVoiceMode.")
        return mode in self.voice_modes

    def unsupported_speaker_ids(
        self,
        speaker_ids: Sequence[str],
    ) -> tuple[str, ...] | None:
        """Return unsupported ids, or ``None`` when the backend cannot enumerate."""

        if isinstance(speaker_ids, (str, bytes)) or not isinstance(
            speaker_ids,
            Sequence,
        ):
            raise TypeError("speaker_ids must be a sequence of strings.")
        values = tuple(speaker_ids)
        for index, speaker in enumerate(values):
            _validate_speaker_id(speaker, f"speaker_ids[{index}]")
        if not self.supports(TTSVoiceMode.SPEAKER_ID):
            return values
        if self.speaker_ids is None:
            return None
        supported = set(self.speaker_ids)
        return tuple(speaker for speaker in values if speaker not in supported)


class TTSKwargs(TypedDict, total=False):
    speaker: str | Sequence[str] | None
    language: str | None
    prompt_text: str | Sequence[str] | None
    instruction: str | Sequence[str] | None
    sample_rate: int | None
    max_new_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None
    extra: Mapping[str, object]


_TTS_OPTION_KEYS = frozenset(TTSKwargs.__annotations__)


@dataclass(init=False)
class TTSOptions:
    speaker: str | Sequence[str] | None = None
    language: str | None = None
    prompt_text: str | Sequence[str] | None = None
    instruction: str | Sequence[str] | None = None
    sample_rate: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    extra: Mapping[str, object] = field(default_factory=dict)
    _set_fields: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __init__(self, **kwargs: Unpack[TTSKwargs]) -> None:
        unknown = set(kwargs) - _TTS_OPTION_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown TTS option keys: {names}.")

        self.speaker = kwargs.get("speaker")
        self.language = kwargs.get("language")
        self.prompt_text = kwargs.get("prompt_text")
        self.instruction = kwargs.get("instruction")
        self.sample_rate = kwargs.get("sample_rate")
        self.max_new_tokens = kwargs.get("max_new_tokens")
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        self.seed = kwargs.get("seed")
        self.extra = kwargs.get("extra", {})
        self._set_fields = frozenset(kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.sample_rate is not None:
            positive_int(self.sample_rate, "sample_rate")
        _validate_speaker(self.speaker)
        optional_non_empty_string(self.language, "language")
        _validate_text_option(self.prompt_text, "prompt_text")
        _validate_text_option(self.instruction, "instruction")
        if self.max_new_tokens is not None:
            positive_int(self.max_new_tokens, "max_new_tokens")
        if self.temperature is not None:
            positive_float(self.temperature, "temperature")
        if self.top_p is not None:
            probability(self.top_p, "top_p")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer when set.")
        validate_mapping(self.extra, "extra")

    def merged(self, override: TTSOptions | None) -> TTSOptions:
        if override is None:
            return self
        options = TTSOptions(
            speaker=override.speaker if override._has("speaker") else self.speaker,
            language=override.language if override._has("language") else self.language,
            prompt_text=(
                override.prompt_text if override._has("prompt_text") else self.prompt_text
            ),
            instruction=(
                override.instruction if override._has("instruction") else self.instruction
            ),
            sample_rate=override.sample_rate
            if override._has("sample_rate")
            else self.sample_rate,
            max_new_tokens=override.max_new_tokens
            if override._has("max_new_tokens")
            else self.max_new_tokens,
            temperature=override.temperature
            if override._has("temperature")
            else self.temperature,
            top_p=override.top_p if override._has("top_p") else self.top_p,
            seed=override.seed if override._has("seed") else self.seed,
            extra=override.extra if override._has("extra") else self.extra,
        )
        options._set_fields = self._set_fields | override._set_fields
        return options

    def _has(self, name: str) -> bool:
        return name in self._set_fields

@dataclass(eq=False)
class TTSOutput:
    waveform: Tensor
    sample_rate: int
    duration: float
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, Tensor):
            raise TypeError("waveform must be a torch.Tensor.")
        if self.waveform.ndim != 2:
            raise ValueError("waveform must have shape [channels, time].")
        if self.waveform.shape[0] <= 0 or self.waveform.shape[-1] <= 0:
            raise ValueError("waveform must not be empty.")
        positive_int(self.sample_rate, "sample_rate")
        positive_float(self.duration, "duration")
        validate_mapping(self.meta, "meta")


class TTSBackend(Protocol):
    name: str
    sample_rate: int

    @property
    def capabilities(self) -> TTSCapabilities: ...

    def config_hash(self) -> str: ...

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
    ) -> TTSOutput | list[TTSOutput]: ...


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text.strip():
        raise ValueError("text must not be empty.")
    return text


def waveform_duration(waveform: Tensor, sample_rate: int) -> float:
    positive_int(sample_rate, "sample_rate")
    if not isinstance(waveform, Tensor):
        raise TypeError("waveform must be a torch.Tensor.")
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, time].")
    if waveform.shape[0] <= 0 or waveform.shape[-1] <= 0:
        raise ValueError("waveform must not be empty.")
    return float(waveform.shape[-1]) / float(sample_rate)


def _validate_speaker(value: str | Sequence[str] | None) -> None:
    if value is None:
        return
    if isinstance(value, str):
        _validate_speaker_id(value, "speaker")
        return
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise TypeError("speaker must be a string, a sequence of strings, or None.")
    if len(value) == 0:
        raise ValueError("speaker sequence must not be empty.")
    for index, speaker in enumerate(value):
        _validate_speaker_id(speaker, f"speaker[{index}]")


def _validate_speaker_id(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")


def _validate_text_option(
    value: str | Sequence[str] | None,
    name: str,
) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty when set.")
        return
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a string, a sequence of strings, or None.")
    if not value:
        raise ValueError(f"{name} sequence must not be empty.")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{index}] must be a string.")
        if not item.strip():
            raise ValueError(f"{name}[{index}] must be non-empty.")
