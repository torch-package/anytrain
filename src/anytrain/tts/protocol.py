from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Protocol, TypedDict, Union, overload

from torch import Tensor
from typing_extensions import Unpack

from ._common import positive_int, validate_mapping

AudioReference = Union[str, PathLike[str]]


class TTSKwargs(TypedDict, total=False):
    speaker: str | Sequence[str] | None
    language: str | None
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
        if self.max_new_tokens is not None:
            positive_int(self.max_new_tokens, "max_new_tokens")
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be positive when set.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when set.")
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
        if self.duration <= 0:
            raise ValueError("duration must be positive.")


class TTSBackend(Protocol):
    name: str
    sample_rate: int

    def config_hash(self) -> str: ...

    @overload
    def synthesize(
        self,
        text: str,
        options: TTSOptions | None = None,
        *,
        reference_audio_path: AudioReference | None = None,
    ) -> TTSOutput: ...

    @overload
    def synthesize(
        self,
        text: Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audio_paths: Sequence[AudioReference] | None = None,
    ) -> list[TTSOutput]: ...

    def synthesize(
        self,
        text: str | Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audio_path: AudioReference | None = None,
        reference_audio_paths: Sequence[AudioReference] | None = None,
    ) -> TTSOutput | list[TTSOutput]: ...


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text.strip():
        raise ValueError("text must not be empty.")
    return text


def waveform_duration(waveform: Tensor, sample_rate: int) -> float:
    positive_int(sample_rate, "sample_rate")
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, time].")
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
