from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioBackendIdentity:
    preset: str
    artifact: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.preset, name="preset")
        if self.artifact is not None:
            _non_empty_string(self.artifact, name="artifact")
        if self.revision is not None:
            _non_empty_string(self.revision, name="revision")


@dataclass(frozen=True)
class AudioCodeStreamSpec:
    """Static shape and vocabulary metadata for one named code stream."""

    name: str
    vocab_size: int
    frame_rate: float | None = None
    fixed_units: int | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.name, name="name")
        _positive_int(self.vocab_size, name="vocab_size")
        if self.frame_rate is not None:
            _positive_float(self.frame_rate, name="frame_rate")
        if self.fixed_units is not None:
            _positive_int(self.fixed_units, name="fixed_units")
        if self.frame_rate is None and self.fixed_units is None:
            raise ValueError("audio code streams require frame_rate or fixed_units.")
        if self.frame_rate is not None and self.fixed_units is not None:
            raise ValueError("audio code streams cannot set both frame_rate and fixed_units.")


@dataclass(frozen=True)
class AudioCodeSpec:
    """Static inventory of structured named code streams."""

    sample_rate: int
    streams: tuple[AudioCodeStreamSpec, ...]

    def __post_init__(self) -> None:
        _positive_int(self.sample_rate, name="sample_rate")
        if not isinstance(self.streams, tuple):
            raise TypeError("streams must be a tuple of AudioCodeStreamSpec values.")
        if not self.streams:
            raise ValueError("streams must not be empty.")
        if any(not isinstance(stream, AudioCodeStreamSpec) for stream in self.streams):
            raise TypeError("streams must contain AudioCodeStreamSpec values.")
        names = tuple(stream.name for stream in self.streams)
        if len(set(names)) != len(names):
            raise ValueError("audio code stream names must be unique.")

    def stream(self, name: str) -> AudioCodeStreamSpec:
        _non_empty_string(name, name="stream name")
        for stream in self.streams:
            if stream.name == name:
                return stream
        raise ValueError(
            f"unknown code stream {name!r}; available streams are "
            f"{tuple(stream.name for stream in self.streams)!r}."
        )

    def select(self, names: tuple[str, ...]) -> AudioCodeSpec:
        requested = _names(names, name="selected names")
        return AudioCodeSpec(
            sample_rate=self.sample_rate,
            streams=tuple(self.stream(name) for name in requested),
        )

    def exclude(self, names: tuple[str, ...]) -> AudioCodeSpec:
        excluded_names = _names(names, name="excluded names")
        for name in excluded_names:
            self.stream(name)
        excluded = set(excluded_names)
        return AudioCodeSpec(
            sample_rate=self.sample_rate,
            streams=tuple(
                stream for stream in self.streams if stream.name not in excluded
            ),
        )


def _names(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of strings.")
    for index, item in enumerate(value):
        _non_empty_string(item, name=f"{name}[{index}]")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique values.")
    return value


def _non_empty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _positive_float(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


__all__ = ["AudioBackendIdentity", "AudioCodeSpec", "AudioCodeStreamSpec"]
