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
class AudioCodeSpec:
    """Static inventory of named code streams."""

    names: tuple[str, ...]
    vocab_sizes: tuple[int, ...]
    sample_rate: int
    frame_rates: tuple[float | None, ...]

    def __post_init__(self) -> None:
        _names(self.names)
        _sizes(self.vocab_sizes)
        _positive_int(self.sample_rate, name="sample_rate")
        _frame_rates(self.frame_rates)
        if len(self.names) != len(self.vocab_sizes):
            raise ValueError("names and vocab_sizes must have the same length.")
        if len(self.names) != len(self.frame_rates):
            raise ValueError("names and frame_rates must have the same length.")

    def select(self, names: tuple[str, ...]) -> AudioCodeSpec:
        requested = _names(names, name="selected names")
        indices = self._indices(requested)
        return AudioCodeSpec(
            names=requested,
            vocab_sizes=tuple(self.vocab_sizes[index] for index in indices),
            sample_rate=self.sample_rate,
            frame_rates=tuple(self.frame_rates[index] for index in indices),
        )

    def exclude(self, names: tuple[str, ...]) -> AudioCodeSpec:
        excluded_names = _names(names, name="excluded names")
        self._indices(excluded_names)
        excluded = set(excluded_names)
        return self.select(tuple(name for name in self.names if name not in excluded))

    def _indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        positions = {name: index for index, name in enumerate(self.names)}
        missing = tuple(name for name in names if name not in positions)
        if missing:
            raise ValueError(f"unknown code names {missing!r}; available names are {self.names!r}.")
        return tuple(positions[name] for name in names)


def _sizes(value: object, *, name: str = "vocab_sizes") -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of integers.")
    for index, size in enumerate(value):
        _positive_int(size, name=f"{name}[{index}]")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _names(value: object, *, name: str = "names") -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of strings.")
    for index, item in enumerate(value):
        _non_empty_string(item, name=f"{name}[{index}]")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique values.")
    return value


def _frame_rates(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("frame_rates must be a tuple of positive numbers or None.")
    for index, rate in enumerate(value):
        if rate is not None:
            _positive_float(rate, name=f"frame_rates[{index}]")


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


__all__ = ["AudioBackendIdentity", "AudioCodeSpec"]
