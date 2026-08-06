from __future__ import annotations

from typing import Protocol, cast

from lightning import Trainer
from torch import Tensor

from . import _capability


class Text(Protocol):
    def add_text(self, tag: str, text: str, global_step: int) -> object: ...


class Scalar(Protocol):
    def add_scalar(self, tag: str, value: float, global_step: int) -> object: ...


class Audio(Protocol):
    def add_audio(
        self,
        tag: str,
        waveform: Tensor,
        global_step: int,
        *,
        sample_rate: int,
    ) -> object: ...


class Histogram(Protocol):
    def add_histogram(self, tag: str, values: Tensor, global_step: int) -> object: ...


class HistogramRaw(Protocol):
    def add_histogram_raw(
        self,
        tag: str,
        *,
        min: float,
        max: float,
        num: float,
        sum: float,
        sum_squares: float,
        bucket_limits: list[float],
        bucket_counts: list[float],
        global_step: int,
    ) -> object: ...


def text(trainer: Trainer) -> Text | None:
    value = _experiment(trainer)
    if value is None or not callable(getattr(value, "add_text", None)):
        return None
    return cast(Text, value)


def scalar(trainer: Trainer) -> Scalar | None:
    value = _experiment(trainer)
    if value is None or not callable(getattr(value, "add_scalar", None)):
        return None
    return cast(Scalar, value)


def audio(trainer: Trainer) -> Audio | None:
    value = _experiment(trainer)
    if value is None or not callable(getattr(value, "add_audio", None)):
        return None
    return cast(Audio, value)


def histogram(trainer: Trainer) -> Histogram | None:
    value = _experiment(trainer)
    if value is None or not callable(getattr(value, "add_histogram", None)):
        return None
    return cast(Histogram, value)


def histogram_raw(trainer: Trainer) -> HistogramRaw | None:
    value = _experiment(trainer)
    if value is None or not callable(getattr(value, "add_histogram_raw", None)):
        return None
    return cast(HistogramRaw, value)


def _experiment(trainer: Trainer) -> object | None:
    return _capability.logger_experiment(trainer)


__all__ = [
    "Audio",
    "Histogram",
    "HistogramRaw",
    "Scalar",
    "Text",
    "audio",
    "histogram",
    "histogram_raw",
    "scalar",
    "text",
]
