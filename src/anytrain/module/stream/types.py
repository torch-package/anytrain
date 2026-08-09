from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _validate_lengths(lengths: Tensor, *, batch_size: int, size: int) -> None:
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError("lengths must have shape [batch].")
    if not _is_integer_dtype(lengths.dtype):
        raise TypeError("lengths must use an integer dtype.")
    if bool((lengths < 0).any()) or bool((lengths > size).any()):
        raise ValueError("lengths must be within [0, sequence size].")


def _validate_frame_rate(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("frame_rate must be a real number or None.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError("frame_rate must be finite and greater than zero.")


@dataclass(eq=False, frozen=True, slots=True)
class Timeline:
    """A batch of stream lengths without any payload values."""

    lengths: Tensor
    size: int
    frame_rate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lengths, Tensor):
            raise TypeError("lengths must be a Tensor.")
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise TypeError("size must be an integer.")
        if self.size < 0:
            raise ValueError("size must be non-negative.")
        if self.lengths.ndim != 1:
            raise ValueError("lengths must have shape [batch].")
        _validate_lengths(self.lengths, batch_size=self.lengths.shape[0], size=self.size)
        _validate_frame_rate(self.frame_rate)

    @property
    def batch_size(self) -> int:
        return self.lengths.shape[0]

    @property
    def mask(self) -> Tensor:
        positions = torch.arange(self.size, device=self.lengths.device)
        return positions.unsqueeze(0) < self.lengths.unsqueeze(1)


@dataclass(eq=False, frozen=True, slots=True)
class Codes:
    """Discrete payload for one named leaf stream.

    The first two dimensions are always ``[batch, time]``. Additional trailing
    dimensions are allowed so an upstream embedder can handle structured values
    such as multiple codebooks without making the layout know that structure.
    """

    name: str
    values: Tensor
    lengths: Tensor
    frame_rate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("codes name must be a non-empty string.")
        if not isinstance(self.values, Tensor):
            raise TypeError("codes values must be a Tensor.")
        if not isinstance(self.lengths, Tensor):
            raise TypeError("codes lengths must be a Tensor.")
        if self.values.ndim < 2:
            raise ValueError("codes values must have shape [batch, time, ...].")
        if not _is_integer_dtype(self.values.dtype):
            raise TypeError("codes values must use an integer dtype.")
        if self.values.device != self.lengths.device:
            raise ValueError("codes values and lengths must be on the same device.")
        _validate_lengths(
            self.lengths,
            batch_size=self.values.shape[0],
            size=self.values.shape[1],
        )
        _validate_frame_rate(self.frame_rate)

    @property
    def timeline(self) -> Timeline:
        return Timeline(
            lengths=self.lengths,
            size=self.values.shape[1],
            frame_rate=self.frame_rate,
        )


Discrete = Codes


@dataclass(eq=False, frozen=True, slots=True)
class Features:
    """Continuous payload for one named leaf stream."""

    name: str
    values: Tensor
    lengths: Tensor
    frame_rate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("feature name must be a non-empty string.")
        if not isinstance(self.values, Tensor):
            raise TypeError("feature values must be a Tensor.")
        if not isinstance(self.lengths, Tensor):
            raise TypeError("feature lengths must be a Tensor.")
        if self.values.ndim != 3:
            raise ValueError("feature values must have shape [batch, time, dim].")
        if not torch.is_floating_point(self.values):
            raise TypeError("feature values must use a floating-point dtype.")
        if self.values.device != self.lengths.device:
            raise ValueError("feature values and lengths must be on the same device.")
        _validate_lengths(
            self.lengths,
            batch_size=self.values.shape[0],
            size=self.values.shape[1],
        )
        if self.values.shape[2] <= 0:
            raise ValueError("feature values must have a positive embedding dimension.")
        _validate_frame_rate(self.frame_rate)

    @property
    def timeline(self) -> Timeline:
        return Timeline(
            lengths=self.lengths,
            size=self.values.shape[1],
            frame_rate=self.frame_rate,
        )


__all__ = ["Codes", "Discrete", "Features", "Timeline"]
