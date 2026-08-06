from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn


@dataclass(frozen=True, eq=False)
class Curve:
    """One scalar value that becomes a curve over optimizer steps."""

    value: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor):
            raise TypeError("Curve.value must be a Tensor.")
        if self.value.ndim != 0:
            raise ValueError(
                f"Curve.value must be scalar, got shape {tuple(self.value.shape)}."
            )
        if self.value.dtype.is_complex or self.value.dtype == torch.bool:
            raise TypeError("Curve.value must be a real numeric tensor.")


@dataclass(frozen=True, eq=False)
class Distribution:
    """One-dimensional continuous observations with fixed histogram edges."""

    values: Tensor
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("Distribution.values must be a Tensor.")
        if self.values.ndim != 1:
            raise ValueError(
                "Distribution.values must be 1D; structured multidimensional continuous "
                f"tensors require an explicit view, got shape {tuple(self.values.shape)}."
            )
        if not self.values.dtype.is_floating_point:
            raise TypeError("Distribution.values must use a floating dtype.")
        _edges(self.edges)


@dataclass(frozen=True, eq=False)
class Bins:
    """Discrete observations with optional grouping and activity masks.

    ``values`` may have any rank. With one bin space, every dimension is an
    observation dimension. With multiple bin spaces, ``group_dim`` identifies
    the axis whose entries select the corresponding value from ``sizes``.
    """

    values: Tensor
    sizes: tuple[int, ...]
    group_dim: int | None = None
    active_mask: Tensor | None = None
    group_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("Bins.values must be a Tensor.")
        if self.values.dtype.is_floating_point or self.values.dtype.is_complex:
            raise TypeError("Bins.values must use an integer or bool dtype.")
        sizes = _sizes(self.sizes)
        if self.group_dim is None:
            if len(sizes) != 1:
                raise ValueError("Bins with multiple sizes require group_dim.")
        else:
            if self.values.ndim == 0:
                raise ValueError("Bins.group_dim requires values with at least one dimension.")
            dim = self.group_dim
            if dim < -self.values.ndim or dim >= self.values.ndim:
                raise ValueError(
                    f"Bins.group_dim={dim} is invalid for shape {tuple(self.values.shape)}."
                )
            observed = self.values.shape[dim]
            if observed <= 0 or observed > len(sizes):
                raise ValueError(
                    "Bins group axis must contain between one and len(sizes) groups: "
                    f"got {observed} and {len(sizes)}."
                )
        if self.active_mask is not None:
            if not isinstance(self.active_mask, Tensor):
                raise TypeError("Bins.active_mask must be a Tensor or None.")
            if self.active_mask.shape != self.values.shape:
                raise ValueError(
                    "Bins.active_mask must match values shape: "
                    f"got {tuple(self.active_mask.shape)} and {tuple(self.values.shape)}."
                )
            if self.active_mask.device != self.values.device:
                raise ValueError("Bins.active_mask must be on the same device as values.")
            if self.active_mask.dtype != torch.bool:
                raise TypeError("Bins.active_mask must use bool dtype.")
        if self.group_names is not None:
            if len(self.group_names) != len(sizes):
                raise ValueError("Bins.group_names must align with sizes.")
            if any(
                not isinstance(name, str) or not name.strip("/")
                for name in self.group_names
            ):
                raise ValueError("Bins.group_names must contain non-empty strings.")
            normalized = tuple(name.strip("/") for name in self.group_names)
            if len(set(normalized)) != len(normalized):
                raise ValueError("Bins.group_names must be unique after normalization.")


@dataclass(frozen=True)
class EMAHistory:
    """Exponentially decayed history measured in optimizer-step half-life."""

    half_life_steps: float = 1_000.0

    def __post_init__(self) -> None:
        value = self.half_life_steps
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("EMAHistory.half_life_steps must be a real number.")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("EMAHistory.half_life_steps must be finite and positive.")


Observation = Union[Curve, Distribution, Bins]
ObservationResult = Union[Observation, Mapping[str, Observation], None]
ForwardCollector = Callable[
    [nn.Module, tuple[Any, ...], Mapping[str, Any], object],
    ObservationResult,
]
BackwardCollector = Callable[
    [nn.Module, tuple[Optional[Tensor], ...]],
    ObservationResult,
]
PollCollector = Callable[[nn.Module], ObservationResult]


class BackwardObservationWarning(UserWarning):
    """Backward module observation is declared but not implemented yet."""


@dataclass(frozen=True)
class ForwardObservation:
    name: str
    collect: ForwardCollector
    history: EMAHistory | None = None
    recommended: bool = False
    covers_descendants: bool = False

    def __post_init__(self) -> None:
        _spec(self.name, self.collect)


@dataclass(frozen=True)
class BackwardObservation:
    """Reserved declaration for a future strictly read-only backward observer.

    ``ObservationCallback`` currently warns and skips this spec. Use
    ``GradientProbeLoggerCallback`` for loss-to-parameter gradient diagnostics.
    """

    name: str
    collect: BackwardCollector
    history: EMAHistory | None = None
    recommended: bool = False
    covers_descendants: bool = False

    def __post_init__(self) -> None:
        _spec(self.name, self.collect)


@dataclass(frozen=True)
class PollObservation:
    name: str
    collect: PollCollector
    history: EMAHistory | None = None
    recommended: bool = False
    covers_descendants: bool = False

    def __post_init__(self) -> None:
        _spec(self.name, self.collect)


ObservationSpec = Union[ForwardObservation, BackwardObservation, PollObservation]


def _edges(values: tuple[float, ...]) -> None:
    if len(values) < 2:
        raise ValueError("Distribution.edges must contain at least two values.")
    previous: float | None = None
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Distribution.edges must contain real numbers.")
        current = float(value)
        if not math.isfinite(current):
            raise ValueError("Distribution.edges must contain finite values.")
        if previous is not None and current <= previous:
            raise ValueError("Distribution.edges must be strictly increasing.")
        previous = current


def _sizes(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("Bins.sizes must not be empty.")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Bins.sizes must contain integers.")
        if value <= 0:
            raise ValueError("Bins.sizes must contain positive values.")
    return values


def _spec(name: str, collect: object) -> None:
    if not isinstance(name, str) or not name.strip("/"):
        raise ValueError("observation name must be a non-empty string.")
    if not callable(collect):
        raise TypeError("observation collector must be callable.")


__all__ = [
    "BackwardObservation",
    "BackwardObservationWarning",
    "Bins",
    "Curve",
    "Distribution",
    "EMAHistory",
    "ForwardObservation",
    "Observation",
    "ObservationResult",
    "ObservationSpec",
    "PollObservation",
]
