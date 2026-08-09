from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import auto
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from anytrain._compat import StrEnum


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
        _edges(self.edges, name="Distribution")


@dataclass(frozen=True, eq=False)
class Profile:
    """One-dimensional values profiled over fixed coordinate intervals."""

    values: Tensor
    coordinates: Tensor
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("Profile.values must be a Tensor.")
        if not isinstance(self.coordinates, Tensor):
            raise TypeError("Profile.coordinates must be a Tensor.")
        if self.values.ndim != 1 or self.coordinates.ndim != 1:
            raise ValueError("Profile values and coordinates must both be 1D.")
        if self.values.shape != self.coordinates.shape:
            raise ValueError("Profile values and coordinates must have the same shape.")
        if self.values.numel() == 0:
            raise ValueError("Profile values and coordinates must be non-empty.")
        if not self.values.dtype.is_floating_point:
            raise TypeError("Profile.values must use a floating dtype.")
        if not self.coordinates.dtype.is_floating_point:
            raise TypeError("Profile.coordinates must use a floating dtype.")
        if self.values.device != self.coordinates.device:
            raise ValueError("Profile values and coordinates must be on the same device.")
        _edges(self.edges, name="Profile")


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


@dataclass(frozen=True, eq=False)
class Heatmap:
    """One semantically selected two-dimensional continuous view.

    The module must select sample, head, and any other structured axes before
    constructing the heatmap. ``value_range=None`` scales each rendered image
    by its finite minimum and maximum; a fixed range keeps colors comparable
    across optimizer steps.
    """

    values: Tensor
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("Heatmap.values must be a Tensor.")
        if self.values.ndim != 2:
            raise ValueError(
                "Heatmap.values must be 2D after the module selects all structured "
                f"axes, got shape {tuple(self.values.shape)}."
            )
        if not self.values.dtype.is_floating_point:
            raise TypeError("Heatmap.values must use a floating dtype.")
        if any(size <= 0 for size in self.values.shape):
            raise ValueError("Heatmap.values dimensions must be non-empty.")
        if self.value_range is not None:
            lower, upper = _value_range(self.value_range)
            if upper <= lower:
                raise ValueError("Heatmap.value_range must be strictly increasing.")


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


class Reduction(StrEnum):
    """How repeated observations from one source are folded in a commit window."""

    Last = auto()
    Mean = auto()
    Merge = auto()


Observation = Curve | Distribution | Profile | Bins | Heatmap
ObservationResult = Observation | Mapping[str, Observation] | None
OutputCollector = Callable[
    [nn.Module, tuple[Any, ...], Mapping[str, Any], object],
    ObservationResult,
]
BackwardCollector = Callable[
    [nn.Module, tuple[Tensor | None, ...]],
    ObservationResult,
]
StateCollector = Callable[[nn.Module], ObservationResult]


class BackwardObservationWarning(UserWarning):
    """Backward module observation is declared but not implemented yet."""


@dataclass(frozen=True)
class OutputObservation:
    name: str
    collect: OutputCollector
    reduction: Reduction
    history: EMAHistory | None = None
    recommended: bool = False
    covers_descendants: bool = False

    def __post_init__(self) -> None:
        _spec(self.name, self.collect)
        _reduction(self.reduction, self.history)


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
class StateObservation:
    name: str
    collect: StateCollector
    history: EMAHistory | None = None
    recommended: bool = False
    covers_descendants: bool = False

    def __post_init__(self) -> None:
        _spec(self.name, self.collect)


@dataclass(frozen=True)
class ForwardEvent:
    """A named local value published explicitly from ``module.forward``."""

    name: str
    reduction: Reduction
    history: EMAHistory | None = None
    recommended: bool = False
    covers_descendants: bool = False

    def __post_init__(self) -> None:
        _name(self.name)
        _reduction(self.reduction, self.history)


# Compatibility aliases. The canonical names describe where the value lives:
# module state or the observable forward input/output boundary.
ForwardObservation = OutputObservation
PollObservation = StateObservation


ObservationSpec = (
    OutputObservation
    | BackwardObservation
    | StateObservation
    | ForwardEvent
)


def _edges(values: tuple[float, ...], *, name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name}.edges must contain at least two values.")
    previous: float | None = None
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}.edges must contain real numbers.")
        current = float(value)
        if not math.isfinite(current):
            raise ValueError(f"{name}.edges must contain finite values.")
        if previous is not None and current <= previous:
            raise ValueError(f"{name}.edges must be strictly increasing.")
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


def _value_range(values: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise TypeError("Heatmap.value_range must be a pair of real numbers.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Heatmap.value_range must contain real numbers.")
        current = float(value)
        if not math.isfinite(current):
            raise ValueError("Heatmap.value_range must contain finite values.")
        result.append(current)
    return result[0], result[1]


def _spec(name: str, collect: object) -> None:
    _name(name)
    if not callable(collect):
        raise TypeError("observation collector must be callable.")


def _reduction(reduction: Reduction, history: EMAHistory | None) -> None:
    if not isinstance(reduction, Reduction):
        raise TypeError("observation reduction must be a Reduction.")
    if reduction == Reduction.Merge:
        if history is None:
            raise ValueError("Merge observation reduction requires EMAHistory.")
        return
    if history is not None:
        raise ValueError("only Merge observation reduction supports EMAHistory.")


def _name(name: str) -> None:
    if not isinstance(name, str) or not name.strip("/"):
        raise ValueError("observation name must be a non-empty string.")


__all__ = [
    "BackwardObservation",
    "BackwardObservationWarning",
    "Bins",
    "Curve",
    "Distribution",
    "EMAHistory",
    "ForwardEvent",
    "Heatmap",
    "Observation",
    "ObservationResult",
    "ObservationSpec",
    "OutputObservation",
    "Profile",
    "Reduction",
    "StateObservation",
]
