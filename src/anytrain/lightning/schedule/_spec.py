from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._curve import LRCurve, curve_ratio
from ._data import UnitProvider, require_unit_name


@dataclass(frozen=True)
class StepClock:
    """Use Lightning's optimizer-step clock for this schedule."""


@dataclass(frozen=True)
class UnitClock:
    """Use cumulative global data units for this schedule."""

    unit: str
    provider: UnitProvider
    log_every_n_units: float | None = None
    measure_window_batches: int = 100
    sync_cuda: bool = True
    sync_distributed: bool = True
    allow_external_lr_changes: bool = False

    def __post_init__(self) -> None:
        require_unit_name(self.unit)
        if not callable(self.provider):
            raise TypeError("unit provider must be callable.")
        if self.log_every_n_units is not None:
            _require_positive(self.log_every_n_units, "log_every_n_units")
        _require_positive_int(self.measure_window_batches, "measure_window_batches")
        if not isinstance(self.sync_cuda, bool):
            raise TypeError("sync_cuda must be a boolean.")
        if not isinstance(self.sync_distributed, bool):
            raise TypeError("sync_distributed must be a boolean.")
        if not isinstance(self.allow_external_lr_changes, bool):
            raise TypeError("allow_external_lr_changes must be a boolean.")


Clock = StepClock | UnitClock


@dataclass(frozen=True)
class Phase:
    name: str
    duration: float
    lr: LRCurve

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("phase name must be a non-empty string.")
        if "/" in self.name:
            raise ValueError("phase name must not contain '/'.")
        _require_positive(self.duration, "phase duration")
        curve_ratio(self.lr, 0.0)


@dataclass(frozen=True)
class LRSchedule:
    clock: Clock
    phases: Sequence[Phase]
    stop_at: float | None = None
    stop_at_end: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.clock, (StepClock, UnitClock)):
            raise TypeError("schedule clock must be StepClock or UnitClock.")
        if isinstance(self.phases, (str, bytes)) or not isinstance(self.phases, Sequence):
            raise TypeError("schedule phases must be a sequence.")
        if not self.phases:
            raise ValueError("schedule phases must contain at least one phase.")
        for phase in self.phases:
            if not isinstance(phase, Phase):
                raise TypeError("schedule phases must contain Phase values.")
        if self.stop_at is not None:
            _require_positive(self.stop_at, "stop_at")
        if not isinstance(self.stop_at_end, bool):
            raise TypeError("stop_at_end must be a boolean.")
        if isinstance(self.clock, StepClock) and (self.stop_at is not None or self.stop_at_end):
            raise ValueError("step-clock schedules use Lightning max_steps for stopping.")

    @property
    def total_duration(self) -> float:
        return math.fsum(float(phase.duration) for phase in self.phases)

    def milestones(self) -> tuple[tuple[str, float], ...]:
        total = 0.0
        milestones: list[tuple[str, float]] = []
        for phase in self.phases:
            total += float(phase.duration)
            milestones.append((phase.name, total))
        return tuple(milestones)

    def stop_unit(self) -> float | None:
        if self.stop_at is not None:
            return float(self.stop_at)
        if self.stop_at_end:
            return self.total_duration
        return None

    def lr_ratio(self, position: float) -> float:
        value = _require_non_negative(position, "schedule position")
        start = 0.0
        for phase in self.phases:
            end = start + float(phase.duration)
            if value <= end:
                progress = (value - start) / float(phase.duration)
                return curve_ratio(phase.lr, progress)
            start = end
        return curve_ratio(self.phases[-1].lr, 1.0)

    def fingerprint(self) -> str:
        clock = "unit" if isinstance(self.clock, UnitClock) else "step"
        unit = self.clock.unit if isinstance(self.clock, UnitClock) else ""
        phases = tuple(
            (phase.name, float(phase.duration), type(phase.lr).__name__, repr(phase.lr))
            for phase in self.phases
        )
        return repr((clock, unit, phases, self.stop_at, self.stop_at_end))


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_positive(value: float, name: str) -> float:
    resolved = _number(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _require_non_negative(value: float, name: str) -> float:
    resolved = _number(value, name)
    if resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _number(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


__all__ = ["Clock", "LRSchedule", "Phase", "StepClock", "UnitClock"]
