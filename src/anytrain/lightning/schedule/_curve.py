from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from anytrain._lr_math import cosine_interpolate


class LRCurve(Protocol):
    def ratio(self, progress: float) -> float: ...


@dataclass(frozen=True)
class Constant:
    value: float = 1.0

    def __post_init__(self) -> None:
        _require_ratio(self.value, "value")

    def ratio(self, progress: float) -> float:
        _progress(progress)
        return float(self.value)


@dataclass(frozen=True)
class Linear:
    start: float = 0.0
    end: float = 1.0

    def __post_init__(self) -> None:
        _require_ratio(self.start, "start")
        _require_ratio(self.end, "end")

    def ratio(self, progress: float) -> float:
        resolved = _progress(progress)
        return float(self.start) + (float(self.end) - float(self.start)) * resolved


@dataclass(frozen=True)
class Cosine:
    start: float = 1.0
    end: float = 0.1

    def __post_init__(self) -> None:
        _require_ratio(self.start, "start")
        _require_ratio(self.end, "end")

    def ratio(self, progress: float) -> float:
        resolved = _progress(progress)
        return cosine_interpolate(resolved, start=float(self.start), end=float(self.end))


def curve_ratio(curve: LRCurve, progress: float) -> float:
    if not hasattr(curve, "ratio") or not callable(curve.ratio):
        raise TypeError("lr curve must implement ratio(progress).")
    return _require_ratio(curve.ratio(progress), "lr ratio")


def _progress(value: float) -> float:
    resolved = _number(value, "curve progress")
    if not math.isfinite(resolved):
        raise ValueError("curve progress must be finite.")
    return min(max(resolved, 0.0), 1.0)


def _require_ratio(value: float, name: str) -> float:
    resolved = _number(value, name)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a host real number, not bool or Tensor.")
    return float(value)


__all__ = ["Constant", "Cosine", "LRCurve", "Linear", "curve_ratio"]
