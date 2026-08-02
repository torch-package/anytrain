from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


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
        cosine_progress = 0.5 * (1.0 - math.cos(math.pi * resolved))
        return float(self.start) + (float(self.end) - float(self.start)) * cosine_progress


def curve_ratio(curve: LRCurve, progress: float) -> float:
    if not hasattr(curve, "ratio") or not callable(curve.ratio):
        raise TypeError("lr curve must implement ratio(progress).")
    value = float(curve.ratio(progress))
    _require_ratio(value, "lr ratio")
    return value


def _progress(value: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("curve progress must be a real number.") from exc
    if not math.isfinite(resolved):
        raise ValueError("curve progress must be finite.")
    return min(max(resolved, 0.0), 1.0)


def _require_ratio(value: float, name: str) -> None:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


__all__ = ["Constant", "Cosine", "LRCurve", "Linear", "curve_ratio"]
