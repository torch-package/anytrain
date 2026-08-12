from __future__ import annotations

import math
from numbers import Real


def positive_real(value: float, name: str) -> float:
    resolved = _host_real(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def non_negative_real(value: float, name: str) -> float:
    resolved = _host_real(value, name)
    if resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _host_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a host real number, not bool or Tensor.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved
