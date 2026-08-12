from __future__ import annotations

import math
from numbers import Real


def strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def integer_at_least(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a host real number, not bool or Tensor.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def positive_real(value: object, name: str) -> float:
    resolved = finite_real(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def non_negative_real(value: object, name: str) -> float:
    resolved = finite_real(value, name)
    if resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


__all__ = [
    "finite_real",
    "integer_at_least",
    "non_negative_int",
    "non_negative_real",
    "positive_int",
    "positive_real",
    "strict_bool",
]
