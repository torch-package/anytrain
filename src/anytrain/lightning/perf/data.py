"""Pure helpers for valid/padded data-unit accounting."""

from __future__ import annotations

import math
import re

_UNIT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def padding_ratio(*, valid_units: float, padded_units: float) -> float:
    """Return ``1 - valid/padded`` for one batch of homogeneous units."""

    valid = _require_non_negative(valid_units, "valid_units")
    padded = _require_positive(padded_units, "padded_units")
    if valid > padded:
        raise ValueError(
            f"valid_units ({valid}) cannot exceed padded_units ({padded})."
        )
    return 1.0 - valid / padded


def units_realtime(*, valid_units: float, step_time: float) -> float:
    """Valid data units consumed per wall second."""

    valid = _require_non_negative(valid_units, "valid_units")
    step_time = _require_positive(step_time, "step_time")
    return valid / step_time


def require_unit_name(unit: str) -> str:
    if not isinstance(unit, str) or not _UNIT_PATTERN.fullmatch(unit):
        raise ValueError(
            "unit must match [a-z][a-z0-9_]* "
            f"(examples: audio_seconds, tokens, frames); got {unit!r}."
        )
    return unit


def _require_positive(value: float, name: str) -> float:
    resolved = _as_float(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _require_non_negative(value: float, name: str) -> float:
    resolved = _as_float(value, name)
    if resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _as_float(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


__all__ = [
    "padding_ratio",
    "require_unit_name",
    "units_realtime",
]
