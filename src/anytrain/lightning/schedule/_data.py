from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

import lightning.pytorch as pl

_UNIT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class UnitBatch:
    """Homogeneous valid/padded units observed for one train batch.

    ``valid`` is the unit count that should advance the training clock. ``padded``
    is optional and, when present, uses the same unit for padding/throughput logs.
    """

    valid: float
    unit: str
    padded: float | None = None

    def __post_init__(self) -> None:
        require_unit_name(self.unit)
        valid = _require_non_negative(self.valid, "valid")
        if self.padded is None:
            return
        padded = _require_non_negative(self.padded, "padded")
        if valid > padded:
            raise ValueError(f"valid ({valid}) cannot exceed padded ({padded}).")


class UnitProvider(Protocol):
    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> UnitBatch: ...


def padding_ratio(*, valid_units: float, padded_units: float) -> float:
    valid = _require_non_negative(valid_units, "valid_units")
    padded = _require_positive(padded_units, "padded_units")
    if valid > padded:
        raise ValueError(f"valid_units ({valid}) cannot exceed padded_units ({padded}).")
    return 1.0 - valid / padded


def units_per_second(*, valid_units: float, step_time: float) -> float:
    valid = _require_non_negative(valid_units, "valid_units")
    elapsed = _require_positive(step_time, "step_time")
    return valid / elapsed


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
    "UnitBatch",
    "UnitProvider",
    "padding_ratio",
    "require_unit_name",
    "units_per_second",
]
