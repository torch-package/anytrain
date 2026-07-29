from __future__ import annotations

import math


def model_flops_utilization(
    *,
    model_flops_per_step: float,
    step_time: float,
    hardware_peak_flops: float,
) -> float:
    model_flops_per_step = _require_positive(model_flops_per_step, "model_flops_per_step")
    step_time = _require_positive(step_time, "step_time")
    hardware_peak_flops = _require_positive(hardware_peak_flops, "hardware_peak_flops")
    return model_flops_per_step / step_time / hardware_peak_flops


def _require_positive(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved
