from __future__ import annotations


def model_flops_utilization(
    *,
    model_flops_per_step: float,
    step_time: float,
    hardware_peak_flops: float,
) -> float:
    _require_positive(model_flops_per_step, "model_flops_per_step")
    _require_positive(step_time, "step_time")
    _require_positive(hardware_peak_flops, "hardware_peak_flops")
    return model_flops_per_step / step_time / hardware_peak_flops


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
