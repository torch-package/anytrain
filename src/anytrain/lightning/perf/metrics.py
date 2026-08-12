from __future__ import annotations

from ._validation import positive_real


def model_flops_utilization(
    *,
    model_flops_per_step: float,
    step_time: float,
    hardware_peak_flops: float,
) -> float:
    model_flops_per_step = positive_real(model_flops_per_step, "model_flops_per_step")
    step_time = positive_real(step_time, "step_time")
    hardware_peak_flops = positive_real(hardware_peak_flops, "hardware_peak_flops")
    return model_flops_per_step / step_time / hardware_peak_flops
