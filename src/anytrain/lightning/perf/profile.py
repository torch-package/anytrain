from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def profile_forward_flops(
    module: torch.nn.Module,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> float:
    call_kwargs = {} if kwargs is None else dict(kwargs)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad(), torch.profiler.profile(with_flops=True) as profiler:
            module(*args, **call_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        module.train(was_training)

    flops = sum(float(getattr(event, "flops", 0) or 0) for event in profiler.key_averages())
    if flops <= 0:
        raise RuntimeError(
            "PyTorch profiler did not report forward FLOPs. "
            "Pass model_flops_per_step explicitly when the model uses unsupported ops."
        )
    return flops


def training_flops_from_forward(
    forward_flops: float,
    *,
    backward_multiplier: float = 2.0,
) -> float:
    forward_flops = _require_positive(forward_flops, "forward_flops")
    try:
        backward_multiplier = float(backward_multiplier)
    except (TypeError, ValueError) as exc:
        raise TypeError("backward_multiplier must be a real number.") from exc
    if not math.isfinite(backward_multiplier) or backward_multiplier < 0:
        raise ValueError("backward_multiplier must be finite and non-negative.")
    return forward_flops * (1.0 + backward_multiplier)


def _require_positive(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved
