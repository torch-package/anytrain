from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ._validation import non_negative_real, positive_real


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    if not isinstance(trainable_only, bool):
        raise TypeError("trainable_only must be a boolean.")
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
    forward_flops = positive_real(forward_flops, "forward_flops")
    backward_multiplier = non_negative_real(backward_multiplier, "backward_multiplier")
    return forward_flops * (1.0 + backward_multiplier)
