from __future__ import annotations

from torch import Tensor


def require_batch(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if x.ndim == 0:
        raise ValueError(f"{name} must include a batch dimension.")
    if x.shape[0] <= 0:
        raise ValueError(f"{name} batch size must be positive.")


def require_pair(x_0: Tensor, x_1: Tensor) -> None:
    if not isinstance(x_0, Tensor) or not isinstance(x_1, Tensor):
        raise TypeError("x_0 and x_1 must be tensors.")
    if x_0.shape != x_1.shape:
        raise ValueError(f"x_0 and x_1 must have the same shape, got {x_0.shape} and {x_1.shape}.")
    if x_0.device != x_1.device:
        raise ValueError("x_0 and x_1 must be on the same device.")
