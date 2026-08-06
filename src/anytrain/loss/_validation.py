from __future__ import annotations

import torch


def scalar(value: object, *, name: str = "loss") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor.")
    return value


__all__ = ["scalar"]
