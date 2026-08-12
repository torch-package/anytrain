from __future__ import annotations

import torch


def scalar(value: object, *, name: str = "loss") -> torch.Tensor:
    """Return ``value`` unchanged after enforcing the scalar-loss protocol.

    Raises:
        TypeError: If ``value`` is not a :class:`torch.Tensor`.
        ValueError: If ``value`` is not a 0-D tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor.")
    return value


__all__ = ["scalar"]
