from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ._validation import scalar
from .balancer import LossBalancerABC, MeanLossBalancer


def modules(losses: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a mapping of names to loss modules.")
    if not losses:
        raise ValueError("losses must contain at least one loss.")

    validated: dict[str, nn.Module] = {}
    for raw_name, loss in losses.items():
        if not isinstance(raw_name, str):
            raise TypeError("loss name must be a string.")
        if not raw_name:
            raise ValueError("loss name must not be empty.")
        if not isinstance(loss, nn.Module):
            raise TypeError(f"Loss {raw_name!r} must be a torch.nn.Module.")
        validated[raw_name] = loss
    return validated


def balancer(value: LossBalancerABC | None) -> LossBalancerABC:
    if value is None:
        return MeanLossBalancer()
    if not isinstance(value, LossBalancerABC):
        raise TypeError("balancer must inherit LossBalancerABC.")
    return value


def compute(
    losses: nn.ModuleDict,
    loss_balancer: LossBalancerABC,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    values: dict[str, torch.Tensor] = {}
    for name, module in losses.items():
        values[name] = scalar(
            module(*args, **kwargs),
            name=f"loss {name!r}",
        )
    if not values:
        raise RuntimeError("loss collection has no loss modules.")
    return scalar(loss_balancer(values), name="balanced loss")


def module_dict(losses: Mapping[str, nn.Module]) -> nn.ModuleDict:
    return nn.ModuleDict(modules(losses))
