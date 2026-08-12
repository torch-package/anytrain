from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ._validation import scalar
from .abc import LossABC
from .balancer import LossBalancerABC, MeanLossBalancer
from .observation import loss as register_loss


def _validate_loss_modules(losses: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a mapping of names to loss modules.")
    if not losses:
        raise ValueError("losses must contain at least one loss.")

    validated: dict[str, nn.Module] = {}
    for name, loss in losses.items():
        if not isinstance(name, str):
            raise TypeError("loss name must be a string.")
        if not name:
            raise ValueError("loss name must not be empty.")
        if not isinstance(loss, nn.Module):
            raise TypeError(f"Loss {name!r} must be a torch.nn.Module.")
        validated[name] = loss
    return validated


def _resolve_balancer(value: LossBalancerABC | None) -> LossBalancerABC:
    if value is None:
        return MeanLossBalancer()
    if not isinstance(value, LossBalancerABC):
        raise TypeError("balancer must inherit LossBalancerABC.")
    return value


def _compute_group_loss(
    losses: nn.ModuleDict,
    balancer: LossBalancerABC,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    values = {
        name: scalar(loss(*args, **kwargs), name=f"loss {name!r}")
        for name, loss in losses.items()
    }
    if not values:
        raise RuntimeError("loss collection has no loss modules.")
    return scalar(balancer(values), name="balanced loss")


@register_loss
class LossGroup(LossABC):
    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        balancer: LossBalancerABC | None = None,
    ) -> None:
        super().__init__()
        self.losses = nn.ModuleDict(_validate_loss_modules(losses))
        self.balancer = _resolve_balancer(balancer)

    def compute_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return _compute_group_loss(self.losses, self.balancer, *args, **kwargs)
