from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from ._observation import loss as register_loss
from ._validation import scalar
from .abc import LossABC
from .balancer import LossBalancerABC, MeanLossBalancer


@register_loss
class LossGroup(LossABC):
    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        balancer: LossBalancerABC | None = None,
    ) -> None:
        super().__init__()
        validated_losses = self._validate_losses(losses)
        self.losses = cast(dict[str, nn.Module], nn.ModuleDict(validated_losses))
        self.balancer = self._validate_balancer(balancer)

    def compute_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        loss_values: dict[str, torch.Tensor] = {}
        for name, module in self.losses.items():
            loss_values[name] = scalar(
                self._call_loss(
                    module,
                    *args,
                    **kwargs,
                ),
                name=f"loss {name!r}",
            )

        if not loss_values:
            raise RuntimeError("LossGroup has no loss modules.")
        return scalar(self.balancer(loss_values), name="balanced loss")

    def _call_loss(
        self,
        module: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return module(*args, **kwargs)

    def _validate_losses(self, losses: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
        if not isinstance(losses, Mapping):
            raise TypeError("losses must be a mapping of names to loss modules.")
        if not losses:
            raise ValueError("losses must contain at least one loss.")

        validated: dict[str, nn.Module] = {}
        for raw_name, loss in losses.items():
            name = self._validate_loss_name(raw_name)
            if not isinstance(loss, nn.Module):
                raise TypeError(f"Loss {name!r} must be a torch.nn.Module.")
            validated[name] = loss
        return validated

    def _validate_balancer(self, balancer: LossBalancerABC | None) -> LossBalancerABC:
        if balancer is None:
            return MeanLossBalancer()
        if not isinstance(balancer, LossBalancerABC):
            raise TypeError("balancer must inherit LossBalancerABC.")
        return balancer

    def _validate_loss_name(self, name: object) -> str:
        if not isinstance(name, str):
            raise TypeError("loss name must be a string.")
        if not name:
            raise ValueError("loss name must not be empty.")
        return name
