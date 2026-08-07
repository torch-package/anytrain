from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ._group import balancer as loss_balancer
from ._group import compute, module_dict
from ._observation import loss as register_loss
from .abc import LossABC
from .balancer import LossBalancerABC


@register_loss
class LossGroup(LossABC):
    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        balancer: LossBalancerABC | None = None,
    ) -> None:
        super().__init__()
        self.losses = module_dict(losses)
        self.balancer = loss_balancer(balancer)

    def compute_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return compute(self.losses, self.balancer, *args, **kwargs)
