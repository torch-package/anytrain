from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor, nn

from .metrics import load_balance_loss, router_z_loss
from .routing import SparseExpertRouting
from .types import MoEOutput


@dataclass(frozen=True)
class SparseMoELossOutput:
    """Explicit sparse-MoE auxiliary loss components.

    ``total`` is the coefficient-weighted scalar that downstream training code
    can add to its task objective. The unweighted and weighted component fields
    stay available for logging without re-running the router.
    """

    total: Tensor
    load_balance: Tensor
    router_z: Tensor
    weighted_load_balance: Tensor
    weighted_router_z: Tensor

    def details(self) -> dict[str, Tensor]:
        return {
            "load_balance": self.load_balance.detach(),
            "router_z": self.router_z.detach(),
            "weighted_load_balance": self.weighted_load_balance.detach(),
            "weighted_router_z": self.weighted_router_z.detach(),
            "total": self.total.detach(),
        }


def sparse_moe_aux_loss(
    source: SparseExpertRouting | MoEOutput,
    *,
    load_balance_weight: float = 1.0,
    router_z_weight: float = 0.0,
) -> SparseMoELossOutput:
    """Compute sparse-MoE auxiliary losses without mutating model outputs."""

    _non_negative_float(load_balance_weight, "load_balance_weight")
    _non_negative_float(router_z_weight, "router_z_weight")
    routing = _routing(source)
    load_balance = load_balance_loss(routing)
    z_loss = router_z_loss(routing)
    weighted_load_balance = load_balance * load_balance_weight
    weighted_router_z = z_loss * router_z_weight
    return SparseMoELossOutput(
        total=weighted_load_balance + weighted_router_z,
        load_balance=load_balance,
        router_z=z_loss,
        weighted_load_balance=weighted_load_balance,
        weighted_router_z=weighted_router_z,
    )


def sparse_moe_loss_metrics(
    loss: SparseMoELossOutput,
    *,
    prefix: str = "loss",
) -> dict[str, Tensor]:
    prefix = prefix.strip("/")
    if prefix:
        prefix = f"{prefix}/"
    return {
        f"{prefix}total": loss.total.detach(),
        f"{prefix}load_balance": loss.load_balance.detach(),
        f"{prefix}router_z": loss.router_z.detach(),
        f"{prefix}weighted_load_balance": loss.weighted_load_balance.detach(),
        f"{prefix}weighted_router_z": loss.weighted_router_z.detach(),
    }


class SparseMoELoss(nn.Module):
    """Small ``nn.Module`` wrapper for explicit sparse-MoE aux loss wiring."""

    def __init__(
        self,
        *,
        load_balance_weight: float = 1.0,
        router_z_weight: float = 0.0,
        return_details: bool = False,
    ) -> None:
        super().__init__()
        _non_negative_float(load_balance_weight, "load_balance_weight")
        _non_negative_float(router_z_weight, "router_z_weight")
        self.load_balance_weight = float(load_balance_weight)
        self.router_z_weight = float(router_z_weight)
        self.return_details = return_details

    def forward(
        self,
        source: SparseExpertRouting | MoEOutput,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        loss = sparse_moe_aux_loss(
            source,
            load_balance_weight=self.load_balance_weight,
            router_z_weight=self.router_z_weight,
        )
        if not self.return_details:
            return loss.total
        return loss.total, loss.details()


def _routing(source: SparseExpertRouting | MoEOutput) -> SparseExpertRouting:
    if isinstance(source, SparseExpertRouting):
        return source
    if isinstance(source, MoEOutput):
        return source.routing
    raise TypeError("source must be SparseExpertRouting or MoEOutput.")


def _non_negative_float(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a non-negative number.")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")


__all__ = [
    "SparseMoELoss",
    "SparseMoELossOutput",
    "sparse_moe_aux_loss",
    "sparse_moe_loss_metrics",
]
