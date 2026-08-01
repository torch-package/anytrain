from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor

from .metrics import load_balance_loss, router_z_loss
from .routing import ExpertRouting
from .types import MoEOutput


@dataclass(frozen=True)
class MoELossOutput:
    """Explicit MoE auxiliary loss components.

    The total field is the coefficient-weighted scalar that downstream training code
    can add to its task objective. The unweighted and weighted component fields
    stay available for logging without re-running the router.
    """

    total: Tensor
    balance_loss: Tensor
    z_loss: Tensor
    weighted_balance: Tensor
    weighted_z: Tensor

    def details(self) -> dict[str, Tensor]:
        return {
            "balance": self.balance_loss.detach(),
            "z": self.z_loss.detach(),
            "weighted_balance": self.weighted_balance.detach(),
            "weighted_z": self.weighted_z.detach(),
            "total": self.total.detach(),
        }


def moe_loss(
    source: ExpertRouting | MoEOutput,
    *,
    balance: float = 1.0,
    z: float = 0.0,
) -> MoELossOutput:
    """Compute MoE auxiliary losses without mutating model outputs."""

    _non_negative_float(balance, "balance")
    _non_negative_float(z, "z")
    routing = _routing(source)
    balance_value = load_balance_loss(routing)
    z_value = router_z_loss(routing)
    weighted_balance = balance_value * balance
    weighted_z = z_value * z
    return MoELossOutput(
        total=weighted_balance + weighted_z,
        balance_loss=balance_value,
        z_loss=z_value,
        weighted_balance=weighted_balance,
        weighted_z=weighted_z,
    )


def _routing(source: ExpertRouting | MoEOutput) -> ExpertRouting:
    if isinstance(source, ExpertRouting):
        return source
    if isinstance(source, MoEOutput):
        return source.routing
    raise TypeError("source must be ExpertRouting or MoEOutput.")


def _non_negative_float(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a non-negative number.")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")


__all__ = [
    "MoELossOutput",
    "moe_loss",
]
