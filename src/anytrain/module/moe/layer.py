from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from ..routing import SparseExpertRouting
from .dispatch import dispatch_to_experts
from .types import MoEOutput


class TokenChoiceMoE(nn.Module):
    """Task-agnostic token-choice MoE wrapper over local expert modules."""

    router: nn.Module
    experts: nn.ModuleList

    def __init__(self, *, router: nn.Module, experts: Sequence[nn.Module]) -> None:
        super().__init__()
        if len(experts) <= 0:
            raise ValueError("experts must contain at least one module.")
        self.router = router
        self.experts = nn.ModuleList(experts)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        **router_kwargs: Any,
    ) -> MoEOutput:
        routing = self.router(hidden_states, mask=mask, **router_kwargs)
        if not isinstance(routing, SparseExpertRouting):
            raise TypeError("router must return a SparseExpertRouting.")
        return self.forward_manually(hidden_states, routing)

    def forward_manually(self, hidden_states: Tensor, routing: SparseExpertRouting) -> MoEOutput:
        hidden, tokens_per_expert = dispatch_to_experts(hidden_states, routing, self.experts)
        return MoEOutput(
            hidden_states=hidden,
            routing=routing,
            tokens_per_expert=tokens_per_expert,
            dropped_mask=routing.weights.sum(dim=-1) <= 0,
        )


MoELayer = TokenChoiceMoE


__all__ = ["MoELayer", "TokenChoiceMoE"]
