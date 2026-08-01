from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .routing import ExpertRouting


@dataclass(eq=False)
class MoEOutput:
    hidden_states: Tensor
    routing: ExpertRouting
    tokens_per_expert: Tensor
    dropped_mask: Tensor | None = None


__all__ = ["MoEOutput"]
