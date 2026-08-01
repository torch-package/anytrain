from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..routing import SparseExpertRouting


@dataclass(eq=False)
class RouterOutput(SparseExpertRouting):
    """MoE-compatible sparse expert routing output."""


@dataclass(eq=False)
class MoEOutput:
    hidden_states: Tensor
    routing: SparseExpertRouting
    tokens_per_expert: Tensor
    dropped_mask: Tensor | None = None


__all__ = ["MoEOutput", "RouterOutput"]
