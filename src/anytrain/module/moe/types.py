from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(eq=False)
class RouterOutput:
    """Sparse router output with feature-agnostic leading dimensions.

    logits and probabilities are shaped (..., num_experts). indices and
    weights are shaped (..., top_k) and describe the selected experts plus
    combine weights for each routed token.
    """

    logits: Tensor
    probabilities: Tensor
    indices: Tensor
    weights: Tensor
    mask: Tensor | None = None

    @property
    def num_experts(self) -> int:
        return int(self.probabilities.shape[-1])

    @property
    def top_k(self) -> int:
        return int(self.indices.shape[-1])


@dataclass(eq=False)
class MoEOutput:
    hidden_states: Tensor
    routing: RouterOutput
    tokens_per_expert: Tensor
    dropped_mask: Tensor | None = None


__all__ = ["MoEOutput", "RouterOutput"]
