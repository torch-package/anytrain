from __future__ import annotations

from torch import Tensor, nn

from .routing import validate_num_experts


class LinearRouter(nn.Module):
    """Feature-last linear projector that returns expert logits."""

    proj: nn.Linear

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer.")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        validate_num_experts(num_experts)
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.proj = nn.Linear(input_dim, num_experts, bias=bias)

    def forward(self, hidden_states: Tensor, *, mask: Tensor | None = None) -> Tensor:
        del mask
        if hidden_states.ndim < 1:
            raise ValueError("hidden_states must have at least one dimension.")
        if hidden_states.size(-1) != self.input_dim:
            raise ValueError(
                "hidden_states last dimension must match input_dim: "
                f"got {hidden_states.size(-1)}, expected {self.input_dim}."
            )
        return self.proj(hidden_states)


__all__ = ["LinearRouter"]
