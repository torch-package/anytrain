from __future__ import annotations

import inspect

import torch
from torch import Tensor, nn

from .types import RouterOutput


class TopKRouter(nn.Module):
    """Linear top-k router for token-choice MoE layers.

    The router projects feature-last hidden states to expert logits, optionally
    applies a calibration/gating module such as AdaptiveDirichletTempering,
    then selects the top-k experts and renormalizes their combine weights.
    """

    gate: nn.Module | None

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        top_k: int = 1,
        gate: nn.Module | None = None,
        bias: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k must be <= num_experts: got top_k={top_k}, "
                f"num_experts={num_experts}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.eps = eps
        self.proj = nn.Linear(input_dim, num_experts, bias=bias)
        self.gate = gate
        self._gate_accepts_mask = _module_accepts_keyword(gate, "mask")
        self._gate_accepts_collect_stats = _module_accepts_keyword(gate, "collect_stats")

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        collect_stats: bool | None = None,
    ) -> RouterOutput:
        self._validate_hidden(hidden_states)
        logits = self.proj(hidden_states)
        broadcast_mask = None
        if mask is not None:
            broadcast_mask = _broadcast_mask(
                mask,
                leading_shape=hidden_states.shape[:-1],
                device=hidden_states.device,
            )

        probabilities = self._gate_logits(
            logits,
            mask=broadcast_mask,
            collect_stats=collect_stats,
        )
        self._validate_probabilities(probabilities, logits=logits)

        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if broadcast_mask is not None:
            weights = torch.where(broadcast_mask.unsqueeze(-1), weights, torch.zeros_like(weights))

        return RouterOutput(
            logits=logits,
            probabilities=probabilities,
            indices=indices,
            weights=weights.to(dtype=probabilities.dtype),
            mask=broadcast_mask,
        )

    def _gate_logits(
        self,
        logits: Tensor,
        *,
        mask: Tensor | None,
        collect_stats: bool | None,
    ) -> Tensor:
        if self.gate is None:
            return logits.softmax(dim=-1)
        kwargs = {}
        if mask is not None and self._gate_accepts_mask:
            kwargs["mask"] = mask
        if self._gate_accepts_collect_stats:
            kwargs["collect_stats"] = collect_stats
        return self.gate(logits, **kwargs)

    def _validate_hidden(self, hidden_states: Tensor) -> None:
        if hidden_states.ndim < 1:
            raise ValueError("hidden_states must have at least one dimension.")
        if hidden_states.size(-1) != self.input_dim:
            raise ValueError(
                "hidden_states last dimension must match input_dim: "
                f"got {hidden_states.size(-1)}, expected {self.input_dim}."
            )

    def _validate_probabilities(self, probabilities: Tensor, *, logits: Tensor) -> None:
        if probabilities.shape != logits.shape:
            raise ValueError(
                "gate must return probabilities with the same shape as logits: "
                f"got {tuple(probabilities.shape)} and {tuple(logits.shape)}."
            )
        if probabilities.device != logits.device:
            raise ValueError(
                "gate probabilities must be on the same device as logits: "
                f"got {probabilities.device} and {logits.device}."
            )
        if not torch.is_floating_point(probabilities) or torch.is_complex(probabilities):
            raise TypeError("gate probabilities must be real floating-point values.")


def _module_accepts_keyword(module: nn.Module | None, keyword: str) -> bool:
    if module is None:
        return False
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )


def _broadcast_mask(mask: Tensor, *, leading_shape: torch.Size, device: torch.device) -> Tensor:
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a bool tensor, got dtype={mask.dtype}.")
    if mask.device != device:
        raise ValueError(
            f"mask must be on the same device as hidden_states: got {mask.device} and {device}."
        )
    try:
        return torch.broadcast_to(mask, leading_shape)
    except RuntimeError as exc:
        raise ValueError(
            "mask must be broadcastable to hidden_states.shape[:-1]: "
            f"got mask shape {tuple(mask.shape)}, leading shape {tuple(leading_shape)}."
        ) from exc


LinearTopKRouter = TopKRouter


__all__ = ["LinearTopKRouter", "TopKRouter"]
