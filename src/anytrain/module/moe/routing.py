"""Density-neutral expert routing contracts for MoE-family components.

Routing records carry full logits/probabilities for every forward. Dense
mixture (`top_k=None`) and sparse top-k (`top_k=K`) share one public type;
expert banks choose an efficient execution path from the materialized fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(eq=False)
class DenseExpertWeights:
    """Dense mixture weights over experts with expert axis on the last dimension."""

    weights: Tensor

    @property
    def num_experts(self) -> int:
        return int(self.weights.shape[-1])

    @property
    def leading_shape(self) -> torch.Size:
        return self.weights.shape[:-1]

    def flatten(self) -> Tensor:
        return self.weights.reshape(-1, self.num_experts)


@dataclass(eq=False)
class ExpertRouting:
    """Density-neutral expert routing record.

    ``logits`` and ``probabilities`` always describe the full expert
    distribution with shape ``(..., num_experts)``. When ``top_k`` is set,
    ``indices`` / ``weights`` (selected combine weights) have shape
    ``(..., top_k)``. Dense mixture leaves those selected fields as ``None``.
    """

    logits: Tensor
    probabilities: Tensor
    indices: Tensor | None = None
    weights: Tensor | None = None
    mask: Tensor | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        has_indices = self.indices is not None
        has_weights = self.weights is not None
        if has_indices != has_weights:
            raise ValueError("routing.indices and routing.weights must both be set or both None.")
        if not has_indices:
            if self.top_k is not None:
                raise ValueError("dense routing requires top_k=None when indices/weights are absent.")
            return
        assert self.indices is not None
        inferred = int(self.indices.shape[-1])
        if self.top_k is None:
            self.top_k = inferred
        elif self.top_k != inferred:
            raise ValueError(
                "routing.top_k must match indices.shape[-1]: "
                f"got top_k={self.top_k}, indices.shape[-1]={inferred}."
            )

    @property
    def selected_weights(self) -> Tensor | None:
        return self.weights

    @property
    def is_dense(self) -> bool:
        return self.top_k is None

    @property
    def num_experts(self) -> int:
        return int(self.probabilities.shape[-1])

    @property
    def leading_shape(self) -> torch.Size:
        return self.probabilities.shape[:-1]

    @property
    def dense_weights(self) -> DenseExpertWeights:
        return DenseExpertWeights(self.probabilities)

    def active_mask(self) -> Tensor:
        return self.flat_active_mask()

    def flat_active_mask(self) -> Tensor:
        return flatten_active_mask(
            self.mask,
            leading_shape=self.probabilities.shape[:-1],
            device=self.probabilities.device,
            target_name="routing.probabilities.shape[:-1]",
        )

    def require_topk(self) -> tuple[Tensor, Tensor, int]:
        """Return selected indices/weights and top_k, or raise if dense."""

        if self.top_k is None or self.indices is None or self.weights is None:
            raise ValueError("sparse expert path requires materialized top-k routing.")
        return self.indices, self.weights, self.top_k

def validate_num_experts(num_experts: int, *, name: str = "num_experts") -> None:
    if num_experts <= 0:
        raise ValueError(f"{name} must be positive, got {num_experts}.")


def validate_top_k(top_k: int, *, num_experts: int, name: str = "top_k") -> None:
    if top_k <= 0:
        raise ValueError(f"{name} must be positive, got {top_k}.")
    if top_k > num_experts:
        raise ValueError(
            f"{name} must be <= num_experts: got {name}={top_k}, num_experts={num_experts}."
        )


def expert_routing_from_weights(weights: Tensor) -> ExpertRouting:
    """Build dense ``ExpertRouting`` from already-normalized expert weights."""

    probs = weights
    logits = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
    return ExpertRouting(logits=logits, probabilities=probs, top_k=None)


def materialize_routing(
    logits: Tensor,
    probabilities: Tensor,
    *,
    top_k: int | None,
    mask: Tensor | None = None,
    eps: float = 1e-9,
) -> ExpertRouting:
    """Build a density-neutral routing record.

    ``probabilities`` are the full expert distribution from the gate (for
    diagnostics and dense execution). Sparse combine weights are formed by
    selecting top-k experts on ``logits``, then softmax over those selected
    logits — not by renormalizing a full-gate top-k slice.
    """

    del eps
    if logits.shape != probabilities.shape:
        raise ValueError(
            "logits and probabilities must share shape: "
            f"got {tuple(logits.shape)} and {tuple(probabilities.shape)}."
        )
    if logits.ndim < 1:
        raise ValueError("logits must have an expert dimension.")

    broadcast_mask = None
    if mask is not None:
        broadcast_mask = broadcast_routing_mask(
            mask,
            leading_shape=probabilities.shape[:-1],
            device=probabilities.device,
            target_name="probabilities.shape[:-1]",
        )

    if top_k is None:
        return ExpertRouting(
            logits=logits,
            probabilities=probabilities,
            mask=broadcast_mask,
            top_k=None,
        )

    validate_top_k(top_k, num_experts=int(probabilities.shape[-1]))
    selected_logits, indices = logits.topk(top_k, dim=-1)
    selected_weights = selected_logits.softmax(dim=-1)
    if broadcast_mask is not None:
        selected_weights = torch.where(
            broadcast_mask.unsqueeze(-1),
            selected_weights,
            torch.zeros_like(selected_weights),
        )
    return ExpertRouting(
        logits=logits,
        probabilities=probabilities,
        indices=indices,
        weights=selected_weights.to(dtype=probabilities.dtype),
        mask=broadcast_mask,
        top_k=top_k,
    )


def dense_expert_weight_tensor(expert_weights: Tensor | DenseExpertWeights) -> Tensor:
    if isinstance(expert_weights, DenseExpertWeights):
        return expert_weights.weights
    return expert_weights


def validate_dense_expert_weights(
    expert_weights: Tensor | DenseExpertWeights,
    *,
    num_experts: int,
    name: str = "expert_weights",
    ndim: int | None = None,
) -> Tensor:
    """Validate dense expert mixture weights and return the underlying tensor."""

    weights = dense_expert_weight_tensor(expert_weights)
    if ndim is not None and weights.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(weights.shape)}.")
    if weights.ndim < 1:
        raise ValueError(f"{name} must have an expert dimension, got shape {tuple(weights.shape)}.")
    if weights.size(-1) != num_experts:
        raise ValueError(
            f"{name} last dimension must match num_experts: "
            f"got {weights.size(-1)}, expected {num_experts}."
        )
    if not torch.is_floating_point(weights) or torch.is_complex(weights):
        raise TypeError(f"{name} must be real floating-point values.")
    return weights


def expand_dense_expert_weights(
    expert_weights: Tensor,
    *,
    batch_size: int,
    original_batch_size: int | None = None,
    num_segments: int = 1,
    name: str = "expert_weights",
) -> Tensor:
    """Expand dense expert weights across batch or segmented dynamic layers."""

    if expert_weights.size(0) == batch_size:
        return expert_weights
    if expert_weights.size(0) == 1:
        return expert_weights.repeat(batch_size, 1)
    if (
        original_batch_size is not None
        and num_segments > 1
        and expert_weights.size(0) == original_batch_size
    ):
        return expert_weights.repeat_interleave(num_segments, dim=0)
    raise ValueError(
        f"{name} batch size {expert_weights.size(0)} must match input batch size "
        f"{batch_size} or be 1."
    )


def broadcast_routing_mask(
    mask: Tensor,
    *,
    leading_shape: torch.Size | tuple[int, ...],
    device: torch.device,
    name: str | None = None,
    leading_name: str | None = None,
    mask_name: str = "routing mask",
    target_name: str = "leading shape",
) -> Tensor:
    if name is not None:
        mask_name = name
    if leading_name is not None:
        target_name = leading_name
    if mask.dtype != torch.bool:
        raise TypeError(f"{mask_name} must be a bool tensor, got dtype={mask.dtype}.")
    if mask.device != device:
        raise ValueError(f"{mask_name} must be on device {device}, got {mask.device}.")
    try:
        return torch.broadcast_to(mask, leading_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{mask_name} must be broadcastable to {target_name}: "
            f"got mask shape {tuple(mask.shape)}, leading shape {tuple(leading_shape)}."
        ) from exc


def flatten_active_mask(
    mask: Tensor | None,
    *,
    leading_shape: torch.Size | tuple[int, ...],
    device: torch.device,
    mask_name: str = "routing mask",
    target_name: str = "leading shape",
) -> Tensor:
    token_count = math.prod(leading_shape)
    if mask is None:
        return torch.ones(token_count, dtype=torch.bool, device=device)
    return broadcast_routing_mask(
        mask,
        leading_shape=leading_shape,
        device=device,
        mask_name=mask_name,
        target_name=target_name,
    ).reshape(-1)


flat_active_mask = flatten_active_mask


def validate_expert_routing(
    routing: ExpertRouting,
    *,
    leading_shape: torch.Size | tuple[int, ...],
    num_experts: int,
    device: torch.device,
    require_topk: bool = False,
    check_index_bounds: bool = True,
) -> None:
    """Validate routing tensors against a routed-cell leading shape.

    ``check_index_bounds=False`` is reserved for routes materialized internally
    by :func:`materialize_routing`, whose ``topk`` indices are already known to
    be in range. External routing records should keep the default strict check.
    """

    expected_probability_shape = (*leading_shape, num_experts)
    if routing.probabilities.shape != expected_probability_shape:
        raise ValueError(
            "routing.probabilities must have shape leading_shape + (num_experts,): "
            f"got {tuple(routing.probabilities.shape)}, expected "
            f"{tuple(expected_probability_shape)}."
        )
    if routing.logits.shape != routing.probabilities.shape:
        raise ValueError(
            "routing.logits must have the same shape as routing.probabilities: "
            f"got {tuple(routing.logits.shape)} and {tuple(routing.probabilities.shape)}."
        )
    if routing.probabilities.device != device or routing.logits.device != device:
        raise ValueError("routing logits and probabilities must be on the routed tensor device.")
    if not torch.is_floating_point(routing.probabilities) or torch.is_complex(
        routing.probabilities
    ):
        raise TypeError("routing.probabilities must be real floating-point values.")
    if routing.mask is not None:
        flatten_active_mask(
            routing.mask,
            leading_shape=leading_shape,
            device=device,
            target_name="routing leading shape",
        )

    if routing.top_k is None:
        if require_topk:
            raise ValueError("sparse expert path requires materialized top-k routing.")
        if routing.indices is not None or routing.weights is not None:
            raise ValueError("dense routing must leave indices/weights as None.")
        return

    if routing.indices is None or routing.weights is None:
        raise ValueError("top-k routing requires both indices and weights.")
    expected_route_shape = (*leading_shape, routing.top_k)
    if routing.indices.shape != expected_route_shape:
        raise ValueError(
            "routing.indices must have shape leading_shape + (top_k,): "
            f"got {tuple(routing.indices.shape)}, expected {tuple(expected_route_shape)}."
        )
    if routing.weights.shape != routing.indices.shape:
        raise ValueError(
            "routing.weights must have the same shape as routing.indices: "
            f"got {tuple(routing.weights.shape)} and {tuple(routing.indices.shape)}."
        )
    if routing.indices.device != device or routing.weights.device != device:
        raise ValueError("routing indices and weights must be on the routed tensor device.")
    if torch.is_floating_point(routing.indices) or torch.is_complex(routing.indices):
        raise TypeError("routing.indices must be an integer tensor.")
    if not torch.is_floating_point(routing.weights) or torch.is_complex(routing.weights):
        raise TypeError("routing.weights must be real floating-point values.")
    if check_index_bounds and routing.indices.numel() > 0:
        invalid = (routing.indices < 0) | (routing.indices >= num_experts)
        if bool(invalid.any().item()):
            raise ValueError("routing.indices contains expert ids outside [0, num_experts).")


def validate_sparse_expert_routing(
    routing: ExpertRouting,
    *,
    leading_shape: torch.Size | tuple[int, ...],
    num_experts: int,
    device: torch.device,
) -> None:
    """Validate sparse top-k routing tensors against a routed token leading shape."""

    validate_expert_routing(
        routing,
        leading_shape=leading_shape,
        num_experts=num_experts,
        device=device,
        require_topk=True,
    )


__all__ = [
    "DenseExpertWeights",
    "ExpertRouting",
    "broadcast_routing_mask",
    "dense_expert_weight_tensor",
    "expand_dense_expert_weights",
    "expert_routing_from_weights",
    "flat_active_mask",
    "flatten_active_mask",
    "materialize_routing",
    "validate_dense_expert_weights",
    "validate_expert_routing",
    "validate_num_experts",
    "validate_sparse_expert_routing",
    "validate_top_k",
]
