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
    """Sparse expert routing result shared by routing-based modules.

    logits and probabilities describe the full expert distribution with shape
    (..., num_experts). indices and weights describe the selected top-k routes
    with shape (..., top_k). mask marks active leading positions and must be
    broadcastable to probabilities.shape[:-1].
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


SparseExpertRouting = ExpertRouting


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


def validate_sparse_expert_routing(
    routing: SparseExpertRouting,
    *,
    leading_shape: torch.Size | tuple[int, ...],
    num_experts: int,
    device: torch.device,
) -> None:
    """Validate sparse routing tensors against a routed token leading shape."""

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
    if routing.indices.device != device or routing.weights.device != device:
        raise ValueError("routing indices and weights must be on the routed tensor device.")
    if routing.probabilities.device != device or routing.logits.device != device:
        raise ValueError("routing logits and probabilities must be on the routed tensor device.")
    if torch.is_floating_point(routing.indices) or torch.is_complex(routing.indices):
        raise TypeError("routing.indices must be an integer tensor.")
    if not torch.is_floating_point(routing.weights) or torch.is_complex(routing.weights):
        raise TypeError("routing.weights must be real floating-point values.")
    if not torch.is_floating_point(routing.probabilities) or torch.is_complex(
        routing.probabilities
    ):
        raise TypeError("routing.probabilities must be real floating-point values.")
    if routing.indices.numel() > 0:
        invalid = (routing.indices < 0) | (routing.indices >= num_experts)
        if bool(invalid.any().item()):
            raise ValueError("routing.indices contains expert ids outside [0, num_experts).")
    if routing.mask is not None:
        flatten_active_mask(
            routing.mask,
            leading_shape=leading_shape,
            device=device,
            target_name="routing leading shape",
        )


__all__ = [
    "DenseExpertWeights",
    "ExpertRouting",
    "SparseExpertRouting",
    "broadcast_routing_mask",
    "dense_expert_weight_tensor",
    "expand_dense_expert_weights",
    "flat_active_mask",
    "flatten_active_mask",
    "validate_dense_expert_weights",
    "validate_num_experts",
    "validate_sparse_expert_routing",
    "validate_top_k",
]
