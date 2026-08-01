from __future__ import annotations

import torch
from torch import Tensor

from ...routing import (
    DenseExpertWeights,
    validate_dense_expert_weights,
    validate_num_experts,
)


def validate_channels(
    *,
    in_channels: int,
    out_channels: int,
    groups: int,
    num_experts: int,
) -> None:
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}.")
    if out_channels <= 0:
        raise ValueError(f"out_channels must be positive, got {out_channels}.")
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}.")
    validate_num_experts(num_experts)
    if in_channels % groups != 0:
        raise ValueError(
            f"in_channels must be divisible by groups: got {in_channels} and {groups}."
        )
    if out_channels % groups != 0:
        raise ValueError(
            f"out_channels must be divisible by groups: got {out_channels} and {groups}."
        )


def validate_expert_weights(
    expert_weights: Tensor | DenseExpertWeights,
    *,
    num_experts: int,
) -> Tensor:
    return validate_dense_expert_weights(expert_weights, num_experts=num_experts, ndim=2)


def expand_expert_weights(
    expert_weights: Tensor,
    *,
    batch_size: int,
    original_batch_size: int | None = None,
    num_segments: int = 1,
) -> Tensor:
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
        f"expert_weights batch size {expert_weights.size(0)} must match input batch size "
        f"{batch_size} or be 1."
    )


def mix_expert_weights(expert_weights: Tensor, weight: Tensor) -> Tensor:
    return torch.matmul(expert_weights, weight.flatten(start_dim=1)).reshape(
        expert_weights.size(0),
        *weight.shape[1:],
    )


def mix_expert_bias(expert_weights: Tensor, bias: Tensor) -> Tensor:
    return expert_weights @ bias
