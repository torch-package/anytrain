"""Grouped convolution execution over bank-owned parameter tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from ._bank_routes import _dense_probabilities
from .conv.bank import mix_expert_weights
from .conv.expert_bank import grouped_conv1d, grouped_conv2d, grouped_conv_transpose1d
from .expert import Conv1dExpert, Conv2dExpert, ConvTranspose1dExpert, Expert
from .routing import ExpertRouting


def conv1d(expert: Expert, weight: Tensor, cells: Tensor, routing: ExpertRouting) -> Tensor:
    if not isinstance(expert, Conv1dExpert):
        raise TypeError(f"CONV1D bank requires Conv1dExpert, got {type(expert)!r}.")
    if cells.ndim != 3:
        raise ValueError(f"Conv1dExpert expects cells shaped (U, C, T), got {tuple(cells.shape)}.")
    if routing.is_dense:
        probs = _dense_probabilities(routing)
        mixed_weight = mix_expert_weights(probs, weight)
        return grouped_conv1d(
            cells.to(dtype=mixed_weight.dtype),
            mixed_weight,
            stride=expert.stride,
            padding=expert.padding,
            dilation=expert.dilation,
            groups=expert.groups,
        )
    indices, selected_weights, top_k = routing.require_topk()
    cells_rep = cells.repeat_interleave(top_k, dim=0)
    expert_outputs = grouped_conv1d(
        cells_rep,
        weight[indices.reshape(-1)],
        stride=expert.stride,
        padding=expert.padding,
        dilation=expert.dilation,
        groups=expert.groups,
    )
    expert_outputs = expert_outputs.reshape(
        cells.size(0), top_k, expert_outputs.size(1), expert_outputs.size(-1)
    )
    combine_dtype = torch.promote_types(selected_weights.dtype, expert_outputs.dtype)
    return torch.einsum(
        "uk,ukot->uot",
        selected_weights.to(dtype=combine_dtype),
        expert_outputs.to(dtype=combine_dtype),
    )


def conv_transpose1d(
    expert: Expert, weight: Tensor, cells: Tensor, routing: ExpertRouting
) -> Tensor:
    if not isinstance(expert, ConvTranspose1dExpert):
        raise TypeError(
            f"CONV_TRANSPOSE1D bank requires ConvTranspose1dExpert, got {type(expert)!r}."
        )
    if cells.ndim != 3:
        raise ValueError(
            f"ConvTranspose1dExpert expects cells shaped (U, C, T), got {tuple(cells.shape)}."
        )
    if routing.is_dense:
        probs = _dense_probabilities(routing)
        mixed_weight = mix_expert_weights(probs, weight)
        return grouped_conv_transpose1d(
            cells.to(dtype=mixed_weight.dtype),
            mixed_weight,
            stride=expert.stride,
            dilation=expert.dilation,
            groups=expert.groups,
        )
    indices, selected_weights, top_k = routing.require_topk()
    cells_rep = cells.repeat_interleave(top_k, dim=0)
    expert_outputs = grouped_conv_transpose1d(
        cells_rep,
        weight[indices.reshape(-1)],
        stride=expert.stride,
        dilation=expert.dilation,
        groups=expert.groups,
    )
    expert_outputs = expert_outputs.reshape(
        cells.size(0), top_k, expert_outputs.size(1), expert_outputs.size(-1)
    )
    combine_dtype = torch.promote_types(selected_weights.dtype, expert_outputs.dtype)
    return torch.einsum(
        "uk,ukot->uot",
        selected_weights.to(dtype=combine_dtype),
        expert_outputs.to(dtype=combine_dtype),
    )


def conv2d(expert: Expert, weight: Tensor, cells: Tensor, routing: ExpertRouting) -> Tensor:
    if not isinstance(expert, Conv2dExpert):
        raise TypeError(f"CONV2D bank requires Conv2dExpert, got {type(expert)!r}.")
    if cells.ndim != 4:
        raise ValueError(
            f"Conv2dExpert expects cells shaped (U, C, H, W), got {tuple(cells.shape)}."
        )
    if routing.is_dense:
        probs = _dense_probabilities(routing)
        mixed_weight = mix_expert_weights(probs, weight)
        return grouped_conv2d(
            cells.to(dtype=mixed_weight.dtype),
            mixed_weight,
            stride=expert.stride,
            padding=expert.padding,
            dilation=expert.dilation,
            groups=expert.groups,
        )
    indices, selected_weights, top_k = routing.require_topk()
    cells_rep = cells.repeat_interleave(top_k, dim=0)
    expert_outputs = grouped_conv2d(
        cells_rep,
        weight[indices.reshape(-1)],
        stride=expert.stride,
        padding=expert.padding,
        dilation=expert.dilation,
        groups=expert.groups,
    )
    expert_outputs = expert_outputs.reshape(
        cells.size(0),
        top_k,
        expert_outputs.size(1),
        expert_outputs.size(-2),
        expert_outputs.size(-1),
    )
    combine_dtype = torch.promote_types(selected_weights.dtype, expert_outputs.dtype)
    return torch.einsum(
        "uk,ukohw->uohw",
        selected_weights.to(dtype=combine_dtype),
        expert_outputs.to(dtype=combine_dtype),
    )
