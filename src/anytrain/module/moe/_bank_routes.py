"""Stateless route planning and accumulation helpers for parameter banks."""

from __future__ import annotations

import torch
from torch import Tensor

from ._dispatch import _host_int_list
from .routing import ExpertRouting


def _restore_active_tokens(
    active_output: Tensor,
    active_indices: Tensor,
    token_count: int,
) -> Tensor:
    output = active_output.new_zeros(
        token_count,
        *active_output.shape[1:],
    )
    return output.index_copy(0, active_indices, active_output)


def _dense_probabilities(routing: ExpertRouting) -> Tensor:
    probabilities = routing.probabilities.reshape(-1, routing.num_experts)
    if routing.mask is None:
        return probabilities
    active = routing.flat_active_mask().unsqueeze(-1)
    return probabilities.masked_fill(~active, 0)


def _validate_generic_expert_output(
    output: Tensor,
    *,
    token_count: int,
    expert_index: int,
) -> None:
    if not isinstance(output, Tensor):
        raise TypeError(f"expert {expert_index} must return a Tensor.")
    if output.ndim < 1:
        raise ValueError(f"expert {expert_index} must return a tensor with a batch dimension.")
    if output.size(0) != token_count:
        raise ValueError(
            "expert output batch size must match routed token count: "
            f"expert {expert_index} returned {output.size(0)}, expected {token_count}."
        )


def _empty_bank_output(
    probe: Tensor,
    *,
    token_count: int,
    route_weights: Tensor,
    graph_inputs: tuple[Tensor, ...],
) -> Tensor:
    output_dtype = torch.promote_types(probe.dtype, route_weights.dtype)
    output = probe.new_zeros(
        token_count,
        *probe.shape[1:],
        dtype=output_dtype,
    )
    graph_zero = output.new_zeros(())
    for graph_input in (*graph_inputs, route_weights):
        graph_zero = graph_zero + graph_input.unsqueeze(0)[:0].sum().to(dtype=output_dtype)
    return output + graph_zero


def _grouped_topk_routes(
    routing: ExpertRouting,
    num_experts: int,
) -> tuple[Tensor, Tensor, list[int]]:
    indices, selected_weights, top_k = routing.require_topk()
    flat_indices = indices.reshape(-1, top_k)
    flat_weights = selected_weights.reshape(-1, top_k)

    route_tokens = torch.arange(
        flat_indices.size(0),
        device=flat_indices.device,
    ).unsqueeze(-1)
    route_tokens = route_tokens.expand_as(flat_indices).reshape(-1)
    route_experts = flat_indices.reshape(-1)
    route_weights = flat_weights.reshape(-1)
    if routing.mask is None:
        group_keys = route_experts
    else:
        active = routing.flat_active_mask().unsqueeze(-1).expand_as(flat_indices)
        sentinel = torch.full_like(route_experts, num_experts)
        group_keys = torch.where(active.reshape(-1), route_experts, sentinel)
    order = group_keys.argsort(stable=True)
    route_tokens = route_tokens.index_select(0, order)
    route_weights = route_weights.index_select(0, order)
    group_keys = group_keys.index_select(0, order)
    counts = torch.bincount(group_keys, minlength=num_experts + 1)
    # Calling distinct expert parameters needs segment sizes on the host.
    # This is one synchronization per bank call, independent of route count.
    # Stock LinearExpert bypasses this fallback entirely.
    token_counts = _host_int_list(counts[:num_experts])
    return route_tokens, route_weights, token_counts
