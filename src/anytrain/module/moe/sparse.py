"""Sparse token executor for module-backed experts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .routing import ExpertRouting, flat_active_mask, validate_sparse_expert_routing


def sparse_dispatch(
    hidden_states: Tensor,
    routing: ExpertRouting,
    experts: Sequence[nn.Module],
) -> tuple[Tensor, Tensor]:
    """Route feature-last tokens to local experts and combine their outputs."""

    if len(experts) <= 0:
        raise ValueError("experts must contain at least one module.")
    if hidden_states.ndim < 1:
        raise ValueError("hidden_states must have at least one dimension.")

    leading_shape = hidden_states.shape[:-1]
    validate_sparse_expert_routing(
        routing,
        leading_shape=leading_shape,
        num_experts=len(experts),
        device=hidden_states.device,
    )
    indices, weights, top_k = routing.require_topk()
    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_indices = indices.reshape(-1, top_k)
    flat_weights = weights.reshape(-1, top_k)
    active_mask = flat_active_mask(
        routing.mask,
        leading_shape=leading_shape,
        device=hidden_states.device,
    )

    output: Tensor | None = None
    output_tail_shape: torch.Size | None = None
    tokens_per_expert = torch.zeros(
        len(experts),
        dtype=torch.long,
        device=hidden_states.device,
    )

    route_token_indices = torch.arange(
        flat_hidden.size(0),
        device=hidden_states.device,
    ).unsqueeze(-1)
    route_token_indices = route_token_indices.expand_as(flat_indices).reshape(-1)
    route_expert_indices = flat_indices.reshape(-1)
    route_weights = flat_weights.reshape(-1)
    valid_routes = active_mask.unsqueeze(-1).expand_as(flat_indices).reshape(-1)
    valid_routes = valid_routes & (route_weights != 0)

    route_expert_indices = route_expert_indices[valid_routes]
    route_token_indices = route_token_indices[valid_routes]
    route_weights = route_weights[valid_routes]

    sorted_order = route_expert_indices.argsort(stable=True)
    route_expert_indices = route_expert_indices.index_select(0, sorted_order)
    route_token_indices = route_token_indices.index_select(0, sorted_order)
    route_weights = route_weights.index_select(0, sorted_order)
    tokens_per_expert = torch.bincount(
        route_expert_indices,
        minlength=len(experts),
    )
    # Python dispatch to distinct nn.Module experts needs segment lengths on the
    # host. Keep that to one synchronization instead of probing every expert.
    token_counts = tokens_per_expert.tolist()

    start = 0
    for expert_index, (expert, token_count) in enumerate(zip(experts, token_counts)):
        if token_count == 0:
            continue
        end = start + token_count
        token_indices = route_token_indices[start:end]
        expert_weights = route_weights[start:end]
        start = end
        expert_input = flat_hidden.index_select(0, token_indices)
        expert_output = expert(expert_input)
        _validate_expert_output(
            expert_output,
            token_count=token_count,
            expert_index=expert_index,
        )

        if output is None:
            tail_shape = expert_output.shape[1:]
            output_tail_shape = tail_shape
            output = expert_output.new_zeros((flat_hidden.size(0), *tail_shape))
        else:
            assert output_tail_shape is not None
            if expert_output.shape[1:] != output_tail_shape:
                raise ValueError(
                    "all experts must return the same non-batch shape: "
                    f"expert 0 returned {tuple(output_tail_shape)}, "
                    f"expert {expert_index} returned {tuple(expert_output.shape[1:])}."
                )

        combine_weight = expert_weights.to(dtype=expert_output.dtype)
        combine_weight = combine_weight.reshape(-1, *([1] * (expert_output.ndim - 1)))
        assert output is not None
        output = output.index_add(0, token_indices, expert_output * combine_weight)

    if output is None:
        output = _empty_output(flat_hidden, experts[0])

    assert output is not None
    return output.reshape(*leading_shape, *output.shape[1:]), tokens_per_expert


def _validate_expert_output(output: Tensor, *, token_count: int, expert_index: int) -> None:
    if not isinstance(output, Tensor):
        raise TypeError(f"expert {expert_index} must return a Tensor.")
    if output.ndim < 1:
        raise ValueError(f"expert {expert_index} must return a tensor with a batch dimension.")
    if output.size(0) != token_count:
        raise ValueError(
            f"expert {expert_index} output batch size must match routed token count: "
            f"got {output.size(0)}, expected {token_count}."
        )


def _empty_output(flat_hidden: Tensor, expert: nn.Module) -> Tensor:
    probe_output = expert(flat_hidden[:0])
    _validate_expert_output(probe_output, token_count=0, expert_index=0)
    return probe_output.new_zeros((flat_hidden.size(0), *probe_output.shape[1:]))


dispatch_to_experts = sparse_dispatch


__all__ = ["dispatch_to_experts", "sparse_dispatch"]
