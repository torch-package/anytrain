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
    output: Tensor | None = None
    output_tail_shape: torch.Size | None = None
    output_tail_expert_index: int | None = None

    route_token_indices = torch.arange(
        flat_hidden.size(0),
        device=hidden_states.device,
    ).unsqueeze(-1)
    route_token_indices = route_token_indices.expand_as(flat_indices).reshape(-1)
    route_expert_indices = flat_indices.reshape(-1)
    route_weights = flat_weights.reshape(-1)

    counted_routes = route_weights != 0
    if routing.mask is not None:
        active_mask = flat_active_mask(
            routing.mask,
            leading_shape=leading_shape,
            device=hidden_states.device,
        )
        active_routes = active_mask.unsqueeze(-1).expand_as(flat_indices).reshape(-1)
        counted_routes = counted_routes & active_routes
        route_expert_indices = route_expert_indices[active_routes]
        route_token_indices = route_token_indices[active_routes]
        route_weights = route_weights[active_routes]

    tokens_per_expert = torch.bincount(
        flat_indices.reshape(-1)[counted_routes],
        minlength=len(experts),
    )

    sorted_order = route_expert_indices.argsort(stable=True)
    route_expert_indices = route_expert_indices.index_select(0, sorted_order)
    route_token_indices = route_token_indices.index_select(0, sorted_order)
    route_weights = route_weights.index_select(0, sorted_order)
    dispatch_counts = torch.bincount(
        route_expert_indices,
        minlength=len(experts),
    )
    # Python dispatch to distinct nn.Module experts needs segment lengths on the
    # host. Keep that to one synchronization instead of probing every expert.
    token_counts = dispatch_counts.tolist()

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
            output_tail_expert_index = expert_index
            combine_dtype = torch.promote_types(
                expert_output.dtype,
                expert_weights.dtype,
            )
            output = expert_output.new_zeros(
                (flat_hidden.size(0), *tail_shape),
                dtype=combine_dtype,
            )
        else:
            assert output_tail_shape is not None
            if expert_output.shape[1:] != output_tail_shape:
                assert output_tail_expert_index is not None
                raise ValueError(
                    "all experts must return the same non-batch shape: "
                    f"expert {output_tail_expert_index} returned "
                    f"{tuple(output_tail_shape)}, "
                    f"expert {expert_index} returned {tuple(expert_output.shape[1:])}."
                )
            combine_dtype = torch.promote_types(
                expert_output.dtype,
                expert_weights.dtype,
            )
            accumulator_dtype = torch.promote_types(output.dtype, combine_dtype)
            if accumulator_dtype != output.dtype:
                output = output.to(dtype=accumulator_dtype)

        assert output is not None
        expert_output = expert_output.to(dtype=output.dtype)
        combine_weight = expert_weights.to(dtype=output.dtype)
        combine_weight = combine_weight.reshape(-1, *([1] * (expert_output.ndim - 1)))
        output = output.index_add(0, token_indices, expert_output * combine_weight)

    if output is None:
        output = _empty_output(flat_hidden, route_weights, experts)

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


def _empty_output(
    flat_hidden: Tensor,
    route_weights: Tensor,
    experts: Sequence[nn.Module],
) -> Tensor:
    probe_output = experts[0](flat_hidden)
    _validate_expert_output(
        probe_output,
        token_count=flat_hidden.size(0),
        expert_index=0,
    )
    output_dtype = torch.promote_types(probe_output.dtype, route_weights.dtype)
    output = probe_output.new_zeros(probe_output.shape, dtype=output_dtype)

    graph_inputs = [flat_hidden, route_weights]
    graph_inputs.extend(
        parameter
        for expert in experts
        for parameter in expert.parameters()
        if parameter.requires_grad and parameter.device == output.device
    )
    for graph_input in graph_inputs:
        graph_zero = graph_input.reshape(-1)[:1].sum().to(dtype=output_dtype) * 0
        output = output + graph_zero
    return output


dispatch_to_experts = sparse_dispatch


__all__ = ["dispatch_to_experts", "sparse_dispatch"]
