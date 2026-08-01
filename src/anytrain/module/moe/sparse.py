"""Sparse token executor for module-backed experts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .routing import SparseExpertRouting, flat_active_mask, validate_sparse_expert_routing


def sparse_dispatch(
    hidden_states: Tensor,
    routing: SparseExpertRouting,
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
    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_indices = routing.indices.reshape(-1, routing.top_k)
    flat_weights = routing.weights.reshape(-1, routing.top_k)
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

    for expert_index, expert in enumerate(experts):
        selected = (flat_indices == expert_index) & active_mask.unsqueeze(-1) & (flat_weights != 0)
        if not bool(selected.any().item()):
            continue

        token_indices, choice_indices = selected.nonzero(as_tuple=True)
        expert_input = flat_hidden.index_select(0, token_indices)
        expert_output = expert(expert_input)
        _validate_expert_output(
            expert_output,
            token_count=token_indices.numel(),
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

        combine_weight = flat_weights[token_indices, choice_indices].to(dtype=expert_output.dtype)
        combine_weight = combine_weight.reshape(-1, *([1] * (expert_output.ndim - 1)))
        assert output is not None
        output = output.index_add(0, token_indices, expert_output * combine_weight)
        tokens_per_expert[expert_index] = token_indices.numel()

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
