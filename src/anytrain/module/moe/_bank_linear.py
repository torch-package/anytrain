"""Linear and generic expert execution over bank-owned parameter tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from ._bank_routes import (
    _dense_probabilities,
    _empty_bank_output,
    _grouped_topk_routes,
    _restore_active_tokens,
    _validate_generic_expert_output,
)
from ._dispatch import _linear_route_memory_fits
from .expert import Expert, LinearExpert
from .routing import ExpertRouting


def linear(
    expert: Expert,
    weight: Tensor,
    bias: Tensor | None,
    num_experts: int,
    cells: Tensor,
    routing: ExpertRouting,
) -> Tensor:
    leading_shape = cells.shape[:-1]
    flat = cells.reshape(-1, cells.size(-1))
    if routing.is_dense:
        probs = _dense_probabilities(routing)
        active_indices: Tensor | None = None
        token_count = flat.size(0)
        if routing.mask is not None:
            active_indices = routing.flat_active_mask().nonzero(as_tuple=False).flatten()
            flat = flat.index_select(0, active_indices)
            probs = probs.index_select(0, active_indices)
        y = None
        for expert_index in range(num_experts):
            expert_y = expert(
                flat,
                weight[expert_index],
                None,
            )
            combine_dtype = torch.promote_types(expert_y.dtype, probs.dtype)
            if y is None:
                y = expert_y.new_zeros(expert_y.shape, dtype=combine_dtype)
            else:
                accumulator_dtype = torch.promote_types(y.dtype, combine_dtype)
                if accumulator_dtype != y.dtype:
                    y = y.to(dtype=accumulator_dtype)
            combine_weight = probs[:, expert_index].to(dtype=y.dtype)
            y.add_(expert_y.to(dtype=y.dtype) * combine_weight.unsqueeze(-1))
        assert y is not None
        if active_indices is not None:
            y = _restore_active_tokens(y, active_indices, token_count)
    else:
        if (
            type(expert) is LinearExpert
            and flat.device.type == "cuda"
            and weight.device == flat.device
            and weight.layout == torch.strided
            and (bias is None or bias.layout == torch.strided)
            and (bias is None or bias.device == flat.device)
            and (bias is None or bias.dtype == weight.dtype)
            and _linear_route_memory_fits(
                routing.indices.numel() if routing.indices is not None else 0,
                weight[0],
            )
        ):
            # A linear expert is pure and has no per-module state, so all
            # selected routes can share one batched matrix multiplication.
            # This avoids host count materialization and one accelerator
            # launch per non-empty expert in the old grouped path. CPU
            # keeps the grouped GEMM path, which is faster for route-wise
            # small matrix products on common BLAS backends.
            y = linear_topk_batched(expert, weight, flat, routing)
        else:
            route_tokens, route_weights, token_counts = _grouped_topk_routes(routing, num_experts)
            y = None
            start = 0
            for expert_index, token_count in enumerate(token_counts):
                if token_count == 0:
                    continue
                end = start + token_count
                token_indices = route_tokens[start:end]
                expert_input = flat.index_select(0, token_indices)
                expert_y = expert(
                    expert_input,
                    weight[expert_index],
                    None,
                )
                combine_dtype = torch.promote_types(
                    expert_y.dtype,
                    route_weights.dtype,
                )
                if y is None:
                    y = expert_y.new_zeros(
                        flat.size(0),
                        expert_y.size(-1),
                        dtype=combine_dtype,
                    )
                else:
                    accumulator_dtype = torch.promote_types(y.dtype, combine_dtype)
                    if accumulator_dtype != y.dtype:
                        y = y.to(dtype=accumulator_dtype)
                combine_weight = route_weights[start:end].to(dtype=y.dtype)
                y.index_add_(
                    0,
                    token_indices,
                    expert_y.to(dtype=y.dtype) * combine_weight.unsqueeze(-1),
                )
                start = end
            if y is None:
                probe = expert(flat[:0], weight[0], None)
                _validate_generic_expert_output(
                    probe,
                    token_count=0,
                    expert_index=0,
                )
                y = _empty_bank_output(
                    probe,
                    token_count=flat.size(0),
                    route_weights=route_weights,
                    graph_inputs=(flat, weight),
                )
    return y.reshape(*leading_shape, y.size(-1))


def linear_topk_batched(
    expert: Expert, weight: Tensor, flat: Tensor, routing: ExpertRouting
) -> Tensor:
    """Execute all active linear routes with one batched matmul.

    ``LinearExpert`` has no observable per-expert state, so grouping routes
    by expert is unnecessary.  Gathering the expert weights by route and
    using ``bmm`` preserves the sparse computation (``N * top_k`` linear
    products) while avoiding a device-to-host count conversion and a
    separate launch for each expert.  Zero-weight routes intentionally stay
    in the batch: their output is zero, but their expert output remains in
    the autograd graph so gradients with respect to routing weights match
    the reference implementation.
    """

    indices, selected_weights, top_k = routing.require_topk()
    flat_indices = indices.reshape(-1, top_k)
    flat_weights = selected_weights.reshape(-1, top_k)
    route_tokens = (
        torch.arange(
            flat.size(0),
            device=flat.device,
        )
        .unsqueeze(-1)
        .expand_as(flat_indices)
        .reshape(-1)
    )
    route_experts = flat_indices.reshape(-1)
    route_weights = flat_weights.reshape(-1)
    original_route_weights = route_weights

    if routing.mask is not None:
        active_routes = routing.flat_active_mask().unsqueeze(-1).expand_as(flat_indices).reshape(-1)
        active_indices = active_routes.nonzero(as_tuple=False).flatten()
        route_tokens = route_tokens.index_select(0, active_indices)
        route_experts = route_experts.index_select(0, active_indices)
        route_weights = route_weights.index_select(0, active_indices)

    if route_tokens.numel() == 0:
        # Probe an empty input to retain the exact output shape/dtype and a
        # zero-gradient edge to inputs, parameters, and route weights.  The
        # empty slice also guarantees masked NaN/Inf values are never read.
        probe = expert(flat[:0], weight[0], None)
        _validate_generic_expert_output(
            probe,
            token_count=0,
            expert_index=0,
        )
        return _empty_bank_output(
            probe,
            token_count=flat.size(0),
            route_weights=route_weights,
            graph_inputs=(flat, weight, original_route_weights),
        )

    route_input = flat.index_select(0, route_tokens)
    route_weight = weight.index_select(0, route_experts)
    # ``LinearExpert`` is F.linear(input, weight, bias=None).  Batched
    # matmul has the same accumulation/dtype behavior under autocast while
    # exposing a single operation to the backend.
    expert_output = torch.bmm(
        route_input.unsqueeze(1),
        route_weight.transpose(1, 2),
    ).squeeze(1)
    combine_dtype = torch.promote_types(
        expert_output.dtype,
        route_weights.dtype,
    )
    weighted_output = expert_output.to(dtype=combine_dtype) * route_weights.to(
        dtype=combine_dtype
    ).unsqueeze(-1)
    output = weighted_output.new_zeros(
        flat.size(0),
        expert_output.size(-1),
    )
    output.index_add_(0, route_tokens, weighted_output)
    if routing.mask is not None:
        # Masked route weights are intentionally not gathered above. Keep
        # the reference API's zero-gradient edge for them without ever
        # reading their (possibly non-finite) values.
        output = output + original_route_weights.unsqueeze(0)[:0].sum().to(dtype=output.dtype)
    return output


def generic(
    expert: Expert, weight: Tensor, num_experts: int, cells: Tensor, routing: ExpertRouting
) -> Tensor:
    leading = routing.leading_shape
    if tuple(cells.shape[: len(leading)]) != tuple(leading):
        raise ValueError(
            "generic Expert cells leading dims must match routing: "
            f"got {tuple(cells.shape)}, leading {tuple(leading)}."
        )
    flat_count = 1
    for dim in leading:
        flat_count *= int(dim)
    flat_cells = cells.reshape(flat_count, *cells.shape[len(leading) :])

    if routing.is_dense:
        probs = _dense_probabilities(routing)
        active_indices: Tensor | None = None
        if routing.mask is not None:
            active_indices = routing.flat_active_mask().nonzero(as_tuple=False).flatten()
            flat_cells = flat_cells.index_select(0, active_indices)
            probs = probs.index_select(0, active_indices)
        dense_count = flat_cells.size(0)
        outputs = []
        output_tail_shape = None
        for expert_index in range(num_experts):
            output = expert(flat_cells, weight[expert_index], None)
            _validate_generic_expert_output(
                output,
                token_count=dense_count,
                expert_index=expert_index,
            )
            if output_tail_shape is None:
                output_tail_shape = output.shape[1:]
            elif output.shape[1:] != output_tail_shape:
                raise ValueError(
                    "all experts must return the same non-batch shape: "
                    f"expected {tuple(output_tail_shape)}, got "
                    f"{tuple(output.shape[1:])} from expert {expert_index}."
                )
            outputs.append(output)
        stacked = torch.stack(outputs, dim=1)
        weight_view = probs.reshape(
            dense_count,
            num_experts,
            *([1] * (stacked.ndim - 2)),
        )
        mixed = (weight_view * stacked).sum(dim=1)
        if active_indices is not None:
            mixed = _restore_active_tokens(mixed, active_indices, flat_count)
    else:
        route_tokens, route_weights, token_counts = _grouped_topk_routes(routing, num_experts)
        mixed = None
        output_tail_shape = None
        start = 0
        for expert_index, token_count in enumerate(token_counts):
            if token_count == 0:
                continue
            end = start + token_count
            token_indices = route_tokens[start:end]
            expert_output = expert(
                flat_cells.index_select(0, token_indices),
                weight[expert_index],
                None,
            )
            _validate_generic_expert_output(
                expert_output,
                token_count=token_count,
                expert_index=expert_index,
            )
            combine_dtype = torch.promote_types(
                expert_output.dtype,
                route_weights.dtype,
            )
            if mixed is None:
                output_tail_shape = expert_output.shape[1:]
                mixed = expert_output.new_zeros(
                    flat_count,
                    *output_tail_shape,
                    dtype=combine_dtype,
                )
            else:
                assert output_tail_shape is not None
                accumulator_dtype = torch.promote_types(
                    mixed.dtype,
                    combine_dtype,
                )
                if accumulator_dtype != mixed.dtype:
                    mixed = mixed.to(dtype=accumulator_dtype)
            if expert_output.shape[1:] != output_tail_shape:
                raise ValueError(
                    "all experts must return the same non-batch shape: "
                    f"expected {tuple(output_tail_shape)}, got "
                    f"{tuple(expert_output.shape[1:])} from expert {expert_index}."
                )
            expert_output = expert_output.to(dtype=mixed.dtype)
            combine_weight = route_weights[start:end].to(dtype=mixed.dtype)
            combine_weight = combine_weight.reshape(
                -1,
                *([1] * (expert_output.ndim - 1)),
            )
            mixed.index_add_(
                0,
                token_indices,
                expert_output * combine_weight,
            )
            start = end
        if mixed is None:
            probe = expert(flat_cells[:0], weight[0], None)
            _validate_generic_expert_output(
                probe,
                token_count=0,
                expert_index=0,
            )
            mixed = _empty_bank_output(
                probe,
                token_count=flat_count,
                route_weights=route_weights,
                graph_inputs=(flat_cells, weight),
            )
    return mixed.reshape(*leading, *mixed.shape[1:])
