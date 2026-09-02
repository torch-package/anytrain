"""Sparse token executor for module-backed experts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.modules import module as nn_module

from .routing import ExpertRouting, flat_active_mask, validate_sparse_expert_routing

# Route-wise parameter gathers are intentionally bounded.  A top-k dispatch
# can otherwise materialize a very large ``(routes, out_features,
# in_features)`` temporary before the batched matrix multiplication starts.
_LINEAR_ROUTE_WEIGHT_BUDGET_BYTES = 64 * 1024 * 1024


def sparse_dispatch(
    hidden_states: Tensor,
    routing: ExpertRouting,
    experts: Sequence[nn.Module],
) -> tuple[Tensor, Tensor]:
    """Route feature-last tokens to local experts and combine their outputs.

    Homogeneous stock ``nn.Linear`` experts use a batched route kernel on
    accelerators; arbitrary/stateful modules retain grouped Python dispatch.
    """

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
    original_route_weights = route_weights

    counted_routes = route_weights != 0
    active_routes: Tensor | None = None
    if routing.mask is not None:
        active_mask = flat_active_mask(
            routing.mask,
            leading_shape=leading_shape,
            device=hidden_states.device,
        )
        active_routes = active_mask.unsqueeze(-1).expand_as(flat_indices).reshape(-1)
        counted_routes = counted_routes & active_routes

    sentinel = torch.full_like(route_expert_indices, len(experts))
    counted_experts = torch.where(counted_routes, route_expert_indices, sentinel)
    tokens_per_expert = torch.bincount(
        counted_experts,
        minlength=len(experts) + 1,
    )[: len(experts)]

    # ``nn.Linear`` experts are stateless apart from their parameters.  Route
    # gathers can therefore be evaluated by one batched matrix multiplication
    # on accelerators, avoiding the host count conversion and one Python/module
    # launch per non-empty expert in the generic path. CPU keeps the grouped
    # GEMM path, which is faster for route-wise small matrix products on common
    # BLAS backends. Keep the generic dispatcher below for arbitrary modules
    # (whose forward methods may have state or side effects that cannot be
    # safely vectorized).
    first_linear = experts[0] if experts else None
    if (
        flat_hidden.device.type == "cuda"
        and _can_batch_linear_experts(experts)
        and isinstance(first_linear, nn.Linear)
        and first_linear.weight.device == flat_hidden.device
        and _linear_fast_path_preserves_unused_gradients(
            route_expert_indices,
            active_routes=active_routes,
            num_experts=len(experts),
            experts=experts,
        )
        and _linear_route_memory_fits(
            route_expert_indices.numel(),
            first_linear.weight,
            first_linear.bias,
            stacked_expert_count=len(experts),
        )
    ):
        return _sparse_dispatch_linear(
            flat_hidden,
            leading_shape=leading_shape,
            route_token_indices=route_token_indices,
            route_expert_indices=route_expert_indices,
            route_weights=route_weights,
            active_routes=active_routes,
            original_route_weights=original_route_weights,
            tokens_per_expert=tokens_per_expert,
            experts=experts,
        )

    if active_routes is None:
        group_keys = route_expert_indices
    else:
        group_keys = torch.where(active_routes, route_expert_indices, sentinel)
    sorted_order = group_keys.argsort(stable=True)
    route_token_indices = route_token_indices.index_select(0, sorted_order)
    route_weights = route_weights.index_select(0, sorted_order)
    group_keys = group_keys.index_select(0, sorted_order)
    dispatch_counts = torch.bincount(
        group_keys,
        minlength=len(experts) + 1,
    )
    # Python dispatch to distinct nn.Module experts needs segment lengths on the
    # host. Keep that to one synchronization instead of probing every expert;
    # the homogeneous Linear fast path above bypasses this conversion.
    token_counts = _host_int_list(dispatch_counts[: len(experts)])

    start = 0
    for expert_index, (expert, token_count) in enumerate(zip(experts, token_counts, strict=False)):
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
    probe_output = _empty_expert_probe(experts[0], flat_hidden[:0])
    _validate_expert_output(
        probe_output,
        token_count=0,
        expert_index=0,
    )
    output_dtype = torch.promote_types(probe_output.dtype, route_weights.dtype)
    output = probe_output.new_zeros(
        (flat_hidden.size(0), *probe_output.shape[1:]),
        dtype=output_dtype,
    )

    graph_inputs = [flat_hidden, route_weights]
    graph_inputs.extend(
        parameter
        for expert in experts
        for parameter in expert.parameters()
        if parameter.requires_grad and parameter.device == output.device
    )
    graph_zero = output.new_zeros(())
    for graph_input in graph_inputs:
        # Empty reductions retain a zero-gradient edge without reading NaN/Inf data.
        graph_zero = graph_zero + graph_input.unsqueeze(0)[:0].sum().to(
            dtype=output_dtype
        )
    return output + graph_zero


def _empty_expert_probe(expert: nn.Module, empty_input: Tensor) -> Tensor:
    training_states = tuple((module, module.training) for module in expert.modules())
    try:
        for module, _training in training_states:
            module.training = False
        with torch.no_grad():
            return expert(empty_input)
    finally:
        for module, training in training_states:
            module.training = training


def _host_int_list(values: Tensor) -> list[int]:
    """Materialize a short device count vector for generic Python dispatch."""

    if values.device.type == "cpu":
        # ``tolist`` uses one native conversion for the whole short vector;
        # iterating scalar tensors is substantially slower for large expert
        # banks and offers no synchronization benefit on CPU.
        return values.tolist()
    host_values = values.detach().to(device="cpu")
    return host_values.tolist()


dispatch_to_experts = sparse_dispatch


def _can_batch_linear_experts(experts: Sequence[nn.Module]) -> bool:
    """Return whether ``experts`` can use the pure batched-linear fast path.

    Restricting this optimization to exact ``nn.Linear`` instances with
    matching parameter metadata preserves the behavior of subclasses and
    modules carrying hooks. The caller separately guards the CUDA path when
    some experts are not routed: stacking every expert would otherwise turn
    an unused parameter's ``grad is None`` into a materialized zero grad.
    Those cases continue through the grouped dispatcher below.
    """

    if (
        not experts
        or any(type(expert) is not nn.Linear for expert in experts)
        or _has_global_module_hooks()
    ):
        return False
    linear_experts = [expert for expert in experts if isinstance(expert, nn.Linear)]
    first = linear_experts[0]
    if (
        _has_module_hooks(first)
        or _has_parameter_hooks(first)
        or getattr(first, "parametrizations", None)
        or "forward" in first.__dict__
        or "_call_impl" in first.__dict__
        or getattr(first, "_compiled_call_impl", None) is not None
    ):
        return False
    has_bias = first.bias is not None
    if first.weight.layout != torch.strided:
        return False
    if has_bias:
        assert first.bias is not None
        if (
            first.bias.device != first.weight.device
            or first.bias.dtype != first.weight.dtype
            or first.bias.layout != torch.strided
        ):
            return False
    for expert in linear_experts[1:]:
        if (
            expert.in_features != first.in_features
            or expert.out_features != first.out_features
            or (expert.bias is not None) != has_bias
            or expert.weight.device != first.weight.device
            or expert.weight.dtype != first.weight.dtype
            or expert.weight.layout != torch.strided
            or _has_module_hooks(expert)
            or _has_parameter_hooks(expert)
            or getattr(expert, "parametrizations", None)
            or "forward" in expert.__dict__
            or "_call_impl" in expert.__dict__
            or getattr(expert, "_compiled_call_impl", None) is not None
        ):
            return False
        if has_bias:
            assert expert.bias is not None and first.bias is not None
            if (
                expert.bias.device != first.bias.device
                or expert.bias.dtype != first.bias.dtype
                or expert.bias.layout != torch.strided
            ):
                return False
    return True


def _linear_route_memory_fits(
    route_count: int,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    stacked_expert_count: int = 0,
    budget_bytes: int = _LINEAR_ROUTE_WEIGHT_BUDGET_BYTES,
) -> bool:
    """Return whether route-parameter temporaries fit the fast-path budget.

    ``route_count`` is supplied from tensor shape metadata (rather than an
    active-route reduction), so checking the budget never synchronizes a CUDA
    scalar back to the host. ``stacked_expert_count`` accounts for the full
    ``torch.stack`` temporary used by the module-backed sparse path; callers
    that already own an expert-major weight tensor can leave it at zero. The
    estimate is deliberately conservative and includes both gathered weights
    and gathered bias when present.
    """

    parameter_bytes = weight.numel() * weight.element_size()
    bytes_required = (route_count + stacked_expert_count) * parameter_bytes
    if bias is not None:
        bias_bytes = bias.numel() * bias.element_size()
        bytes_required += (route_count + stacked_expert_count) * bias_bytes
    return bytes_required <= budget_bytes


def _linear_fast_path_preserves_unused_gradients(
    route_expert_indices: Tensor,
    *,
    active_routes: Tensor | None,
    num_experts: int,
    experts: Sequence[nn.Module],
) -> bool:
    """Guard route-batched dispatch against unused-parameter grad changes.

    The route-batched kernel stacks every expert parameter before gathering
    rows.  Autograd consequently creates a zero gradient for a parameter that
    was never routed, whereas grouped dispatch leaves that parameter's
    ``grad`` as ``None``.  Keep the no-synchronization fast path for inference
    and frozen experts. During training, perform one compact device-to-host
    check and use grouped dispatch whenever any expert has no active route.
    """

    if not torch.is_grad_enabled():
        return True

    # If no expert parameter participates in autograd, there is no observable
    # ``None`` versus zero-gradient difference to preserve. This also avoids a
    # needless device/host synchronization for frozen inference modules whose
    # input happens to require gradients.
    if not any(
        parameter.requires_grad
        for expert in experts
        for parameter in expert.parameters(recurse=False)
    ):
        return True

    # The caller has already restricted this helper to exact Linear modules;
    # requiring every expert to be routed is conservative for shared/frozen
    # mixtures and keeps the semantic check independent of parameter aliasing.
    if active_routes is None:
        routed_experts = route_expert_indices
    else:
        routed_experts = route_expert_indices[active_routes]
    if routed_experts.numel() == 0:
        return False

    routed_counts = torch.bincount(routed_experts, minlength=num_experts)
    # This is the sole synchronization introduced by the semantic guard. It
    # happens only while autograd is enabled and only on the CUDA fast-path
    # candidate; inference remains fully device-local.
    return bool(routed_counts.gt(0).all().item())


def _has_global_module_hooks() -> bool:
    """Whether PyTorch global module hooks would observe expert execution."""

    return any(
        bool(getattr(nn_module, name, ()))
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_forward_hooks_with_kwargs",
            "_global_forward_hooks_always_called",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    )


def _has_module_hooks(module: nn.Module) -> bool:
    """Return whether a module-local hook can observe its forward/backward."""

    return any(
        bool(getattr(module, name, ()))
        for name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks_always_called",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    )


def _has_parameter_hooks(module: nn.Module) -> bool:
    """Return whether a parameter hook would observe expert gradients."""

    return any(
        bool(getattr(parameter, hook_name, ()))
        for parameter in module.parameters(recurse=False)
        for hook_name in ("_backward_hooks", "_post_accumulate_grad_hooks")
    )


def _sparse_dispatch_linear(
    flat_hidden: Tensor,
    *,
    leading_shape: torch.Size,
    route_token_indices: Tensor,
    route_expert_indices: Tensor,
    route_weights: Tensor,
    active_routes: Tensor | None,
    original_route_weights: Tensor,
    tokens_per_expert: Tensor,
    experts: Sequence[nn.Module],
) -> tuple[Tensor, Tensor]:
    """Dispatch homogeneous ``nn.Linear`` experts with one batched matmul."""

    if active_routes is not None:
        active_indices = active_routes.nonzero(as_tuple=False).flatten()
        route_token_indices = route_token_indices.index_select(0, active_indices)
        route_expert_indices = route_expert_indices.index_select(0, active_indices)
        route_weights = route_weights.index_select(0, active_indices)

    if route_token_indices.numel() == 0:
        # Preserve zero-gradient edges and avoid reading masked NaN/Inf inputs;
        # the empty probe also obtains the module's output feature shape.
        output = _empty_output(flat_hidden, route_weights, experts)
        if active_routes is not None:
            output = output + original_route_weights.unsqueeze(0)[:0].sum().to(
                dtype=output.dtype
            )
        return output.reshape(*leading_shape, *output.shape[1:]), tokens_per_expert

    linear_experts = [expert for expert in experts if isinstance(expert, nn.Linear)]
    # _can_batch_linear_experts() checked exact type and matching metadata.
    stacked_weight = torch.stack(
        [expert.weight for expert in linear_experts],
        dim=0,
    )
    route_weight = stacked_weight.index_select(0, route_expert_indices)
    route_input = flat_hidden.index_select(0, route_token_indices)
    if linear_experts[0].bias is not None:
        stacked_bias = torch.stack(
            [
                expert.bias
                for expert in linear_experts
                if expert.bias is not None
            ],
            dim=0,
        )
        route_bias = stacked_bias.index_select(0, route_expert_indices)
        # ``baddbmm`` follows F.linear's addmm accumulation (including CPU
        # autocast) while evaluating every route in one batched kernel.
        expert_output = torch.baddbmm(
            route_bias.unsqueeze(1),
            route_input.unsqueeze(1),
            route_weight.transpose(1, 2),
        ).squeeze(1)
    else:
        expert_output = torch.bmm(
            route_input.unsqueeze(1),
            route_weight.transpose(1, 2),
        ).squeeze(1)

    combine_dtype = torch.promote_types(expert_output.dtype, route_weights.dtype)
    weighted_output = expert_output.to(dtype=combine_dtype) * route_weights.to(
        dtype=combine_dtype
    ).reshape(-1, 1)
    output = weighted_output.new_zeros(
        flat_hidden.size(0),
        expert_output.size(-1),
    )
    output.index_add_(0, route_token_indices, weighted_output)
    if active_routes is not None:
        # Preserve a zero-gradient edge for masked route weights that were
        # filtered before the batched kernel, without reading their values.
        output = output + original_route_weights.unsqueeze(0)[:0].sum().to(
            dtype=output.dtype
        )
    return output.reshape(*leading_shape, *output.shape[1:]), tokens_per_expert


__all__ = ["dispatch_to_experts", "sparse_dispatch"]
