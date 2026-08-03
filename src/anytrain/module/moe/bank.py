"""Parameter-bank executor: ``expert(input, weight, bias)`` with MoE-owned params."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, cast

import torch
from torch import Tensor, nn

from .conv.bank import mix_expert_bias, mix_expert_weights
from .conv.expert_bank import grouped_conv1d, grouped_conv2d, grouped_conv_transpose1d
from .expert import (
    BiasKind,
    Conv1dExpert,
    Conv2dExpert,
    ConvTranspose1dExpert,
    Expert,
    ExpertFamily,
)
from .routing import ExpertRouting, validate_expert_routing, validate_num_experts


class ParamExpertBank(nn.Module):
    """Runs a shared :class:`Expert` over expert-major ``weight`` / ``bias``.

    Cell execution always uses the weight-only path (``bias=None`` / no bias in
    grouped kernels) when ``bias_kind == "additive_channel"``. After
    ``layout.restore``, :meth:`apply_bias` adds the mixed expert bias as an
    additive channel term — the same meaning as ``nn.Linear`` / ``nn.Conv*``
    bias. Conv1d and ConvTranspose1d share one restore path:
    ``restore(bias_cells) / restore(coverage_cells)`` (coverage is 1 for concat
    grids, >1 on transpose overlaps).
    """

    expert: Expert
    weight: nn.Parameter
    bias: nn.Parameter | None
    family: ExpertFamily
    bias_kind: BiasKind

    def __init__(
        self,
        expert: Expert,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        family: ExpertFamily,
        bias_kind: BiasKind = "additive_channel",
    ) -> None:
        super().__init__()
        validate_num_experts(num_experts)
        if weight.ndim < 1:
            raise ValueError(
                "weight must include an expert axis as dim 0, "
                f"got shape {tuple(weight.shape)}."
            )
        if int(weight.shape[0]) != num_experts:
            raise ValueError(
                "weight.shape[0] must equal num_experts: "
                f"got {weight.shape[0]}, expected {num_experts}."
            )
        if bias is not None and (bias.ndim < 1 or int(bias.shape[0]) != num_experts):
            raise ValueError(
                "bias.shape[0] must equal num_experts: "
                f"got {tuple(bias.shape)}, expected ({num_experts}, ...)."
            )
        if bias_kind == "none" and bias is not None:
            raise ValueError(
                "ParamExpertBank bias_kind is 'none' but bias was provided."
            )
        self.expert = expert
        self.num_experts = num_experts
        self.family = family
        self.bias_kind = bias_kind
        self.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight)
        if bias is None:
            self.register_parameter("bias", None)
            self.bias = None
        else:
            self.bias = bias if isinstance(bias, nn.Parameter) else nn.Parameter(bias)

    def forward(
        self,
        cells: Tensor,
        routing: ExpertRouting,
        *,
        check_index_bounds: bool = True,
    ) -> Tensor:
        leading = self._leading_shape(cells)
        validate_expert_routing(
            routing,
            leading_shape=leading,
            num_experts=self.num_experts,
            device=cells.device,
            require_topk=not routing.is_dense,
            check_index_bounds=check_index_bounds,
        )
        if routing.leading_shape != leading:
            raise ValueError(
                "routing leading shape must match cells layout: "
                f"got {tuple(routing.leading_shape)}, expected {tuple(leading)}."
            )
        if self.family is ExpertFamily.LINEAR:
            outputs = self._forward_linear(cells, routing)
        elif self.family is ExpertFamily.CONV1D:
            outputs = self._forward_conv1d(cells, routing)
        elif self.family is ExpertFamily.CONV_TRANSPOSE1D:
            outputs = self._forward_conv_transpose1d(cells, routing)
        elif self.family is ExpertFamily.CONV2D:
            outputs = self._forward_conv2d(cells, routing)
        else:
            outputs = self._forward_generic(cells, routing)
        return _mask_cell_outputs(outputs, routing)

    def apply_bias(
        self,
        hidden_states: Tensor,
        routing: ExpertRouting,
        layout: object,
        *,
        cell_outputs: Tensor,
    ) -> Tensor:
        if self.bias_kind != "additive_channel":
            return hidden_states
        mixed_bias = self._mixed_bias(routing)
        if mixed_bias is None:
            return hidden_states
        active = None
        if routing.mask is not None:
            active = routing.flat_active_mask()
            mixed_bias = _mask_mixed_bias(mixed_bias, active)
        if self.family in (ExpertFamily.CONV1D, ExpertFamily.CONV_TRANSPOSE1D):
            return _apply_1d_bias(
                hidden_states,
                mixed_bias,
                layout,
                cell_outputs=cell_outputs,
                active_cells=active,
            )
        if self.family is ExpertFamily.CONV2D:
            return hidden_states + mixed_bias[..., None, None]
        return _add_feature_last_bias(hidden_states, mixed_bias, routing)

    def _leading_shape(self, cells: Tensor) -> torch.Size:
        if self.family in (
            ExpertFamily.CONV1D,
            ExpertFamily.CONV2D,
            ExpertFamily.CONV_TRANSPOSE1D,
        ):
            return cells.shape[:1]
        return cells.shape[:-1]

    def _mixed_bias(self, routing: ExpertRouting) -> Tensor | None:
        if self.bias is None:
            return None
        if routing.is_dense:
            probs = routing.probabilities.reshape(-1, self.num_experts)
            return mix_expert_bias(probs, self.bias)
        indices, selected_weights, top_k = routing.require_topk()
        flat_indices = indices.reshape(-1, top_k)
        flat_weights = selected_weights.reshape(-1, top_k)
        return torch.einsum("uk,uko->uo", flat_weights, self.bias[flat_indices])

    def _forward_linear(self, cells: Tensor, routing: ExpertRouting) -> Tensor:
        leading_shape = cells.shape[:-1]
        flat = cells.reshape(-1, cells.size(-1))
        if routing.is_dense:
            probs = routing.probabilities.reshape(-1, self.num_experts)
            y = None
            for expert_index in range(self.num_experts):
                expert_y = self.expert(
                    flat,
                    self.weight[expert_index],
                    None,
                )
                if y is None:
                    y = torch.zeros_like(expert_y)
                combine_weight = probs[:, expert_index].to(dtype=expert_y.dtype)
                y.add_(expert_y * combine_weight.unsqueeze(-1))
            assert y is not None
        else:
            route_tokens, route_weights, token_counts = self._grouped_topk_routes(routing)
            y = None
            start = 0
            for expert_index, token_count in enumerate(token_counts):
                if token_count == 0:
                    continue
                end = start + token_count
                token_indices = route_tokens[start:end]
                expert_input = flat.index_select(0, token_indices)
                expert_y = self.expert(
                    expert_input,
                    self.weight[expert_index],
                    None,
                )
                if y is None:
                    y = expert_y.new_zeros(flat.size(0), expert_y.size(-1))
                combine_weight = route_weights[start:end].to(dtype=expert_y.dtype)
                y.index_add_(
                    0,
                    token_indices,
                    expert_y * combine_weight.unsqueeze(-1),
                )
                start = end
            if y is None:
                probe = self.expert(flat, self.weight[0], None)
                zero = route_weights.sum().to(dtype=probe.dtype) * 0
                y = probe * zero
        return y.reshape(*leading_shape, y.size(-1))

    def _forward_conv1d(self, cells: Tensor, routing: ExpertRouting) -> Tensor:
        expert = self.expert
        if not isinstance(expert, Conv1dExpert):
            raise TypeError(
                f"CONV1D bank requires Conv1dExpert, got {type(expert)!r}."
            )
        if cells.ndim != 3:
            raise ValueError(
                f"Conv1dExpert expects cells shaped (U, C, T), got {tuple(cells.shape)}."
            )
        if routing.is_dense:
            probs = routing.probabilities.reshape(-1, self.num_experts)
            mixed_weight = mix_expert_weights(probs, self.weight)
            return grouped_conv1d(
                cells,
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
            self.weight[indices.reshape(-1)],
            stride=expert.stride,
            padding=expert.padding,
            dilation=expert.dilation,
            groups=expert.groups,
        )
        expert_outputs = expert_outputs.reshape(
            cells.size(0), top_k, expert_outputs.size(1), expert_outputs.size(-1)
        )
        return torch.einsum("uk,ukot->uot", selected_weights, expert_outputs)

    def _forward_conv_transpose1d(self, cells: Tensor, routing: ExpertRouting) -> Tensor:
        expert = self.expert
        if not isinstance(expert, ConvTranspose1dExpert):
            raise TypeError(
                "CONV_TRANSPOSE1D bank requires ConvTranspose1dExpert, "
                f"got {type(expert)!r}."
            )
        if cells.ndim != 3:
            raise ValueError(
                "ConvTranspose1dExpert expects cells shaped (U, C, T), "
                f"got {tuple(cells.shape)}."
            )
        if routing.is_dense:
            probs = routing.probabilities.reshape(-1, self.num_experts)
            mixed_weight = mix_expert_weights(probs, self.weight)
            return grouped_conv_transpose1d(
                cells,
                mixed_weight,
                stride=expert.stride,
                dilation=expert.dilation,
                groups=expert.groups,
            )
        indices, selected_weights, top_k = routing.require_topk()
        cells_rep = cells.repeat_interleave(top_k, dim=0)
        expert_outputs = grouped_conv_transpose1d(
            cells_rep,
            self.weight[indices.reshape(-1)],
            stride=expert.stride,
            dilation=expert.dilation,
            groups=expert.groups,
        )
        expert_outputs = expert_outputs.reshape(
            cells.size(0), top_k, expert_outputs.size(1), expert_outputs.size(-1)
        )
        return torch.einsum("uk,ukot->uot", selected_weights, expert_outputs)

    def _forward_conv2d(self, cells: Tensor, routing: ExpertRouting) -> Tensor:
        expert = self.expert
        if not isinstance(expert, Conv2dExpert):
            raise TypeError(
                f"CONV2D bank requires Conv2dExpert, got {type(expert)!r}."
            )
        if cells.ndim != 4:
            raise ValueError(
                f"Conv2dExpert expects cells shaped (U, C, H, W), got {tuple(cells.shape)}."
            )
        if routing.is_dense:
            probs = routing.probabilities.reshape(-1, self.num_experts)
            mixed_weight = mix_expert_weights(probs, self.weight)
            return grouped_conv2d(
                cells,
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
            self.weight[indices.reshape(-1)],
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
        return torch.einsum("uk,ukohw->uohw", selected_weights, expert_outputs)

    def _forward_generic(self, cells: Tensor, routing: ExpertRouting) -> Tensor:
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
            probs = routing.probabilities.reshape(flat_count, self.num_experts)
            outputs = [
                self.expert(flat_cells, self.weight[expert_index], None)
                for expert_index in range(self.num_experts)
            ]
            stacked = torch.stack(outputs, dim=1)
            weight_view = probs.reshape(flat_count, self.num_experts, *([1] * (stacked.ndim - 2)))
            mixed = (weight_view * stacked).sum(dim=1)
        else:
            route_tokens, route_weights, token_counts = self._grouped_topk_routes(routing)
            mixed = None
            output_tail_shape = None
            start = 0
            for expert_index, token_count in enumerate(token_counts):
                if token_count == 0:
                    continue
                end = start + token_count
                token_indices = route_tokens[start:end]
                expert_output = self.expert(
                    flat_cells.index_select(0, token_indices),
                    self.weight[expert_index],
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
                probe = self.expert(flat_cells, self.weight[0], None)
                _validate_generic_expert_output(
                    probe,
                    token_count=flat_count,
                    expert_index=0,
                )
                combine_dtype = torch.promote_types(
                    probe.dtype,
                    route_weights.dtype,
                )
                zero = route_weights.sum().to(dtype=combine_dtype) * 0
                mixed = probe.to(dtype=combine_dtype) * zero
        return mixed.reshape(*leading, *mixed.shape[1:])

    def _grouped_topk_routes(
        self,
        routing: ExpertRouting,
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
            sentinel = torch.full_like(route_experts, self.num_experts)
            group_keys = torch.where(active.reshape(-1), route_experts, sentinel)
        order = group_keys.argsort(stable=True)
        route_tokens = route_tokens.index_select(0, order)
        route_weights = route_weights.index_select(0, order)
        group_keys = group_keys.index_select(0, order)
        counts = torch.bincount(group_keys, minlength=self.num_experts + 1)
        # Calling distinct expert parameters needs segment sizes on the host.
        # This is one synchronization per bank call, independent of route count.
        token_counts = counts[: self.num_experts].tolist()
        return route_tokens, route_weights, token_counts


def with_execution_padding(expert: Conv1dExpert, *, execution_padding: int) -> Conv1dExpert:
    """Expert used inside the bank after the partitioner has applied logical pad."""

    if expert.padding == execution_padding:
        return expert
    return replace(expert, padding=execution_padding)


def _add_feature_last_bias(
    hidden_states: Tensor,
    mixed_bias: Tensor,
    routing: ExpertRouting,
) -> Tensor:
    leading = routing.leading_shape
    flat_bias = mixed_bias.reshape(*leading, *mixed_bias.shape[1:])
    return hidden_states + flat_bias


def _mask_cell_outputs(outputs: Tensor, routing: ExpertRouting) -> Tensor:
    if routing.mask is None:
        return outputs
    tail_ndim = outputs.ndim - len(routing.leading_shape)
    active = routing.flat_active_mask().reshape(
        *routing.leading_shape,
        *([1] * tail_ndim),
    )
    return outputs.masked_fill(~active, 0)


def _mask_mixed_bias(mixed_bias: Tensor, active: Tensor) -> Tensor:
    active = active.reshape(
        -1,
        *([1] * (mixed_bias.ndim - 1)),
    )
    return mixed_bias.masked_fill(~active, 0)


def _validate_generic_expert_output(
    output: Tensor,
    *,
    token_count: int,
    expert_index: int,
) -> None:
    if not isinstance(output, Tensor):
        raise TypeError(f"expert {expert_index} must return a Tensor.")
    if output.ndim < 1:
        raise ValueError(
            f"expert {expert_index} must return a tensor with a batch dimension."
        )
    if output.size(0) != token_count:
        raise ValueError(
            "expert output batch size must match routed token count: "
            f"expert {expert_index} returned {output.size(0)}, expected {token_count}."
        )


def _apply_1d_bias(
    hidden_states: Tensor,
    mixed_bias: Tensor,
    layout: object,
    *,
    cell_outputs: Tensor,
    active_cells: Tensor | None,
) -> Tensor:
    """Additive channel bias after the same ``layout.restore`` as activations.

    Shared by Conv1d and ConvTranspose1d. Coverage from ``restore(coverage_cells)`` is 1
    for concat segments and >1 on transpose overlap-add regions.
    """

    restore = getattr(layout, "restore", None)
    if not callable(restore):
        raise TypeError(
            "1d conv bias restore requires a layout with restore(), "
            f"got {type(layout)!r}."
        )
    restore_tensor = cast(Callable[[Tensor], Tensor], restore)
    time = cell_outputs.size(-1)
    bias_cells = mixed_bias[..., None].expand(-1, -1, time)
    restored_bias = restore_tensor(bias_cells)
    if active_cells is None:
        coverage_cells = bias_cells.new_ones(mixed_bias.size(0), 1, time)
    else:
        coverage_cells = active_cells.reshape(-1, 1, 1).to(dtype=bias_cells.dtype)
        coverage_cells = coverage_cells.expand(-1, 1, time)
    coverage = restore_tensor(coverage_cells).clamp_min(1)
    return hidden_states + restored_bias / coverage


__all__ = ["ParamExpertBank", "with_execution_padding"]
