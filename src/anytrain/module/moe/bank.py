"""Parameter-bank executor: ``expert(input, weight, bias)`` with MoE-owned params."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn

from . import _bank_conv, _bank_linear
from ._bank_routes import _dense_probabilities
from .conv.bank import mix_expert_bias
from .expert import (
    BiasKind,
    Conv1dExpert,
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
                f"weight must include an expert axis as dim 0, got shape {tuple(weight.shape)}."
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
            raise ValueError("ParamExpertBank bias_kind is 'none' but bias was provided.")
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
        execution_cells = cells
        if (
            routing.is_dense
            and routing.mask is not None
            and self.family
            in (
                ExpertFamily.CONV1D,
                ExpertFamily.CONV_TRANSPOSE1D,
                ExpertFamily.CONV2D,
            )
        ):
            execution_cells = _mask_cell_outputs(cells, routing)
        if self.family is ExpertFamily.LINEAR:
            outputs = _bank_linear.linear(
                self.expert, self.weight, self.bias, self.num_experts, execution_cells, routing
            )
        elif self.family is ExpertFamily.CONV1D:
            outputs = _bank_conv.conv1d(self.expert, self.weight, execution_cells, routing)
        elif self.family is ExpertFamily.CONV_TRANSPOSE1D:
            outputs = _bank_conv.conv_transpose1d(
                self.expert, self.weight, execution_cells, routing
            )
        elif self.family is ExpertFamily.CONV2D:
            outputs = _bank_conv.conv2d(self.expert, self.weight, execution_cells, routing)
        else:
            outputs = _bank_linear.generic(
                self.expert, self.weight, self.num_experts, execution_cells, routing
            )
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
            probs = _dense_probabilities(routing)
            return mix_expert_bias(probs, self.bias)
        indices, selected_weights, top_k = routing.require_topk()
        flat_indices = indices.reshape(-1, top_k)
        flat_weights = selected_weights.reshape(-1, top_k)
        if routing.mask is not None:
            active = routing.flat_active_mask().unsqueeze(-1)
            flat_weights = flat_weights.masked_fill(~active, 0)
        selected_bias = self.bias[flat_indices]
        combine_dtype = torch.promote_types(flat_weights.dtype, selected_bias.dtype)
        return torch.einsum(
            "uk,uko->uo",
            flat_weights.to(dtype=combine_dtype),
            selected_bias.to(dtype=combine_dtype),
        )


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
            f"1d conv bias restore requires a layout with restore(), got {type(layout)!r}."
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
