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

    def forward(self, cells: Tensor, routing: ExpertRouting) -> Tensor:
        leading = self._leading_shape(cells)
        validate_expert_routing(
            routing,
            leading_shape=leading,
            num_experts=self.num_experts,
            device=cells.device,
            require_topk=not routing.is_dense,
        )
        if routing.leading_shape != leading:
            raise ValueError(
                "routing leading shape must match cells layout: "
                f"got {tuple(routing.leading_shape)}, expected {tuple(leading)}."
            )
        if self.family is ExpertFamily.LINEAR:
            return self._forward_linear(cells, routing)
        if self.family is ExpertFamily.CONV1D:
            return self._forward_conv1d(cells, routing)
        if self.family is ExpertFamily.CONV_TRANSPOSE1D:
            return self._forward_conv_transpose1d(cells, routing)
        if self.family is ExpertFamily.CONV2D:
            return self._forward_conv2d(cells, routing)
        return self._forward_generic(cells, routing)

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
        if self.family in (ExpertFamily.CONV1D, ExpertFamily.CONV_TRANSPOSE1D):
            return _apply_1d_bias(
                hidden_states, mixed_bias, layout, cell_outputs=cell_outputs
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
        active = routing.flat_active_mask().to(dtype=flat.dtype).unsqueeze(-1)
        if routing.is_dense:
            probs = routing.probabilities.reshape(-1, self.num_experts)
            mixed_weight = mix_expert_weights(probs, self.weight)
            y = torch.einsum("ni,noi->no", flat, mixed_weight)
        else:
            indices, selected_weights, top_k = routing.require_topk()
            flat_indices = indices.reshape(-1, top_k)
            flat_weights = selected_weights.reshape(-1, top_k)
            selected_weight = self.weight[flat_indices]
            expert_y = torch.einsum("ni,nkoi->nko", flat, selected_weight)
            y = torch.einsum("nk,nko->no", flat_weights, expert_y)
        return (y * active).reshape(*leading_shape, y.size(-1))

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
        active = routing.flat_active_mask().to(dtype=flat_cells.dtype)
        active_view = active.reshape(flat_count, *([1] * (flat_cells.ndim - 1)))

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
            indices, selected_weights, top_k = routing.require_topk()
            flat_indices = indices.reshape(flat_count, top_k)
            flat_weights = selected_weights.reshape(flat_count, top_k)
            slot_outputs = []
            for slot in range(top_k):
                expert_indices = flat_indices[:, slot]
                slot_y = torch.stack(
                    [
                        self.expert(
                            flat_cells[row : row + 1],
                            self.weight[int(expert_indices[row].item())],
                            None,
                        ).squeeze(0)
                        for row in range(flat_count)
                    ],
                    dim=0,
                )
                slot_outputs.append(slot_y)
            stacked = torch.stack(slot_outputs, dim=1)
            weight_view = flat_weights.reshape(
                flat_count, top_k, *([1] * (stacked.ndim - 2))
            )
            mixed = (weight_view * stacked).sum(dim=1)
        return (mixed * active_view).reshape(*leading, *mixed.shape[1:])


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


def _apply_1d_bias(
    hidden_states: Tensor,
    mixed_bias: Tensor,
    layout: object,
    *,
    cell_outputs: Tensor,
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
    coverage_cells = bias_cells.new_ones(bias_cells.size(0), 1, time)
    coverage = restore_tensor(coverage_cells).clamp_min(1)
    return hidden_states + restored_bias / coverage


__all__ = ["ParamExpertBank", "with_execution_padding"]
