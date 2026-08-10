from __future__ import annotations

import unittest
from typing import Any, Literal, cast

import torch
from torch import nn

from anytrain.module.dirichlet_tempering import AdaptiveDirichletTempering
from anytrain.module.moe import (
    Conv1dExpert,
    ExpertRouting,
    Grid1d,
    MoE,
    ParamExpertBank,
    expand_expert,
    expert_routing_from_weights,
)
from anytrain.module.moe.conv import (
    Conv1d,
    ConvRouter1d,
    ConvRouter2d,
    effective_kernel_size_1d,
    effective_kernel_size_2d,
    infer_padding_1d,
    infer_padding_2d,
    validate_dynamic_conv1d_args,
)
from anytrain.module.moe.conv.bank import expand_expert_weights
from anytrain.module.moe.routing import DenseExpertWeights


class MeanPoolRouter1d(nn.Module):
    def __init__(self, in_channels: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.proj = nn.Linear(in_channels, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=-1))


class MeanPoolRouter2d(nn.Module):
    def __init__(self, in_channels: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.proj = nn.Linear(in_channels, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1)))


def make_conv1d_moe(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    num_experts: int = 4,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding: int | None = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    segment_size: int | None = None,
    causal: bool = False,
    context: str | None = None,
    fill: str | None = None,
    router: nn.Module | None = None,
) -> MoE:
    if padding is None:
        padding_size = infer_padding_1d(
            effective_kernel_size_1d(torch.Size([kernel_size]), torch.Size([dilation])),
            torch.Size([stride]),
        )
        padding = int(padding_size[0])
    if segment_size is not None:
        validate_dynamic_conv1d_args(
            kernel_size=torch.Size([kernel_size]),
            stride=torch.Size([stride]),
            padding=torch.Size([padding]),
            dilation=torch.Size([dilation]),
            segment_size=torch.Size([segment_size]),
        )
    if router is None:
        router = MeanPoolRouter1d(in_channels, num_experts)
    grid = None
    if segment_size is not None:
        grid = Grid1d(segment_size, context=context, fill=fill)  # type: ignore[arg-type]
    template: nn.Module
    if causal:
        template = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            causal=True,
        )
    else:
        template = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    expert, weight, bias_param = expand_expert(template, num_experts, init="kaiming")
    return MoE(
        num_experts=num_experts,
        router=router,
        expert=expert,
        weight=weight,
        bias=bias_param,
        grid=grid,
        top_k=None,
    )


def make_conv_transpose1d_moe(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    num_experts: int = 4,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding: int | None = None,
    output_padding: int = 0,
    segment_size: int | None = None,
    router: nn.Module | None = None,
) -> MoE:
    if padding is None:
        padding_size = infer_padding_1d(
            effective_kernel_size_1d(torch.Size([kernel_size]), torch.Size([dilation])),
            torch.Size([stride]),
        )
        padding = int(padding_size[0])
    if router is None:
        router = MeanPoolRouter1d(in_channels, num_experts)
    grid = None if segment_size is None else Grid1d(segment_size)
    expert, weight, bias_param = expand_expert(
        nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        num_experts,
        init="kaiming",
    )
    return MoE(
        num_experts=num_experts,
        router=router,
        expert=expert,
        weight=weight,
        bias=bias_param,
        grid=grid,
        top_k=None,
    )


def make_conv2d_moe(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    num_experts: int = 4,
    stride: int | tuple[int, int] = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding: int | tuple[int, int] | None = None,
    router: nn.Module | None = None,
) -> MoE:
    kernel = (
        torch.Size([kernel_size, kernel_size])
        if isinstance(kernel_size, int)
        else torch.Size(kernel_size)
    )
    stride_size = torch.Size([stride, stride]) if isinstance(stride, int) else torch.Size(stride)
    dilation_size = torch.Size([dilation, dilation])
    if padding is None:
        padding_size = infer_padding_2d(
            effective_kernel_size_2d(kernel, dilation_size),
            stride_size,
        )
    elif isinstance(padding, int):
        padding_size = torch.Size([padding, padding])
    else:
        padding_size = torch.Size(padding)
    if router is None:
        router = MeanPoolRouter2d(in_channels, num_experts)
    expert, weight, bias_param = expand_expert(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=cast(tuple[int, int], tuple(kernel)),
            stride=cast(tuple[int, int], tuple(stride_size)),
            padding=cast(tuple[int, int], tuple(padding_size)),
            dilation=cast(tuple[int, int], tuple(dilation_size)),
            groups=groups,
            bias=bias,
        ),
        num_experts,
        init="kaiming",
    )
    return MoE(
        num_experts=num_experts,
        router=router,
        expert=expert,
        weight=weight,
        bias=bias_param,
        top_k=None,
    )


def run_manual(moe: MoE, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    cells, _features, layout = moe.partitioner.partition(x)
    batch_size = getattr(layout, "batch_size", cells.size(0))
    num_segments = getattr(layout, "num_segments", 1)
    weights = expand_expert_weights(
        expert_weights,
        batch_size=cells.size(0),
        original_batch_size=batch_size,
        num_segments=num_segments if isinstance(num_segments, int) else num_segments[0],
    )
    return moe.forward_manually(x, expert_routing_from_weights(weights)).hidden_states


class Conv1dMoETest(unittest.TestCase):
    def test_initialization_shapes(self):
        moe = make_conv1d_moe(16, 32, kernel_size=3, num_experts=4)
        bank = moe.expert_bank
        self.assertIsInstance(bank, ParamExpertBank)
        self.assertIsInstance(bank.expert, Conv1dExpert)
        self.assertEqual(bank.weight.shape, (4, 32, 16, 3))
        assert bank.bias is not None
        self.assertEqual(bank.bias.shape, (4, 32))

    def test_forward_preserves_length_with_stride_one(self):
        moe = make_conv1d_moe(16, 32, kernel_size=3, num_experts=4)
        y = moe(torch.randn(2, 16, 100)).hidden_states
        self.assertEqual(y.shape, (2, 32, 100))
        self.assertTrue(torch.isfinite(y).all())

    def test_sparse_mixes_expert_outputs_not_parameters(self):
        torch.manual_seed(0)
        moe = make_conv1d_moe(4, 5, kernel_size=3, num_experts=3, padding=1)
        bank = moe.expert_bank
        x = torch.randn(2, 4, 16)
        routing = ExpertRouting(
            logits=torch.tensor([[2.0, 1.0, -1.0], [0.0, 3.0, 1.0]]),
            probabilities=torch.full((2, 3), 1 / 3),
            indices=torch.tensor([[0, 1], [1, 2]]),
            weights=torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
        )
        y = moe.forward_manually(x, routing).hidden_states

        cells, _features, layout = moe.partitioner.partition(x)
        bank_padding = int(cast(Any, bank.expert).padding)
        assert routing.indices is not None
        assert routing.weights is not None
        expected_cells = []
        for cell_index in range(cells.size(0)):
            mixed = None
            for slot in range(2):
                expert_index = int(routing.indices[cell_index, slot].item())
                expert_y = torch.nn.functional.conv1d(
                    cells[cell_index : cell_index + 1],
                    bank.weight[expert_index],
                    bias=None,
                    padding=bank_padding,
                )
                scale = routing.weights[cell_index, slot]
                mixed = expert_y * scale if mixed is None else mixed + expert_y * scale
            assert mixed is not None
            expected_cells.append(mixed)
        cell_outputs = torch.cat(expected_cells, dim=0)
        expected = layout.restore(cell_outputs)
        assert bank.bias is not None
        mixed_bias = torch.einsum("uk,uko->uo", routing.weights, bank.bias[routing.indices])
        expected = expected + mixed_bias[..., None]
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    def test_gradient_flow_to_router_and_experts(self):
        router = MeanPoolRouter1d(16, 4)
        moe = make_conv1d_moe(16, 32, kernel_size=3, num_experts=4, router=router)
        moe(torch.randn(2, 16, 24)).hidden_states.sum().backward()
        bank = moe.expert_bank
        self.assertIsNotNone(bank.weight.grad)
        self.assertIsNotNone(router.proj.weight.grad)

    def test_segmented_forward_preserves_length(self):
        moe = make_conv1d_moe(16, 32, kernel_size=3, num_experts=4, segment_size=8)
        y = moe(torch.randn(2, 16, 100)).hidden_states
        self.assertEqual(y.shape, (2, 32, 100))

    def test_from_module_conv1d_copies_template_and_builds_default_router(self):
        torch.manual_seed(0)
        template = nn.Conv1d(
            8,
            8,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        moe = MoE.from_module(
            template,
            num_experts=3,
            grid=Grid1d(segment_size=8),
            top_k=None,
        )
        x = torch.randn(2, 8, 24)

        y = moe(x).hidden_states

        self.assertIsInstance(moe.router, ConvRouter1d)
        self.assertTrue(torch.allclose(y, template(x), atol=1e-5))

    def test_segmented_global_context_matches_full_conv_padding_modes(self):
        for padding_mode in ("zeros", "reflect", "replicate"):
            with self.subTest(padding_mode=padding_mode):
                full = make_conv1d_moe(
                    8, 8, kernel_size=3, num_experts=2, padding_mode=padding_mode
                )
                segmented = make_conv1d_moe(
                    8,
                    8,
                    kernel_size=3,
                    num_experts=2,
                    segment_size=8,
                    padding_mode=padding_mode,
                )
                bank_full = full.expert_bank
                bank_seg = segmented.expert_bank
                with torch.no_grad():
                    bank_seg.weight.copy_(bank_full.weight)
                    assert bank_seg.bias is not None and bank_full.bias is not None
                    bank_seg.bias.copy_(bank_full.bias)
                x = torch.randn(2, 8, 24)
                weights = torch.softmax(torch.randn(2, 2), dim=-1)
                self.assertTrue(
                    torch.allclose(
                        run_manual(full, x, weights),
                        run_manual(segmented, x, weights),
                        atol=1e-5,
                    )
                )

    def test_forward_manually_accepts_broadcast_and_dense_weights(self):
        moe = make_conv1d_moe(16, 32, kernel_size=3, num_experts=4)
        x = torch.randn(2, 16, 24)
        y = run_manual(moe, x, torch.ones(1, 4) / 4)
        self.assertEqual(y.shape, (2, 32, 24))
        y2 = run_manual(moe, x, DenseExpertWeights(torch.ones(2, 4) / 4).weights)
        self.assertEqual(y2.shape, (2, 32, 24))

    def test_forward_manually_repeats_batch_weights_across_segments(self):
        moe = make_conv1d_moe(4, 4, kernel_size=3, num_experts=2, segment_size=8)
        x = torch.randn(2, 4, 16)
        y = run_manual(moe, x, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(y.shape, (2, 4, 16))


class ConvTranspose1dMoETest(unittest.TestCase):
    def test_forward_preserves_length_with_stride_one(self):
        moe = make_conv_transpose1d_moe(16, 32, kernel_size=3, num_experts=4)
        y = moe(torch.randn(2, 16, 40)).hidden_states
        self.assertEqual(y.shape, (2, 32, 40))

    def test_segmented_forward_matches_full_conv_with_bias(self):
        torch.manual_seed(0)
        full = make_conv_transpose1d_moe(8, 8, kernel_size=3, num_experts=2)
        segmented = make_conv_transpose1d_moe(8, 8, kernel_size=3, num_experts=2, segment_size=8)
        bank_full = full.expert_bank
        bank_seg = segmented.expert_bank
        with torch.no_grad():
            bank_seg.weight.copy_(bank_full.weight)
            assert bank_seg.bias is not None and bank_full.bias is not None
            bank_seg.bias.copy_(bank_full.bias)
        x = torch.randn(2, 8, 16)
        weights = torch.softmax(torch.randn(2, 2), dim=-1)
        self.assertTrue(
            torch.allclose(
                run_manual(full, x, weights),
                run_manual(segmented, x, weights),
                atol=1e-5,
            )
        )

    def test_stride_two_upsamples(self):
        moe = make_conv_transpose1d_moe(16, 32, kernel_size=4, num_experts=2, stride=2)
        y = moe(torch.randn(2, 16, 20)).hidden_states
        self.assertEqual(y.shape, (2, 32, 40))


class Conv2dMoETest(unittest.TestCase):
    def test_forward_preserves_size_with_stride_one(self):
        moe = make_conv2d_moe(16, 32, kernel_size=3, num_experts=4)
        y = moe(torch.randn(2, 16, 13, 17)).hidden_states
        self.assertEqual(y.shape, (2, 32, 13, 17))

    def test_all_experts_copy_matches_static_conv2d(self):
        torch.manual_seed(0)
        for groups in (1, 2):
            with self.subTest(groups=groups):
                moe = make_conv2d_moe(4, 6, kernel_size=3, num_experts=3, groups=groups)
                bank = moe.expert_bank
                static_weight = torch.randn(6, 4 // groups, 3, 3)
                static_bias = torch.randn(6)
                with torch.no_grad():
                    bank.weight.copy_(static_weight.unsqueeze(0).expand_as(bank.weight))
                    assert bank.bias is not None
                    bank.bias.copy_(static_bias.unsqueeze(0).expand_as(bank.bias))
                x = torch.randn(5, 4, 11, 13)
                expert_weights = torch.rand(5, 3)
                expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
                y = run_manual(moe, x, expert_weights)
                expected = torch.nn.functional.conv2d(
                    x,
                    static_weight,
                    static_bias,
                    padding=1,
                    groups=groups,
                )
                self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    def test_longcat_subsampling_equivalence(self):
        torch.manual_seed(0)
        baseline = nn.Sequential(
            nn.ConstantPad2d((0, 0, 2, 0), 0),
            nn.Conv2d(1, 32, 3, (2, 1)),
        )
        moe = make_conv2d_moe(1, 32, kernel_size=3, num_experts=4, stride=(2, 1), padding=0)
        bank = moe.expert_bank
        source_conv: Any = baseline[1]
        source_pad: Any = baseline[0]
        with torch.no_grad():
            bank.weight.copy_(source_conv.weight.unsqueeze(0).expand_as(bank.weight))
            assert source_conv.bias is not None and bank.bias is not None
            bank.bias.copy_(source_conv.bias.unsqueeze(0).expand_as(bank.bias))
        x = torch.randn(3, 1, 20, 80)
        expert_weights = torch.rand(3, 4)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        y = run_manual(moe, source_pad(x), expert_weights)
        self.assertTrue(torch.allclose(y, baseline(x), atol=1e-6))


class ConvRouterTest(unittest.TestCase):
    def test_router_returns_logits(self):
        router = ConvRouter1d(16, 4)
        logits = router(torch.randn(3, 16, 24))
        self.assertEqual(logits.shape, (3, 4))
        self.assertTrue(torch.isfinite(logits).all())

    def test_router_supports_multiscale(self):
        router = ConvRouter1d(16, 12, multi_scale=True, hidden_size=4)
        logits = router(torch.randn(3, 16, 24))
        self.assertEqual(logits.shape, (3, 12))

    def test_router2d_returns_logits(self):
        router = ConvRouter2d(16, 4)
        logits = router(torch.randn(3, 16, 13, 17))
        self.assertEqual(logits.shape, (3, 4))

    def test_moe_uses_adt_as_separate_gate(self):
        gate = AdaptiveDirichletTempering(num_experts=4)
        expert, weight, bias = expand_expert(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            num_experts=4,
            init="kaiming",
        )
        moe = MoE(
            num_experts=4,
            router=ConvRouter1d(16, 4),
            expert=expert,
            weight=weight,
            bias=bias,
            gate=gate,
            top_k=None,
        )
        y = moe(torch.randn(2, 16, 24)).hidden_states
        self.assertEqual(y.shape, (2, 32, 24))
        self.assertEqual(gate.num_updates, 1)


class ConvShapeTest(unittest.TestCase):
    def test_effective_kernel_and_padding(self):
        self.assertEqual(
            effective_kernel_size_1d(torch.Size([3]), torch.Size([2])), torch.Size([5])
        )
        self.assertEqual(infer_padding_1d(torch.Size([5]), torch.Size([1])), torch.Size([2]))
        self.assertEqual(
            effective_kernel_size_2d(torch.Size([3, 5]), torch.Size([2, 1])),
            torch.Size([5, 5]),
        )
        self.assertEqual(
            infer_padding_2d(torch.Size([5, 3]), torch.Size([1, 2])),
            torch.Size([2, 1]),
        )

    def test_segment_validation_only_applies_when_segment_size_is_set(self):
        validate_dynamic_conv1d_args(
            kernel_size=torch.Size([4]),
            stride=torch.Size([1]),
            padding=torch.Size([0]),
            dilation=torch.Size([1]),
            segment_size=None,
        )
        with self.assertRaisesRegex(ValueError, "odd effective kernel"):
            validate_dynamic_conv1d_args(
                kernel_size=torch.Size([4]),
                stride=torch.Size([1]),
                padding=torch.Size([0]),
                dilation=torch.Size([1]),
                segment_size=torch.Size([8]),
            )


if __name__ == "__main__":
    unittest.main()
