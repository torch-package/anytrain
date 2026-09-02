from __future__ import annotations

import unittest
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor, nn

import anytrain.module as module_api
from anytrain.module.dirichlet_tempering import ADT
from anytrain.module.moe import (
    CallableExpert,
    Conv1dExpert,
    ExpertRouting,
    Grid1d,
    LinearRouter,
    MoE,
    MoEOutput,
    ParamExpertBank,
    bank_weight_shape,
    dropped_token_ratio,
    effective_experts,
    expand_expert,
    expand_param,
    expert_probability_mean,
    expert_route_weight_mean,
    expert_routing_from_weights,
    expert_usage,
    linear,
    load_balance_loss,
    materialize_routing,
    moe_loss,
    moe_routing_metrics,
    router_z_loss,
    tokens_per_expert,
)
from anytrain.module.moe.conv import Conv1d


def _scale_expert(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    del bias
    return input * weight


scale_expert = CallableExpert(_scale_expert)


class LinearRouterTest(unittest.TestCase):
    def test_returns_logits(self):
        router = LinearRouter(2, 3, bias=False)
        with torch.no_grad():
            router.proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                    ]
                )
            )
        hidden = torch.tensor([[[3.0, 1.0], [0.5, 2.0]]])
        logits = router(hidden)
        self.assertEqual(logits.shape, (1, 2, 3))

    def test_shape_controls_require_integers(self):
        with self.assertRaisesRegex(TypeError, "input_dim must be an integer"):
            LinearRouter(cast(Any, 2.0), 3)
        with self.assertRaisesRegex(TypeError, "num_experts must be an integer"):
            LinearRouter(2, cast(Any, True))
        with self.assertRaisesRegex(TypeError, "segment_size must be an integer"):
            Grid1d(cast(Any, 2.0))


class MoETest(unittest.TestCase):
    def test_rejects_non_integer_declared_expert_count(self):
        class InvalidRouter(nn.Module):
            num_experts = True

            def forward(self, hidden: Tensor) -> Tensor:
                return hidden

        with self.assertRaisesRegex(TypeError, "declared num_experts must be an integer"):
            MoE(
                num_experts=2,
                router=InvalidRouter(),
                expert=scale_expert,
                weight=nn.Parameter(torch.ones(2)),
            )

    def test_topk_generic_bank_matches_reference(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        routing = ExpertRouting(
            logits=torch.zeros(1, 3, 3),
            probabilities=torch.full((1, 3, 3), 1 / 3),
            indices=torch.tensor([[[0, 1], [1, 2], [2, 0]]]),
            weights=torch.tensor([[[0.75, 0.25], [0.40, 0.60], [1.00, 0.00]]]),
        )
        moe = MoE(
            num_experts=3,
            router=LinearRouter(2, 3),
            expert=scale_expert,
            weight=nn.Parameter(torch.tensor([1.0, 10.0, -1.0])),
            top_k=2,
        )
        output = moe.forward_manually(hidden, routing)
        expected = torch.stack(
            [
                hidden[0, 0] * 0.75 + hidden[0, 0] * 10.0 * 0.25,
                hidden[0, 1] * 10.0 * 0.40 + hidden[0, 1] * -1.0 * 0.60,
                hidden[0, 2] * -1.0,
            ]
        ).unsqueeze(0)
        self.assertTrue(torch.allclose(output.hidden_states, expected))
        self.assertTrue(torch.equal(output.tokens_per_expert.cpu(), torch.tensor([1, 2, 2])))

    def test_topk_generic_bank_batches_routes_by_expert(self):
        batch_sizes: list[int] = []

        def recording_expert(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None = None,
        ) -> Tensor:
            del bias
            batch_sizes.append(input.size(0))
            return input * weight

        hidden = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)
        indices = torch.tensor([[0, 2], [0, 3], [2, 3], [2, 3]])
        selected_weights = torch.tensor(
            [[0.75, 0.25], [0.0, 1.0], [0.4, 0.6], [0.2, 0.8]],
            requires_grad=True,
        )
        routing = ExpertRouting(
            logits=torch.zeros(4, 4),
            probabilities=torch.full((4, 4), 0.25),
            indices=indices,
            weights=selected_weights,
        )
        expert_weights = nn.Parameter(torch.tensor([1.0, 10.0, -1.0, 2.0]))
        moe = MoE(
            num_experts=4,
            router=LinearRouter(2, 4),
            expert=CallableExpert(recording_expert),
            weight=expert_weights,
            top_k=2,
        )

        actual = moe.forward_manually(hidden, routing).hidden_states
        selected_expert_weights = expert_weights[indices]
        expected = (
            hidden.unsqueeze(1)
            * selected_expert_weights.unsqueeze(-1)
            * selected_weights.unsqueeze(-1)
        ).sum(dim=1)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(batch_sizes, [2, 3, 3])
        actual.sum().backward()
        self.assertIsNotNone(selected_weights.grad)
        assert selected_weights.grad is not None
        self.assertNotEqual(float(selected_weights.grad[1, 0]), 0.0)

    def test_topk_generic_all_masked_uses_empty_shape_probe(self):
        batch_sizes: list[int] = []

        def probed_expert(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None = None,
        ) -> Tensor:
            del bias
            batch_sizes.append(input.size(0))
            return input * weight

        hidden = torch.ones(3, 2, requires_grad=True)
        selected_weights = torch.ones(3, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(3, 2),
            probabilities=torch.full((3, 2), 0.5),
            indices=torch.tensor([[0], [1], [0]]),
            weights=selected_weights,
            mask=torch.zeros(3, dtype=torch.bool),
        )
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=CallableExpert(probed_expert),
            weight=nn.Parameter(torch.tensor([2.0, 3.0])),
            top_k=1,
        )

        output = moe.forward_manually(hidden, routing)

        self.assertTrue(torch.equal(output.hidden_states, torch.zeros_like(hidden)))
        self.assertEqual(batch_sizes, [0])
        output.hidden_states.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(selected_weights.grad)
        self.assertIsNotNone(moe.expert_bank.weight.grad)

    def test_sparse_dispatch_skips_unused_experts(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        routing = ExpertRouting(
            logits=torch.zeros(2, 4),
            probabilities=torch.full((2, 4), 0.25),
            indices=torch.tensor([[0], [2]], dtype=torch.long),
            weights=torch.ones(2, 1),
        )
        experts: Any = nn.ModuleList(
            [
                nn.Linear(2, 2, bias=False),
                nn.Linear(2, 2, bias=False),
                nn.Linear(2, 2, bias=False),
                nn.Linear(2, 2, bias=False),
            ]
        )
        with torch.no_grad():
            for expert in experts:
                expert.weight.copy_(torch.eye(2))

        output, counts = sparse_dispatch(hidden, routing, experts)

        self.assertTrue(torch.allclose(output, hidden))
        self.assertTrue(torch.equal(counts.cpu(), torch.tensor([1, 0, 1, 0])))

    def test_sparse_dispatch_all_masked_preserves_zero_gradient_graph(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.randn(3, 2, requires_grad=True)
        selected_weights = torch.ones(3, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(3, 1),
            probabilities=torch.ones(3, 1),
            indices=torch.zeros(3, 1, dtype=torch.long),
            weights=selected_weights,
            mask=torch.zeros(3, dtype=torch.bool),
        )
        expert = nn.Linear(2, 3)

        output, counts = sparse_dispatch(hidden, routing, [expert])
        output.sum().backward()

        self.assertEqual(output.shape, (3, 3))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(torch.equal(counts, torch.zeros_like(counts)))
        for grad in (
            hidden.grad,
            selected_weights.grad,
            expert.weight.grad,
            expert.bias.grad,
        ):
            self.assertIsNotNone(grad)
            assert grad is not None
            self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_sparse_dispatch_all_masked_does_not_read_nan_inputs(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.tensor([[float("nan"), float("inf")]], requires_grad=True)
        selected_weights = torch.full((1, 1), float("nan"), requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=selected_weights,
            mask=torch.zeros(1, dtype=torch.bool),
        )
        expert = nn.Linear(2, 1)

        output, _counts = sparse_dispatch(hidden, routing, [expert])
        output.sum().backward()

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        for grad in (hidden.grad, selected_weights.grad, expert.weight.grad, expert.bias.grad):
            self.assertIsNotNone(grad)
            assert grad is not None
            self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_sparse_dispatch_all_masked_does_not_update_batch_norm_mean(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.full((4, 2), 7.0)
        routing = ExpertRouting(
            logits=torch.zeros(4, 1),
            probabilities=torch.ones(4, 1),
            indices=torch.zeros(4, 1, dtype=torch.long),
            weights=torch.ones(4, 1),
            mask=torch.zeros(4, dtype=torch.bool),
        )
        expert = nn.BatchNorm1d(2)
        assert expert.running_mean is not None
        assert expert.num_batches_tracked is not None
        before = expert.running_mean.detach().clone()
        before_batches = expert.num_batches_tracked.detach().clone()

        output, _counts = sparse_dispatch(hidden, routing, [expert])

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        torch.testing.assert_close(expert.running_mean, before)
        torch.testing.assert_close(expert.num_batches_tracked, before_batches)

    def test_sparse_dispatch_zero_weights_keep_route_derivatives(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        selected_weights = torch.zeros(2, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(2, 1),
            probabilities=torch.ones(2, 1),
            indices=torch.zeros(2, 1, dtype=torch.long),
            weights=selected_weights,
        )
        expert = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            expert.weight.copy_(torch.tensor([[1.0, 2.0]]))

        output, counts = sparse_dispatch(hidden, routing, [expert])
        output.sum().backward()

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(torch.equal(counts, torch.zeros_like(counts)))
        torch.testing.assert_close(
            selected_weights.grad,
            torch.tensor([[5.0], [11.0]]),
        )
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(expert.weight.grad)
        assert hidden.grad is not None
        assert expert.weight.grad is not None
        self.assertTrue(torch.equal(hidden.grad, torch.zeros_like(hidden.grad)))
        self.assertTrue(torch.equal(expert.weight.grad, torch.zeros_like(expert.weight.grad)))

    def test_sparse_dispatch_promotes_bfloat16_expert_and_float32_routes(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        class Scale(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.5, dtype=torch.bfloat16))

            def forward(self, input: Tensor) -> Tensor:
                return input.to(dtype=self.scale.dtype) * self.scale

        hidden = torch.randn(3, 2, dtype=torch.bfloat16, requires_grad=True)
        selected_weights = torch.rand(3, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(3, 1),
            probabilities=torch.ones(3, 1),
            indices=torch.zeros(3, 1, dtype=torch.long),
            weights=selected_weights,
        )
        expert = Scale()

        actual, _counts = sparse_dispatch(hidden, routing, [expert])
        expected = expert(hidden) * selected_weights
        actual_grads = torch.autograd.grad(
            actual.square().sum(),
            (hidden, expert.scale, selected_weights),
            retain_graph=True,
        )
        expected_grads = torch.autograd.grad(
            expected.square().sum(),
            (hidden, expert.scale, selected_weights),
        )

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=False):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_sparse_dispatch_promotes_accumulator_for_later_expert(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        class Scale(nn.Module):
            def __init__(self, dtype: torch.dtype, value: float) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(value, dtype=dtype))

            def forward(self, input: Tensor) -> Tensor:
                return input.to(dtype=self.scale.dtype) * self.scale

        hidden = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
        selected_weights = torch.rand(2, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.full((2, 2), 0.5),
            indices=torch.tensor([[0], [1]]),
            weights=selected_weights,
        )
        experts: Any = nn.ModuleList(
            [
                Scale(torch.bfloat16, 1.5),
                Scale(torch.float64, -0.5),
            ]
        )

        actual, _counts = sparse_dispatch(hidden, routing, experts)
        expected = torch.cat(
            [
                experts[0](hidden[:1]) * selected_weights[:1],
                experts[1](hidden[1:]) * selected_weights[1:],
            ],
            dim=0,
        )
        parameters = (hidden, experts[0].scale, experts[1].scale, selected_weights)
        actual_grads = torch.autograd.grad(
            actual.square().sum(),
            parameters,
            retain_graph=True,
        )
        expected_grads = torch.autograd.grad(expected.square().sum(), parameters)

        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=False):
            torch.testing.assert_close(actual_grad, expected_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sparse_dispatch_unused_experts_match_on_cuda(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        device = torch.device("cuda")
        hidden = torch.randn(32, 8, device=device)
        indices = torch.zeros(32, 1, dtype=torch.long, device=device)
        # Only experts 0 and 2 receive tokens; 1 and 3 stay empty.
        indices[16:] = 2
        routing = ExpertRouting(
            logits=torch.zeros(32, 4, device=device),
            probabilities=torch.full((32, 4), 0.25, device=device),
            indices=indices,
            weights=torch.ones(32, 1, device=device),
        )
        experts: Any = nn.ModuleList([nn.Linear(8, 8, bias=False).to(device) for _ in range(4)])
        with torch.no_grad():
            for expert in experts:
                expert.weight.copy_(torch.eye(8, device=device))

        output, counts = sparse_dispatch(hidden, routing, experts)
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(output, hidden))
        self.assertTrue(torch.equal(counts.cpu(), torch.tensor([16, 0, 16, 0])))

    def test_forward_ignores_masked_tokens(self):
        hidden = torch.ones(1, 3, 2)
        routing = ExpertRouting(
            logits=torch.zeros(1, 3, 2),
            probabilities=torch.full((1, 3, 2), 0.5),
            indices=torch.zeros(1, 3, 1, dtype=torch.long),
            weights=torch.ones(1, 3, 1),
            mask=torch.tensor([[True, False, True]]),
        )
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=scale_expert,
            weight=nn.Parameter(torch.tensor([2.0, 3.0])),
            top_k=1,
        )
        output = moe.forward_manually(hidden, routing)
        self.assertTrue(torch.equal(output.hidden_states[0, 0], torch.full((2,), 2.0)))
        self.assertTrue(torch.equal(output.hidden_states[0, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(output.hidden_states[0, 2], torch.full((2,), 2.0)))

    def test_masked_linear_bias_is_zero_for_dense_and_topk_routes(self):
        hidden = torch.ones(2, 2)
        for top_k in (None, 1):
            with self.subTest(top_k=top_k):
                if top_k is None:
                    routing = ExpertRouting(
                        logits=torch.zeros(2, 2),
                        probabilities=torch.tensor([[0.25, 0.75], [0.25, 0.75]]),
                        mask=torch.tensor([True, False]),
                    )
                else:
                    routing = ExpertRouting(
                        logits=torch.zeros(2, 2),
                        probabilities=torch.full((2, 2), 0.5),
                        indices=torch.tensor([[1], [0]]),
                        weights=torch.ones(2, 1),
                        mask=torch.tensor([True, False]),
                    )
                moe = MoE(
                    num_experts=2,
                    router=LinearRouter(2, 2),
                    expert=linear,
                    weight=nn.Parameter(torch.ones(2, 1, 2)),
                    bias=nn.Parameter(torch.tensor([[3.0], [5.0]])),
                    top_k=top_k,
                )

                output = moe.forward_manually(hidden, routing).hidden_states

                expected_active = torch.tensor([7.0]) if top_k == 1 else torch.tensor([6.5])
                torch.testing.assert_close(output[0], expected_active)
                self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))

    def test_masked_conv1d_bias_is_zero_for_manual_routing(self):
        template = nn.Conv1d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            template.weight.fill_(2.0)
            assert template.bias is not None
            template.bias.fill_(3.0)
        expert, weight, bias = expand_expert(template, num_experts=1, init="copy")
        assert bias is not None
        moe = MoE(
            num_experts=1,
            router=LinearRouter(1, 1),
            expert=expert,
            weight=weight,
            bias=bias,
            top_k=None,
        )
        routing = ExpertRouting(
            logits=torch.zeros(2, 1),
            probabilities=torch.ones(2, 1),
            mask=torch.tensor([True, False]),
        )

        output = moe.forward_manually(torch.ones(2, 1, 3), routing).hidden_states

        self.assertTrue(torch.equal(output[0], torch.full((1, 3), 5.0)))
        self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))

    def test_public_forward_applies_conv1d_and_conv2d_masks(self):
        conv1d = nn.Conv1d(1, 1, kernel_size=1, bias=True)
        conv2d = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            conv1d.weight.fill_(2.0)
            assert conv1d.bias is not None
            conv1d.bias.fill_(3.0)
            conv2d.weight.fill_(2.0)
            assert conv2d.bias is not None
            conv2d.bias.fill_(3.0)

        moe_1d = MoE.from_module(conv1d, num_experts=1)
        moe_2d = MoE.from_module(conv2d, num_experts=1)
        mask = torch.tensor([True, False])

        output_1d = moe_1d(torch.ones(2, 1, 3), mask=mask).hidden_states
        output_2d = moe_2d(torch.ones(2, 1, 2, 2), mask=mask).hidden_states

        self.assertTrue(torch.equal(output_1d[0], torch.full((1, 3), 5.0)))
        self.assertTrue(torch.equal(output_1d[1], torch.zeros_like(output_1d[1])))
        self.assertTrue(torch.equal(output_2d[0], torch.full((1, 2, 2), 5.0)))
        self.assertTrue(torch.equal(output_2d[1], torch.zeros_like(output_2d[1])))

    def test_masked_conv_transpose_overlap_does_not_dilute_active_bias(self):
        template = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, bias=True)
        with torch.no_grad():
            template.weight.zero_()
            assert template.bias is not None
            template.bias.fill_(2.0)
        moe = MoE.from_module(
            template,
            num_experts=1,
            router=LinearRouter(1, 1),
            grid=Grid1d(segment_size=2),
            top_k=None,
        )
        routing = ExpertRouting(
            logits=torch.zeros(2, 1),
            probabilities=torch.ones(2, 1),
            mask=torch.tensor([True, False]),
        )

        output = moe.forward_manually(torch.ones(1, 1, 4), routing).hidden_states

        torch.testing.assert_close(
            output,
            torch.tensor([[[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]]]),
        )

    def test_generic_bank_masks_rank_changing_outputs(self):
        def rank_changing_expert(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None = None,
        ) -> Tensor:
            del bias
            scaled = input * weight
            return torch.stack((scaled, scaled + weight), dim=1)

        hidden = torch.arange(1, 13, dtype=torch.float32).reshape(2, 3, 2)
        mask = torch.tensor([[True], [False]])
        expert_weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        for top_k in (None, 1):
            with self.subTest(top_k=top_k):
                probabilities = torch.tensor(
                    [
                        [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]],
                        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                    ]
                )
                if top_k is None:
                    routing = ExpertRouting(
                        logits=probabilities.log(),
                        probabilities=probabilities,
                        mask=mask,
                    )
                else:
                    routing = ExpertRouting(
                        logits=probabilities.log(),
                        probabilities=probabilities,
                        indices=torch.zeros(2, 3, 1, dtype=torch.long),
                        weights=torch.ones(2, 3, 1),
                        mask=mask,
                    )
                moe = MoE(
                    num_experts=2,
                    router=LinearRouter(2, 2),
                    expert=CallableExpert(rank_changing_expert),
                    weight=expert_weight,
                    top_k=top_k,
                )

                output = moe.forward_manually(hidden, routing).hidden_states

                self.assertEqual(output.shape, (2, 3, 2, 2))
                self.assertTrue(torch.count_nonzero(output[0]) > 0)
                self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))

    def test_unmasked_bank_skips_output_and_bias_masked_fill(self):
        hidden = torch.ones(2, 2)
        routing = ExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.full((2, 2), 0.5),
        )
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=linear,
            weight=nn.Parameter(torch.ones(2, 1, 2)),
            bias=nn.Parameter(torch.ones(2, 1)),
            top_k=None,
        )

        with patch.object(
            torch.Tensor,
            "masked_fill",
            side_effect=AssertionError("unmasked bank called Tensor.masked_fill()"),
        ):
            output = moe.forward_manually(hidden, routing).hidden_states

        torch.testing.assert_close(output, torch.full((2, 1), 3.0))

    def test_public_forward_mask_preserves_transpose_overlap_coverage(self):
        template = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, bias=True)
        with torch.no_grad():
            template.weight.zero_()
            assert template.bias is not None
            template.bias.fill_(2.0)
        moe = MoE.from_module(
            template,
            num_experts=1,
            grid=Grid1d(segment_size=2),
        )

        output = moe(
            torch.ones(1, 1, 4),
            mask=torch.tensor([[True, False]]),
        ).hidden_states

        torch.testing.assert_close(
            output,
            torch.tensor([[[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]]]),
        )

    def test_segmented_manual_forward_accepts_flat_routing_mask(self):
        template = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, bias=True)
        with torch.no_grad():
            template.weight.zero_()
            assert template.bias is not None
            template.bias.fill_(2.0)
        moe = MoE.from_module(
            template,
            num_experts=1,
            router=LinearRouter(1, 1),
            grid=Grid1d(segment_size=2),
        )
        routing = ExpertRouting(
            logits=torch.zeros(4, 1),
            probabilities=torch.ones(4, 1),
            mask=torch.tensor([True, False, False, True]),
        )

        output = moe.forward_manually(torch.ones(2, 1, 4), routing).hidden_states

        torch.testing.assert_close(
            output,
            torch.tensor(
                [
                    [[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                ]
            ),
        )

    def test_dense_and_topk_linear_bank_runtime_switch(self):
        torch.manual_seed(0)
        router = LinearRouter(4, 3, bias=False)
        weight = nn.Parameter(torch.randn(3, 5, 4))
        moe = MoE(
            num_experts=3,
            router=router,
            expert=linear,
            weight=weight,
            bias=None,
            top_k=None,
        )
        hidden = torch.randn(2, 3, 4)

        dense = moe(hidden)
        sparse = moe(hidden, top_k=2)

        self.assertEqual(dense.hidden_states.shape, sparse.hidden_states.shape)
        self.assertTrue(dense.routing.is_dense)
        self.assertEqual(sparse.routing.top_k, 2)
        self.assertIsNone(dense.dropped_mask)
        self.assertIsNotNone(sparse.dropped_mask)
        self.assertIsInstance(moe.expert_bank, ParamExpertBank)

    def test_linear_bank_matches_reference_outputs_and_gradients(self):
        torch.manual_seed(0)
        for top_k in (None, 2):
            with self.subTest(top_k=top_k):
                hidden = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
                expert_weight = nn.Parameter(torch.randn(3, 5, 4, dtype=torch.double))
                logits = torch.randn(2, 3, 3, dtype=torch.double)
                probabilities = logits.softmax(dim=-1).detach().requires_grad_(True)
                if top_k is None:
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                    )
                    route_values = probabilities
                    mixed_weight = torch.einsum(
                        "...e,eoi->...oi",
                        probabilities,
                        expert_weight,
                    )
                    expected = torch.einsum("...i,...oi->...o", hidden, mixed_weight)
                else:
                    indices = logits.topk(top_k, dim=-1).indices
                    selected_weights = torch.randn(
                        2,
                        3,
                        top_k,
                        dtype=torch.double,
                        requires_grad=True,
                    )
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                        indices=indices,
                        weights=selected_weights,
                    )
                    route_values = selected_weights
                    selected_weight = expert_weight[indices]
                    expert_output = torch.einsum(
                        "...i,...koi->...ko",
                        hidden,
                        selected_weight,
                    )
                    expected = torch.einsum(
                        "...k,...ko->...o",
                        selected_weights,
                        expert_output,
                    )

                moe = MoE(
                    num_experts=3,
                    router=LinearRouter(4, 3).double(),
                    expert=linear,
                    weight=expert_weight,
                    top_k=top_k,
                )
                if top_k is None:
                    with patch(
                        "anytrain.module.moe.bank.mix_expert_weights",
                        side_effect=AssertionError("linear path mixed full weights"),
                    ):
                        actual = moe.forward_manually(hidden, routing).hidden_states
                else:
                    actual = moe.forward_manually(hidden, routing).hidden_states

                actual_grads = torch.autograd.grad(
                    actual.square().sum(),
                    (hidden, expert_weight, route_values),
                    retain_graph=True,
                )
                expected_grads = torch.autograd.grad(
                    expected.square().sum(),
                    (hidden, expert_weight, route_values),
                )
                torch.testing.assert_close(actual, expected)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=False):
                    torch.testing.assert_close(actual_grad, expected_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_topk_linear_bank_batches_routes_without_host_count_conversion(self):
        torch.manual_seed(11)
        hidden = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
        expert_weight = nn.Parameter(torch.randn(4, 5, 4, dtype=torch.double))
        indices = torch.tensor(
            [
                [[0, 1], [1, 0], [2, 3]],
                [[3, 2], [0, 3], [1, 2]],
            ]
        )
        selected_weights = torch.tensor(
            [
                [[0.8, 0.2], [0.0, 1.0], [0.4, 0.6]],
                [[0.3, 0.7], [1.0, 0.0], [0.2, 0.8]],
            ],
            dtype=torch.double,
            requires_grad=True,
        )
        routing = ExpertRouting(
            logits=torch.zeros(2, 3, 4, dtype=torch.double),
            probabilities=torch.full((2, 3, 4), 0.25, dtype=torch.double),
            indices=indices,
            weights=selected_weights,
        )
        moe = MoE(
            num_experts=4,
            router=LinearRouter(4, 4).double(),
            expert=linear,
            weight=expert_weight,
            top_k=2,
        )

        with patch.object(
            torch.Tensor,
            "tolist",
            side_effect=AssertionError("linear top-k path converted route counts to host"),
        ):
            actual = moe.forward_manually(hidden, routing).hidden_states

        selected_weight = expert_weight[indices]
        expert_output = torch.einsum(
            "...i,...koi->...ko",
            hidden,
            selected_weight,
        )
        expected = torch.einsum(
            "...k,...ko->...o",
            selected_weights,
            expert_output,
        )
        actual_grads = torch.autograd.grad(
            actual.square().sum(),
            (hidden, expert_weight, selected_weights),
            retain_graph=True,
        )
        expected_grads = torch.autograd.grad(
            expected.square().sum(),
            (hidden, expert_weight, selected_weights),
        )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=False):
            torch.testing.assert_close(actual_grad, expected_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sparse_linear_fast_path_batches_routes_without_host_count_conversion(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        torch.manual_seed(12)
        hidden = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        experts: Any = nn.ModuleList(
            nn.Linear(3, 2, dtype=torch.double) for _ in range(3)
        )
        indices = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]])
        selected_weights = torch.tensor(
            [[0.75, 0.25], [1.0, 0.0], [0.4, 0.6], [0.2, 0.8]],
            dtype=torch.double,
            requires_grad=True,
        )
        routing = ExpertRouting(
            logits=torch.zeros(4, 3, dtype=torch.double),
            probabilities=torch.full((4, 3), 1 / 3, dtype=torch.double),
            indices=indices,
            weights=selected_weights,
        )

        with patch.object(
            torch.Tensor,
            "tolist",
            side_effect=AssertionError("linear sparse path converted route counts to host"),
        ):
            actual, counts = sparse_dispatch(hidden, routing, experts)

        route_tokens = torch.arange(hidden.size(0)).unsqueeze(-1).expand_as(indices).reshape(-1)
        route_experts = indices.reshape(-1)
        route_outputs = torch.stack(
            [
                torch.nn.functional.linear(
                    hidden[token],
                    experts[int(expert_index)].weight,
                    experts[int(expert_index)].bias,
                )
                for token, expert_index in zip(route_tokens, route_experts, strict=False)
            ]
        )
        expected = torch.zeros_like(actual).reshape(-1, actual.size(-1)).index_add(
            0,
            route_tokens,
            route_outputs * selected_weights.reshape(-1, 1),
        )
        expected = expected.reshape_as(actual)
        expected_counts = torch.tensor([2, 2, 3])

        actual_grads = torch.autograd.grad(
            actual.square().sum(),
            (hidden, selected_weights, *experts.parameters()),
            retain_graph=True,
        )
        expected_grads = torch.autograd.grad(
            expected.square().sum(),
            (hidden, selected_weights, *experts.parameters()),
        )
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(counts.cpu(), expected_counts))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=False):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_topk_linear_bank_uses_grouped_gemm_on_cpu(self):
        hidden = torch.randn(4, 3)
        routing = ExpertRouting(
            logits=torch.zeros(4, 2),
            probabilities=torch.full((4, 2), 0.5),
            indices=torch.tensor([[0], [1], [0], [1]]),
            weights=torch.ones(4, 1),
        )
        moe = MoE(
            num_experts=2,
            router=LinearRouter(3, 2),
            expert=linear,
            weight=nn.Parameter(torch.randn(2, 4, 3)),
            top_k=1,
        )

        with patch.object(
            moe.expert_bank,
            "_forward_linear_topk_batched",
            side_effect=AssertionError("CPU should use grouped GEMM"),
        ):
            output = moe.forward_manually(hidden, routing).hidden_states

        self.assertEqual(output.shape, (4, 4))

    def test_sparse_linear_dispatch_preserves_global_module_hooks(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.randn(3, 2)
        routing = ExpertRouting(
            logits=torch.zeros(3, 2),
            probabilities=torch.full((3, 2), 0.5),
            indices=torch.tensor([[0], [1], [0]]),
            weights=torch.ones(3, 1),
        )
        experts: Any = nn.ModuleList(nn.Linear(2, 2) for _ in range(2))
        calls: list[nn.Module] = []

        def hook(module: nn.Module, _inputs: Any, _output: Any) -> None:
            calls.append(module)

        handle = nn.modules.module.register_module_forward_hook(hook)
        try:
            output, _counts = sparse_dispatch(hidden, routing, experts)
        finally:
            handle.remove()

        self.assertEqual(len(calls), len(experts))
        self.assertEqual(output.shape, (3, 2))

    def test_sparse_linear_fast_path_guard_preserves_autograd_and_observers(self):
        from anytrain.module.moe.sparse import (
            _can_batch_linear_experts,
            _linear_fast_path_preserves_unused_gradients,
        )

        experts: Any = nn.ModuleList(nn.Linear(2, 2) for _ in range(2))

        # The structural guard accepts plain Linear modules; the route-aware
        # guard below selects grouped dispatch when an expert is absent.
        self.assertTrue(_can_batch_linear_experts(experts))
        route_ids = torch.tensor([0, 0, 0], dtype=torch.long)
        self.assertFalse(
            _linear_fast_path_preserves_unused_gradients(
                route_ids,
                active_routes=None,
                num_experts=2,
                experts=experts,
            )
        )
        self.assertTrue(
            _linear_fast_path_preserves_unused_gradients(
                torch.tensor([0, 1], dtype=torch.long),
                active_routes=None,
                num_experts=2,
                experts=experts,
            )
        )
        with torch.no_grad():
            self.assertTrue(_can_batch_linear_experts(experts))

        with torch.no_grad():
            handle = experts[0].register_forward_hook(
                lambda _module, _inputs, _output: None
            )
            try:
                self.assertFalse(_can_batch_linear_experts(experts))
            finally:
                handle.remove()

        with torch.no_grad():
            handle = experts[0].weight.register_hook(lambda grad: grad)
            try:
                self.assertFalse(_can_batch_linear_experts(experts))
            finally:
                handle.remove()

        with (
            torch.no_grad(),
            patch.object(
                experts[0],
                "_compiled_call_impl",
                lambda *_args, **_kwargs: None,
            ),
        ):
            self.assertFalse(_can_batch_linear_experts(experts))

    def test_sparse_dispatch_unused_expert_gradients_stay_none(self):
        from anytrain.module.moe.sparse import sparse_dispatch

        hidden = torch.randn(3, 2, requires_grad=True)
        selected_weights = torch.ones(3, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(3, 2),
            probabilities=torch.full((3, 2), 0.5),
            indices=torch.zeros(3, 1, dtype=torch.long),
            weights=selected_weights,
        )
        experts: Any = nn.ModuleList(nn.Linear(2, 2) for _ in range(2))

        output, _counts = sparse_dispatch(hidden, routing, experts)
        output.sum().backward()

        self.assertIsNotNone(experts[0].weight.grad)
        self.assertIsNotNone(experts[0].bias.grad)
        self.assertIsNone(experts[1].weight.grad)
        self.assertIsNone(experts[1].bias.grad)

    @unittest.skipUnless(
        torch.amp.autocast_mode.is_autocast_available("cpu"),
        "requires CPU autocast",
    )
    def test_linear_bank_cpu_autocast_promotes_float32_routes_and_backward(self):
        torch.manual_seed(0)
        for top_k in (None, 2):
            with self.subTest(top_k=top_k):
                hidden = torch.randn(2, 3, 4, requires_grad=True)
                weight = nn.Parameter(torch.randn(3, 5, 4))
                bias = nn.Parameter(torch.randn(3, 5))
                logits = torch.randn(2, 3, 3)
                probabilities = logits.softmax(dim=-1).detach().requires_grad_(True)
                if top_k is None:
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                    )
                    route_values = probabilities
                else:
                    indices = logits.topk(top_k, dim=-1).indices
                    selected_weights = torch.randn(2, 3, top_k).softmax(dim=-1)
                    selected_weights = selected_weights.detach().requires_grad_(True)
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                        indices=indices,
                        weights=selected_weights,
                    )
                    route_values = selected_weights
                moe = MoE(
                    num_experts=3,
                    router=LinearRouter(4, 3),
                    expert=linear,
                    weight=weight,
                    bias=bias,
                    top_k=top_k,
                )

                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = moe.forward_manually(hidden, routing).hidden_states
                output.float().square().mean().backward()

                self.assertEqual(output.dtype, torch.float32)
                for grad in (
                    hidden.grad,
                    weight.grad,
                    bias.grad,
                    route_values.grad,
                ):
                    self.assertIsNotNone(grad)
                    assert grad is not None
                    self.assertTrue(torch.isfinite(grad).all())

    def test_linear_bank_promotes_bfloat16_parameters_with_float32_routes(self):
        torch.manual_seed(0)
        for top_k in (None, 2):
            with self.subTest(top_k=top_k):
                hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
                weight = nn.Parameter(torch.randn(3, 5, 4, dtype=torch.bfloat16))
                bias = nn.Parameter(torch.randn(3, 5, dtype=torch.bfloat16))
                logits = torch.randn(2, 3, 3)
                probabilities = logits.softmax(dim=-1).detach().requires_grad_(True)
                if top_k is None:
                    routing = ExpertRouting(logits=logits, probabilities=probabilities)
                    route_values = probabilities
                else:
                    indices = logits.topk(top_k, dim=-1).indices
                    selected_weights = torch.rand(2, 3, top_k, requires_grad=True)
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                        indices=indices,
                        weights=selected_weights,
                    )
                    route_values = selected_weights
                moe = MoE(
                    num_experts=3,
                    router=LinearRouter(4, 3),
                    expert=linear,
                    weight=weight,
                    bias=bias,
                    top_k=top_k,
                )

                output = moe.forward_manually(hidden, routing).hidden_states
                output.square().mean().backward()

                self.assertEqual(output.dtype, torch.float32)
                for grad in (hidden.grad, weight.grad, bias.grad, route_values.grad):
                    self.assertIsNotNone(grad)
                    assert grad is not None
                    self.assertTrue(torch.isfinite(grad).all())

    def test_linear_bank_all_masked_does_not_read_nonfinite_values(self):
        for top_k in (None, 1):
            with self.subTest(top_k=top_k):
                hidden = torch.tensor(
                    [[float("nan"), float("inf")]],
                    requires_grad=True,
                )
                weight = nn.Parameter(torch.ones(2, 1, 2))
                bias = nn.Parameter(torch.ones(2, 1))
                probabilities = torch.full((1, 2), float("nan"), requires_grad=True)
                logits = torch.zeros(1, 2)
                if top_k is None:
                    routing = ExpertRouting(logits=logits, probabilities=probabilities)
                    route_values = probabilities
                else:
                    selected_weights = torch.full(
                        (1, 1),
                        float("nan"),
                        requires_grad=True,
                    )
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                        indices=torch.zeros(1, 1, dtype=torch.long),
                        weights=selected_weights,
                    )
                    route_values = selected_weights
                routing.mask = torch.zeros(1, dtype=torch.bool)
                moe = MoE(
                    num_experts=2,
                    router=LinearRouter(2, 2),
                    expert=linear,
                    weight=weight,
                    bias=bias,
                    top_k=top_k,
                )

                output = moe.forward_manually(hidden, routing).hidden_states
                output.sum().backward()

                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
                for grad in (hidden.grad, weight.grad, bias.grad, route_values.grad):
                    self.assertIsNotNone(grad)
                    assert grad is not None
                    self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_linear_bank_masked_nonfinite_token_has_zero_gradient(self):
        for top_k in (None, 1):
            with self.subTest(top_k=top_k):
                hidden = torch.tensor(
                    [[1.0, 2.0], [float("nan"), float("inf")]],
                    requires_grad=True,
                )
                weight = nn.Parameter(torch.ones(2, 1, 2))
                bias = nn.Parameter(torch.ones(2, 1))
                probabilities = torch.tensor(
                    [[1.0, 0.0], [float("nan"), float("nan")]],
                    requires_grad=True,
                )
                logits = torch.zeros(2, 2)
                kwargs = {}
                route_values = probabilities
                if top_k is not None:
                    selected_weights = torch.tensor(
                        [[1.0], [float("nan")]],
                        requires_grad=True,
                    )
                    kwargs = {
                        "indices": torch.zeros(2, 1, dtype=torch.long),
                        "weights": selected_weights,
                    }
                    route_values = selected_weights
                routing = ExpertRouting(
                    logits=logits,
                    probabilities=probabilities,
                    mask=torch.tensor([True, False]),
                    **cast(Any, kwargs),
                )
                moe = MoE(
                    num_experts=2,
                    router=LinearRouter(2, 2),
                    expert=linear,
                    weight=weight,
                    bias=bias,
                    top_k=top_k,
                )

                output = moe.forward_manually(hidden, routing).hidden_states
                output.sum().backward()

                torch.testing.assert_close(output[0], torch.tensor([4.0]))
                self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
                for grad in (hidden.grad, weight.grad, bias.grad, route_values.grad):
                    self.assertIsNotNone(grad)
                    assert grad is not None
                    self.assertTrue(torch.isfinite(grad).all())
                assert hidden.grad is not None
                assert route_values.grad is not None
                self.assertTrue(torch.equal(hidden.grad[1], torch.zeros_like(hidden.grad[1])))
                self.assertTrue(
                    torch.equal(route_values.grad[1], torch.zeros_like(route_values.grad[1]))
                )

    def test_normal_forward_avoids_tensor_scalar_checks(self):
        moe = MoE(
            num_experts=3,
            router=LinearRouter(4, 3),
            expert=linear,
            weight=nn.Parameter(torch.randn(3, 5, 4)),
            top_k=2,
        )
        hidden = torch.randn(2, 3, 4)

        with (
            patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("normal forward called Tensor.item()"),
            ),
            patch.object(
                torch.Tensor,
                "__bool__",
                side_effect=AssertionError("normal forward converted Tensor to bool"),
            ),
        ):
            output = moe(hidden)

        self.assertEqual(output.hidden_states.shape, (2, 3, 5))

    def test_forward_manually_keeps_strict_expert_id_validation(self):
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=linear,
            weight=nn.Parameter(torch.randn(2, 3, 2)),
            top_k=1,
        )
        routing = ExpertRouting(
            logits=torch.zeros(1, 2),
            probabilities=torch.full((1, 2), 0.5),
            indices=torch.tensor([[2]]),
            weights=torch.ones(1, 1),
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            moe.forward_manually(torch.randn(1, 2), routing)

    def test_generic_bank_dense_mix(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        probs = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
        routing = expert_routing_from_weights(probs)
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=scale_expert,
            weight=nn.Parameter(torch.tensor([1.0, 10.0])),
            top_k=None,
        )
        output = moe.forward_manually(hidden, routing)
        expected = torch.stack(
            [
                hidden[0] * 0.25 + hidden[0] * 10.0 * 0.75,
                hidden[1] * 0.5 + hidden[1] * 10.0 * 0.5,
            ]
        )
        self.assertTrue(torch.allclose(output.hidden_states, expected))
        self.assertIsInstance(moe.expert_bank, ParamExpertBank)

    def test_generic_bank_dense_rejects_wrong_expert_batch_size(self):
        def wrong_batch_expert(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None = None,
        ) -> Tensor:
            del bias
            return input[:1] * weight

        routing = expert_routing_from_weights(torch.full((2, 2), 0.5))
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=CallableExpert(wrong_batch_expert),
            weight=nn.Parameter(torch.ones(2)),
            top_k=None,
        )

        with self.assertRaisesRegex(ValueError, "batch size"):
            moe.forward_manually(torch.ones(2, 2), routing)

    def test_generic_bank_dense_executes_only_active_tokens(self):
        batch_sizes: list[int] = []

        def recording_expert(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None = None,
        ) -> Tensor:
            del bias
            batch_sizes.append(input.size(0))
            return input * weight

        hidden = torch.tensor([[1.0, 2.0], [float("nan"), float("inf")], [3.0, 4.0]])
        probabilities = torch.full((3, 2), 0.5)
        routing = ExpertRouting(
            logits=torch.zeros(3, 2),
            probabilities=probabilities,
            mask=torch.tensor([True, False, True]),
        )
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=CallableExpert(recording_expert),
            weight=nn.Parameter(torch.tensor([1.0, 10.0])),
            top_k=None,
        )

        output = moe.forward_manually(hidden, routing).hidden_states

        self.assertEqual(batch_sizes, [2, 2])
        self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
        self.assertTrue(torch.isfinite(output).all())

    def test_generic_bank_promotes_bfloat16_outputs_with_float32_routes(self):
        def bfloat16_expert(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None = None,
        ) -> Tensor:
            del bias
            return (input.float() * weight).to(dtype=torch.bfloat16)

        torch.manual_seed(0)
        for top_k in (None, 2):
            with self.subTest(top_k=top_k):
                hidden = torch.randn(3, 2, dtype=torch.bfloat16, requires_grad=True)
                expert_weight = nn.Parameter(torch.randn(3))
                probabilities = torch.rand(3, 3, requires_grad=True)
                logits = torch.randn(3, 3)
                if top_k is None:
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                    )
                    route_values = probabilities
                    expert_outputs = torch.stack(
                        [bfloat16_expert(hidden, expert_weight[index]) for index in range(3)],
                        dim=1,
                    )
                    expected = (probabilities.unsqueeze(-1) * expert_outputs).sum(dim=1)
                else:
                    indices = logits.topk(top_k, dim=-1).indices
                    selected_weights = torch.rand(3, top_k, requires_grad=True)
                    routing = ExpertRouting(
                        logits=logits,
                        probabilities=probabilities,
                        indices=indices,
                        weights=selected_weights,
                    )
                    route_values = selected_weights
                    expert_outputs = (
                        hidden.float().unsqueeze(1) * expert_weight[indices].unsqueeze(-1)
                    ).to(dtype=torch.bfloat16)
                    expected = (selected_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
                moe = MoE(
                    num_experts=3,
                    router=LinearRouter(2, 3),
                    expert=CallableExpert(bfloat16_expert),
                    weight=expert_weight,
                    top_k=top_k,
                )

                actual = moe.forward_manually(hidden, routing).hidden_states
                actual_grads = torch.autograd.grad(
                    actual.square().sum(),
                    (hidden, expert_weight, route_values),
                    retain_graph=True,
                )
                expected_grads = torch.autograd.grad(
                    expected.square().sum(),
                    (hidden, expert_weight, route_values),
                )

                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=False):
                    torch.testing.assert_close(actual_grad, expected_grad)

        hidden = torch.ones(2, 2, dtype=torch.bfloat16, requires_grad=True)
        expert_weight = nn.Parameter(torch.ones(2))
        selected_weights = torch.ones(2, 1, requires_grad=True)
        routing = ExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.full((2, 2), 0.5),
            indices=torch.tensor([[0], [1]]),
            weights=selected_weights,
            mask=torch.zeros(2, dtype=torch.bool),
        )
        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=CallableExpert(bfloat16_expert),
            weight=expert_weight,
            top_k=1,
        )

        masked = moe.forward_manually(hidden, routing).hidden_states
        masked.sum().backward()

        self.assertEqual(masked.dtype, torch.float32)
        self.assertTrue(torch.equal(masked, torch.zeros_like(masked)))
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(expert_weight.grad)
        self.assertIsNotNone(selected_weights.grad)

    def test_gate_receives_collect_stats(self):
        gate = ADT(num_experts=3)
        moe = MoE(
            num_experts=3,
            router=LinearRouter(2, 3),
            expert=scale_expert,
            weight=nn.Parameter(torch.ones(3)),
            top_k=1,
            gate=gate,
        )
        hidden = torch.randn(4, 2)
        moe(hidden, collect_stats=False)
        self.assertEqual(gate.num_updates, 0)
        moe(hidden, collect_stats=True)
        self.assertEqual(gate.num_updates, 1)

    def test_gradients_flow(self):
        torch.manual_seed(0)
        router = LinearRouter(4, 3)
        weight = nn.Parameter(torch.randn(3, 5, 4))
        bias = nn.Parameter(torch.randn(3, 5))
        moe = MoE(
            num_experts=3,
            router=router,
            expert=linear,
            weight=weight,
            bias=bias,
            top_k=3,
        )
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        output = moe(hidden)
        loss = (
            output.hidden_states.pow(2).mean()
            + load_balance_loss(output.routing)
            + 0.001 * router_z_loss(output.routing)
        )
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(router.proj.weight.grad)
        self.assertIsNotNone(weight.grad)

    def test_exports_stay_off_module_root(self):
        self.assertIsInstance(
            MoEOutput(
                hidden_states=torch.zeros(1, 2),
                routing=ExpertRouting(
                    logits=torch.zeros(1, 1),
                    probabilities=torch.ones(1, 1),
                    indices=torch.zeros(1, 1, dtype=torch.long),
                    weights=torch.ones(1, 1),
                ),
                tokens_per_expert=torch.ones(1, dtype=torch.long),
            ),
            MoEOutput,
        )
        self.assertFalse(hasattr(module_api, "MoE"))
        self.assertFalse(hasattr(module_api, "LinearRouter"))

    def test_rejects_bare_callable_expert(self):
        with self.assertRaisesRegex(TypeError, "Expert object"):
            MoE(
                num_experts=2,
                router=LinearRouter(2, 2),
                expert=cast(Any, _scale_expert),
                weight=nn.Parameter(torch.ones(2)),
                top_k=1,
            )

    def test_from_module_copies_linear_and_builds_default_router(self):
        torch.manual_seed(0)
        template = nn.Linear(4, 5)
        moe = MoE.from_module(template, num_experts=3, top_k=None)
        hidden = torch.randn(2, 7, 4)

        output = moe(hidden).hidden_states

        self.assertIsInstance(moe.router, LinearRouter)
        self.assertTrue(torch.allclose(output, template(hidden), atol=1e-6))

    def test_from_module_rejects_non_stock_module(self):
        with self.assertRaisesRegex(TypeError, "expand_expert supports"):
            MoE.from_module(nn.Sequential(nn.Linear(2, 2)), num_experts=2)


class ExpertRoutingTest(unittest.TestCase):
    def test_materialize_dense_and_topk_share_type(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]])
        probs = logits.softmax(dim=-1)
        dense = materialize_routing(logits, probs, top_k=None)
        sparse = materialize_routing(logits, probs, top_k=2)
        self.assertIsInstance(dense, ExpertRouting)
        self.assertTrue(dense.is_dense)
        self.assertEqual(sparse.top_k, 2)

    def test_sparse_weights_are_softmax_over_selected_logits(self):
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 3.0, 1.0]])
        # Gate probs need not equal softmax(logits); sparse weights still come
        # from top-k on logits + softmax over those selected logits.
        probs = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.1, 0.4]])
        sparse = materialize_routing(logits, probs, top_k=2)
        selected_logits, indices = logits.topk(2, dim=-1)
        expected = selected_logits.softmax(dim=-1)
        assert sparse.indices is not None
        assert sparse.weights is not None
        self.assertTrue(torch.equal(sparse.indices, indices))
        self.assertTrue(torch.allclose(sparse.weights, expected))
        renorm = probs.gather(-1, indices)
        renorm = renorm / renorm.sum(dim=-1, keepdim=True)
        self.assertFalse(torch.allclose(sparse.weights, renorm))


class MoEMetricsTest(unittest.TestCase):
    def test_metrics_ignore_masked_positions(self):
        routing = ExpertRouting(
            logits=torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]]),
            probabilities=torch.tensor(
                [[[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.20, 0.20, 0.60]]]
            ),
            indices=torch.tensor([[[0], [1], [2]]]),
            weights=torch.ones(1, 3, 1),
            mask=torch.tensor([[True, False, True]]),
        )
        usage = expert_usage(routing)
        self.assertTrue(torch.allclose(usage, torch.tensor([0.5, 0.0, 0.5])))
        self.assertTrue(torch.isfinite(load_balance_loss(routing)))
        self.assertGreaterEqual(float(effective_experts(routing)), 1.0)

    def test_routing_metrics_include_loggable_scalars(self):
        routing = ExpertRouting(
            logits=torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
            probabilities=torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]),
            indices=torch.tensor([[0], [1], [0]]),
            weights=torch.ones(3, 1),
            mask=torch.tensor([True, True, False]),
        )
        metrics = moe_routing_metrics(routing)
        self.assertTrue(torch.equal(tokens_per_expert(routing), torch.tensor([1, 1])))
        torch.testing.assert_close(expert_probability_mean(routing), torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(expert_route_weight_mean(routing), torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(metrics["dropped/ratio"], torch.tensor(0.0))

    def test_dense_routing_metrics_use_actual_route_weights(self):
        routing = ExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
            top_k=None,
        )
        metrics = moe_routing_metrics(routing)

        torch.testing.assert_close(expert_usage(routing), torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(expert_route_weight_mean(routing), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.equal(tokens_per_expert(routing), torch.tensor([2, 2])))
        torch.testing.assert_close(metrics["dropped/ratio"], torch.tensor(0.0))

    def test_dropped_ratio_respects_active_mask(self):
        routing = ExpertRouting(
            logits=torch.zeros(3, 2),
            probabilities=torch.full((3, 2), 0.5),
            indices=torch.zeros(3, 1, dtype=torch.long),
            weights=torch.tensor([[1.0], [0.0], [0.0]]),
            mask=torch.tensor([True, True, False]),
        )
        torch.testing.assert_close(dropped_token_ratio(routing), torch.tensor(0.5))

    def test_all_masked_metrics_ignore_nonfinite_values_and_gradients(self):
        logits = torch.full((2, 2), float("nan"), requires_grad=True)
        probabilities = torch.full((2, 2), float("nan"), requires_grad=True)
        weights = torch.full((2, 1), float("nan"), requires_grad=True)
        routing = ExpertRouting(
            logits=logits,
            probabilities=probabilities,
            indices=torch.zeros(2, 1, dtype=torch.long),
            weights=weights,
            mask=torch.zeros(2, dtype=torch.bool),
        )

        metrics = moe_routing_metrics(routing)

        for value in metrics.values():
            self.assertTrue(torch.isfinite(value).all())
        loss = (
            router_z_loss(routing)
            + expert_usage(routing).sum()
            + expert_probability_mean(routing).sum()
            + expert_route_weight_mean(routing).sum()
        )
        loss.backward()
        for gradient in (logits.grad, probabilities.grad, weights.grad):
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_metrics_avoid_tensor_scalar_checks(self):
        routing = ExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.full((2, 2), 0.5),
            indices=torch.tensor([[0], [1]]),
            weights=torch.ones(2, 1),
        )

        with (
            patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("MoE metrics called Tensor.item()"),
            ),
            patch.object(
                torch.Tensor,
                "__bool__",
                side_effect=AssertionError("MoE metrics converted Tensor to bool"),
            ),
        ):
            metrics = moe_routing_metrics(routing)

        self.assertIn("effective_experts", metrics)

    def test_metrics_do_not_read_nonfinite_masked_positions(self):
        for top_k in (None, 1):
            with self.subTest(top_k=top_k):
                kwargs = {}
                if top_k is not None:
                    kwargs = {
                        "indices": torch.tensor([[0], [1]]),
                        "weights": torch.tensor([[1.0], [float("nan")]]),
                    }
                routing = ExpertRouting(
                    logits=torch.tensor([[0.0, 0.0], [float("nan"), float("inf")]]),
                    probabilities=torch.tensor([[0.75, 0.25], [float("nan"), float("nan")]]),
                    mask=torch.tensor([True, False]),
                    **cast(Any, kwargs),
                )

                metrics = moe_routing_metrics(routing)

                self.assertTrue(torch.isfinite(router_z_loss(routing)))
                self.assertTrue(torch.isfinite(load_balance_loss(routing)))
                for value in metrics.values():
                    self.assertTrue(torch.isfinite(value))


class ExpandExpertTest(unittest.TestCase):
    def test_bank_weight_shape_from_module_and_tensor(self):
        conv = nn.Conv1d(4, 5, kernel_size=3)
        self.assertEqual(bank_weight_shape(conv, 4), torch.Size([4, 5, 4, 3]))
        self.assertEqual(
            bank_weight_shape(conv.weight, 4),
            torch.Size([4, 5, 4, 3]),
        )

    def test_expand_param_copy_tiles_single_weight(self):
        single = torch.randn(5, 4, 3)
        bank = expand_param(single, 3, init="copy")
        self.assertEqual(bank.shape, (3, 5, 4, 3))
        self.assertTrue(torch.allclose(bank[0], single))
        self.assertTrue(torch.allclose(bank[2], single))

    def test_expand_expert_copy_matches_conv1d_output(self):
        torch.manual_seed(0)
        template = nn.Conv1d(4, 5, kernel_size=3, padding=1)
        expert, weight, bias = expand_expert(template, num_experts=3, init="copy")
        self.assertIsInstance(expert, Conv1dExpert)
        self.assertEqual(weight.shape, (3, 5, 4, 3))
        assert bias is not None
        moe = MoE(
            num_experts=3,
            router=LinearRouter(4, 3),
            expert=expert,
            weight=weight,
            bias=bias,
            top_k=None,
        )
        # Router is unused for forward_manually; LinearRouter dim is irrelevant here.
        x = torch.randn(2, 4, 16)
        expected = torch.nn.functional.conv1d(x, template.weight, template.bias, padding=1)
        dense = moe.forward_manually(
            x, expert_routing_from_weights(torch.ones(2, 3) / 3)
        ).hidden_states
        self.assertTrue(torch.allclose(dense, expected, atol=1e-5))
        one_hot = moe.forward_manually(
            x,
            ExpertRouting(
                logits=torch.zeros(2, 3),
                probabilities=torch.full((2, 3), 1 / 3),
                indices=torch.tensor([[1], [2]]),
                weights=torch.ones(2, 1),
            ),
        ).hidden_states
        self.assertTrue(torch.allclose(one_hot, expected, atol=1e-5))

    def test_expand_expert_reads_causal_flag(self):
        expert, weight, bias = expand_expert(
            Conv1d(4, 5, kernel_size=3, padding=1, causal=True),
            num_experts=2,
            init="empty",
        )
        self.assertIsInstance(expert, Conv1dExpert)
        self.assertTrue(cast(Any, expert).causal)
        self.assertEqual(weight.shape, (2, 5, 4, 3))
        self.assertIsNotNone(bias)

    def test_expand_expert_kaiming_and_empty_smoke(self):
        linear_mod = nn.Linear(4, 5, bias=False)
        expert, weight, bias = expand_expert(linear_mod, num_experts=2, init="kaiming")
        self.assertIs(expert, linear)
        self.assertEqual(weight.shape, (2, 5, 4))
        self.assertIsNone(bias)
        self.assertTrue(torch.isfinite(weight).all())
        empty_weight = expand_param(linear_mod.weight, 2, init="empty")
        self.assertEqual(empty_weight.shape, (2, 5, 4))


class MoELossTest(unittest.TestCase):
    def test_moe_loss_combines_weighted_components(self):
        routing = ExpertRouting(
            logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            probabilities=torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            indices=torch.tensor([[0], [1]]),
            weights=torch.ones(2, 1),
        )
        output = moe_loss(
            routing,
            balance=0.1,
            z=0.01,
        )
        expected = output.balance_loss * 0.1 + output.z_loss * 0.01
        torch.testing.assert_close(output.total, expected)
        self.assertEqual(
            set(output.details()), {"balance", "z", "weighted_balance", "weighted_z", "total"}
        )


if __name__ == "__main__":
    unittest.main()
