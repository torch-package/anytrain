from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

import anytrain.module as module_api
from anytrain.module.dirichlet_tempering import ADT
from anytrain.module.moe import (
    CallableExpert,
    Conv1dExpert,
    ExpertRouting,
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
    moe_routing_metrics,
    moe_loss,
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


class MoETest(unittest.TestCase):
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
                expert=_scale_expert,
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
        expected = torch.nn.functional.conv1d(
            x, template.weight, template.bias, padding=1
        )
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
        self.assertTrue(expert.causal)
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
        self.assertEqual(set(output.details()), {"balance", "z", "weighted_balance", "weighted_z", "total"})


if __name__ == "__main__":
    unittest.main()
