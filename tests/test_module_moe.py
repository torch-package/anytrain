from __future__ import annotations

import unittest

import torch
from torch import nn

import anytrain.module as module_api
from anytrain.module.dirichlet_tempering import ADT
from anytrain.module.moe import (
    LinearTopKRouter,
    MoELayer,
    MoEOutput,
    RouterOutput,
    TokenChoiceMoE,
    TopKRouter,
    effective_experts,
    expert_usage,
    load_balance_loss,
    router_z_loss,
)
from anytrain.module.moe.routing import SparseExpertRouting


class ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FixedRouter(nn.Module):
    def __init__(self, output: SparseExpertRouting) -> None:
        super().__init__()
        self.output = output

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> SparseExpertRouting:
        if mask is not None:
            return RouterOutput(
                logits=self.output.logits,
                probabilities=self.output.probabilities,
                indices=self.output.indices,
                weights=self.output.weights,
                mask=mask,
            )
        return self.output


class TopKRouterTest(unittest.TestCase):
    def test_router_returns_topk_shapes_and_simplex_weights(self):
        router = TopKRouter(2, 3, top_k=2, bias=False)
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

        output = router(hidden)

        self.assertEqual(output.logits.shape, (1, 2, 3))
        self.assertEqual(output.probabilities.shape, (1, 2, 3))
        self.assertEqual(output.indices.shape, (1, 2, 2))
        self.assertEqual(output.weights.shape, (1, 2, 2))
        self.assertIsInstance(output, SparseExpertRouting)
        self.assertTrue(torch.equal(output.dense_weights.weights, output.probabilities))
        self.assertTrue(torch.equal(output.indices[0, 0], torch.tensor([0, 1])))
        self.assertTrue(torch.allclose(output.weights.sum(dim=-1), torch.ones(1, 2)))
        self.assertTrue(torch.allclose(output.probabilities.sum(dim=-1), torch.ones(1, 2)))

    def test_router_mask_zeroes_invalid_combine_weights(self):
        router = TopKRouter(2, 3, top_k=2)
        hidden = torch.randn(1, 3, 2)
        mask = torch.tensor([[True, False, True]])

        output = router(hidden, mask=mask)

        self.assertTrue(torch.allclose(output.weights[0, 0].sum(), torch.tensor(1.0)))
        self.assertTrue(torch.equal(output.weights[0, 1], torch.zeros(2)))
        self.assertTrue(torch.allclose(output.weights[0, 2].sum(), torch.tensor(1.0)))
        self.assertTrue(torch.equal(output.mask, mask))

    def test_router_passes_mask_to_adt_gate(self):
        gate = ADT.from_kwargs(num_experts=3)
        router = TopKRouter(2, 3, top_k=1, gate=gate)
        hidden = torch.randn(4, 2)

        router(hidden, mask=torch.zeros(4, dtype=torch.bool))

        self.assertEqual(gate.num_updates, 0)

    def test_router_can_control_adt_stat_collection(self):
        gate = ADT.from_kwargs(num_experts=3)
        router = TopKRouter(2, 3, top_k=1, gate=gate)
        hidden = torch.randn(4, 2)

        router(hidden, collect_stats=False)
        self.assertEqual(gate.num_updates, 0)

        router(hidden, collect_stats=True)
        self.assertEqual(gate.num_updates, 1)

    def test_linear_topk_router_alias(self):
        self.assertIs(LinearTopKRouter, TopKRouter)


class TokenChoiceMoETest(unittest.TestCase):
    def test_forward_manually_matches_dense_reference(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        routing = RouterOutput(
            logits=torch.zeros(1, 3, 3),
            probabilities=torch.full((1, 3, 3), 1 / 3),
            indices=torch.tensor([[[0, 1], [1, 2], [2, 0]]]),
            weights=torch.tensor([[[0.75, 0.25], [0.40, 0.60], [1.00, 0.00]]]),
        )
        moe = TokenChoiceMoE(
            router=FixedRouter(routing),
            experts=[
                ScaleExpert(1.0),
                ScaleExpert(10.0),
                ScaleExpert(-1.0),
            ],
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
        self.assertIs(output.routing, routing)

    def test_forward_ignores_masked_tokens(self):
        hidden = torch.ones(1, 3, 2)
        routing = RouterOutput(
            logits=torch.zeros(1, 3, 2),
            probabilities=torch.full((1, 3, 2), 0.5),
            indices=torch.zeros(1, 3, 1, dtype=torch.long),
            weights=torch.ones(1, 3, 1),
            mask=torch.tensor([[True, False, True]]),
        )
        moe = TokenChoiceMoE(
            router=FixedRouter(routing),
            experts=[ScaleExpert(2.0), ScaleExpert(3.0)],
        )

        output = moe.forward_manually(hidden, routing)

        self.assertTrue(torch.equal(output.hidden_states[0, 0], torch.full((2,), 2.0)))
        self.assertTrue(torch.equal(output.hidden_states[0, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(output.hidden_states[0, 2], torch.full((2,), 2.0)))
        self.assertTrue(torch.equal(output.tokens_per_expert.cpu(), torch.tensor([2, 0])))

    def test_empty_experts_are_allowed(self):
        hidden = torch.randn(2, 4)
        routing = RouterOutput(
            logits=torch.zeros(2, 3),
            probabilities=torch.full((2, 3), 1 / 3),
            indices=torch.zeros(2, 1, dtype=torch.long),
            weights=torch.ones(2, 1),
        )
        moe = MoELayer(
            router=FixedRouter(routing),
            experts=[
                nn.Linear(4, 5, bias=False),
                nn.Linear(4, 5, bias=False),
                nn.Linear(4, 5, bias=False),
            ],
        )

        output = moe.forward_manually(hidden, routing)

        self.assertEqual(output.hidden_states.shape, (2, 5))
        self.assertTrue(torch.equal(output.tokens_per_expert.cpu(), torch.tensor([2, 0, 0])))

    def test_forward_manually_accepts_sparse_expert_routing_base(self):
        hidden = torch.ones(2, 2)
        routing = SparseExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.full((2, 2), 0.5),
            indices=torch.tensor([[0], [1]]),
            weights=torch.ones(2, 1),
        )
        moe = TokenChoiceMoE(
            router=FixedRouter(routing),
            experts=[ScaleExpert(2.0), ScaleExpert(3.0)],
        )

        output = moe.forward_manually(hidden, routing)

        self.assertTrue(torch.equal(output.hidden_states, torch.tensor([[2.0, 2.0], [3.0, 3.0]])))
        self.assertIs(output.routing, routing)

    def test_forward_propagates_gradients_to_router_and_experts(self):
        torch.manual_seed(0)
        router = TopKRouter(4, 3, top_k=3)
        moe = TokenChoiceMoE(
            router=router,
            experts=[nn.Linear(4, 5) for _ in range(3)],
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
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertIsNotNone(router.proj.weight.grad)
        self.assertTrue(torch.isfinite(router.proj.weight.grad).all())
        for expert in moe.experts:
            self.assertIsNotNone(expert.weight.grad)
            self.assertTrue(torch.isfinite(expert.weight.grad).all())

    def test_exports_do_not_pollute_module_root(self):
        self.assertIs(MoELayer, TokenChoiceMoE)
        self.assertIsInstance(
            MoEOutput(
                hidden_states=torch.zeros(1, 2),
                routing=RouterOutput(
                    logits=torch.zeros(1, 1),
                    probabilities=torch.ones(1, 1),
                    indices=torch.zeros(1, 1, dtype=torch.long),
                    weights=torch.ones(1, 1),
                ),
                tokens_per_expert=torch.ones(1, dtype=torch.long),
            ),
            MoEOutput,
        )
        self.assertFalse(hasattr(module_api, "TopKRouter"))


class MoEMetricsTest(unittest.TestCase):
    def test_metrics_ignore_masked_positions(self):
        routing = RouterOutput(
            logits=torch.tensor(
                [
                    [
                        [2.0, 0.0, -1.0],
                        [0.0, 2.0, -1.0],
                        [0.0, -1.0, 2.0],
                    ]
                ]
            ),
            probabilities=torch.tensor(
                [
                    [
                        [0.80, 0.15, 0.05],
                        [0.10, 0.80, 0.10],
                        [0.20, 0.20, 0.60],
                    ]
                ]
            ),
            indices=torch.tensor([[[0], [1], [2]]]),
            weights=torch.ones(1, 3, 1),
            mask=torch.tensor([[True, False, True]]),
        )

        usage = expert_usage(routing)

        self.assertTrue(torch.allclose(usage, torch.tensor([0.5, 0.0, 0.5])))
        self.assertTrue(torch.isfinite(load_balance_loss(routing)))
        self.assertTrue(torch.isfinite(router_z_loss(routing)))
        self.assertGreaterEqual(float(effective_experts(routing)), 1.0)

    def test_metrics_return_zero_for_empty_masks(self):
        routing = RouterOutput(
            logits=torch.randn(2, 3),
            probabilities=torch.randn(2, 3).softmax(dim=-1),
            indices=torch.zeros(2, 1, dtype=torch.long),
            weights=torch.ones(2, 1),
            mask=torch.zeros(2, dtype=torch.bool),
        )

        self.assertTrue(torch.equal(expert_usage(routing), torch.zeros(3)))
        self.assertTrue(torch.equal(effective_experts(routing), torch.tensor(0.0)))
        self.assertTrue(torch.equal(load_balance_loss(routing), torch.tensor(0.0)))
        self.assertTrue(torch.equal(router_z_loss(routing), torch.tensor(0.0)))


if __name__ == "__main__":
    unittest.main()
