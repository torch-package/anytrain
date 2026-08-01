from __future__ import annotations

import unittest

import torch

from anytrain.module.routing import (
    DenseExpertWeights,
    SparseExpertRouting,
    flat_active_mask,
    validate_dense_expert_weights,
    validate_sparse_expert_routing,
)


class DenseExpertWeightsTest(unittest.TestCase):
    def test_validate_returns_underlying_tensor(self):
        weights = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        expert_weights = DenseExpertWeights(weights)

        validated = validate_dense_expert_weights(expert_weights, num_experts=2, ndim=2)

        self.assertIs(validated, weights)
        self.assertEqual(expert_weights.num_experts, 2)
        self.assertEqual(expert_weights.leading_shape, torch.Size([2]))
        self.assertTrue(torch.equal(expert_weights.flatten(), weights))

    def test_validate_rejects_wrong_expert_count(self):
        with self.assertRaisesRegex(ValueError, "num_experts"):
            validate_dense_expert_weights(torch.ones(2, 3), num_experts=2, ndim=2)

    def test_validate_rejects_wrong_rank_when_requested(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            validate_dense_expert_weights(torch.ones(2, 3, 4), num_experts=4, ndim=2)


class SparseExpertRoutingTest(unittest.TestCase):
    def test_sparse_routing_exposes_dense_weights_and_active_mask(self):
        routing = SparseExpertRouting(
            logits=torch.zeros(1, 3, 2),
            probabilities=torch.full((1, 3, 2), 0.5),
            indices=torch.zeros(1, 3, 1, dtype=torch.long),
            weights=torch.ones(1, 3, 1),
            mask=torch.tensor([[True, False, True]]),
        )

        active_mask = routing.active_mask()

        self.assertTrue(torch.equal(routing.dense_weights.weights, routing.probabilities))
        self.assertTrue(torch.equal(active_mask, torch.tensor([True, False, True])))
        self.assertEqual(routing.num_experts, 2)
        self.assertEqual(routing.top_k, 1)

    def test_flat_active_mask_broadcasts_to_leading_shape(self):
        mask = torch.tensor([[True], [False]])

        active_mask = flat_active_mask(
            mask,
            leading_shape=torch.Size([2, 3]),
            device=mask.device,
        )

        self.assertTrue(
            torch.equal(
                active_mask,
                torch.tensor([True, True, True, False, False, False]),
            )
        )

    def test_validate_sparse_rejects_invalid_expert_id(self):
        routing = SparseExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.full((2, 2), 0.5),
            indices=torch.tensor([[0], [2]]),
            weights=torch.ones(2, 1),
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            validate_sparse_expert_routing(
                routing,
                leading_shape=torch.Size([2]),
                num_experts=2,
                device=routing.probabilities.device,
            )


if __name__ == "__main__":
    unittest.main()
