import math
import unittest
from importlib import import_module

import torch
from torch.nn import functional as F

from anytrain.framework.rl import (
    ContinuousGRPOLoss,
    DPOLoss,
    GRPOLoss,
    NeighborGRPOLoss,
    gather_token_logps,
    group_relative_advantages,
    neighbor_logps_from_distances,
    sequence_logps,
)


class RLImportTest(unittest.TestCase):
    def test_framework_root_does_not_export_rl(self):
        framework = import_module("anytrain.framework")

        self.assertEqual(framework.__all__, [])
        self.assertFalse(hasattr(framework, "DPOLoss"))


class LogProbTest(unittest.TestCase):
    def test_gather_token_logps_and_sequence_sum(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]],
                [[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]],
            ]
        )
        token_ids = torch.tensor([[0, 2], [1, 0]])
        mask = torch.tensor([[True, True], [True, False]])

        token_logps = gather_token_logps(logits, token_ids)
        expected = F.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

        torch.testing.assert_close(token_logps, expected)
        torch.testing.assert_close(
            sequence_logps(token_logps, mask),
            torch.tensor([expected[0].sum(), expected[1, 0]]),
        )

    def test_gather_rejects_out_of_vocab_token(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            gather_token_logps(torch.zeros(2, 3), torch.tensor([0, 3]))

    def test_neighbor_logps_from_distances_uses_distance_softmax(self):
        distances = torch.tensor([[[[0.0, 1.0, 4.0]]]])
        mask = torch.tensor([[[[True, True, False]]]])

        logps = neighbor_logps_from_distances(distances, mask, temperature=2.0)

        expected = F.log_softmax(
            torch.tensor([[[[-0.0, -0.5, float("-inf")]]]]),
            dim=-1,
        )
        torch.testing.assert_close(logps, expected)

    def test_neighbor_logps_reject_empty_distribution(self):
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            neighbor_logps_from_distances(
                torch.zeros(1, 1, 1, 2),
                torch.zeros(1, 1, 1, 2, dtype=torch.bool),
            )


class DPOLossTest(unittest.TestCase):
    def test_reference_free_dpo_values_and_details_are_detached(self):
        chosen = torch.tensor([1.0, 0.5], requires_grad=True)
        rejected = torch.tensor([0.0, 0.5], requires_grad=True)
        loss_fn = DPOLoss(beta=1.0, reference_free=True)

        loss, details = loss_fn(
            policy_chosen_logps=chosen,
            policy_rejected_logps=rejected,
        )

        expected = torch.stack(
            [
                -F.logsigmoid(torch.tensor(1.0)),
                -F.logsigmoid(torch.tensor(0.0)),
            ]
        ).mean()
        torch.testing.assert_close(loss, expected)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(details["policy_margin"].requires_grad)
        self.assertTrue(torch.isclose(details["accuracy"], torch.tensor(0.5)))

    def test_reference_margin_cancels_policy_margin(self):
        loss_fn = DPOLoss(beta=2.0)

        loss, details = loss_fn(
            policy_chosen_logps=torch.tensor([3.0]),
            policy_rejected_logps=torch.tensor([1.0]),
            ref_chosen_logps=torch.tensor([5.0]),
            ref_rejected_logps=torch.tensor([3.0]),
        )

        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
        torch.testing.assert_close(details["preference_logit"], torch.tensor(0.0))

    def test_reference_logps_are_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            DPOLoss()(policy_chosen_logps=torch.ones(1), policy_rejected_logps=torch.zeros(1))


class AdvantageTest(unittest.TestCase):
    def test_group_relative_advantages_uses_valid_candidates(self):
        rewards = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
        mask = torch.tensor([[True, True, False], [True, True, True]])

        advantages = group_relative_advantages(rewards, mask)

        torch.testing.assert_close(advantages[0], torch.tensor([-1.0, 1.0, 0.0]))
        self.assertTrue(torch.isclose(advantages[1].mean(), torch.tensor(0.0), atol=1e-6))

    def test_group_relative_advantages_requires_two_candidates(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            group_relative_advantages(torch.tensor([[1.0, 2.0]]), torch.tensor([[True, False]]))


class GRPOLossTest(unittest.TestCase):
    def test_grpo_policy_loss_uses_group_advantages_and_clipping(self):
        old = torch.zeros(1, 2, 2)
        policy = torch.log(torch.tensor([[[0.8, 0.8], [1.5, 1.5]]]))
        rewards = torch.tensor([[1.0, 3.0]])
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        loss_fn = GRPOLoss(clip_range=0.2, token_reduction="sequence_mean")

        loss, details = loss_fn(
            policy_token_logps=policy,
            old_token_logps=old,
            rewards=rewards,
            response_mask=mask,
        )

        torch.testing.assert_close(loss, torch.tensor((0.8 - 1.2) / 2))
        torch.testing.assert_close(details["pg_loss"], loss.detach())
        torch.testing.assert_close(details["clip_fraction"], torch.tensor(0.5))
        torch.testing.assert_close(details["candidate_count"], torch.tensor(2.0))
        torch.testing.assert_close(details["token_count"], torch.tensor(4.0))

    def test_grpo_adds_reference_kl(self):
        policy = torch.zeros(1, 2, 1)
        old = torch.zeros_like(policy)
        reference = torch.full_like(policy, -1.0)
        rewards = torch.tensor([[1.0, 3.0]])
        mask = torch.ones_like(policy, dtype=torch.bool)
        loss_fn = GRPOLoss(kl_beta=0.5)

        loss, details = loss_fn(
            policy_token_logps=policy,
            old_token_logps=old,
            ref_token_logps=reference,
            rewards=rewards,
            response_mask=mask,
        )

        expected_kl = math.exp(-1.0)
        torch.testing.assert_close(details["kl_loss"], torch.tensor(expected_kl))
        torch.testing.assert_close(loss, details["pg_loss"] + 0.5 * details["kl_loss"])

    def test_grpo_keeps_gradients_on_policy_only(self):
        policy = torch.zeros(1, 2, 1, requires_grad=True)
        old = torch.zeros(1, 2, 1, requires_grad=True)
        reference = torch.full((1, 2, 1), -0.5, requires_grad=True)
        rewards = torch.tensor([[1.0, 3.0]], requires_grad=True)
        mask = torch.ones(1, 2, 1, dtype=torch.bool)

        loss, details = GRPOLoss(kl_beta=0.1)(
            policy_token_logps=policy,
            old_token_logps=old,
            ref_token_logps=reference,
            rewards=rewards,
            response_mask=mask,
        )
        loss.backward()

        self.assertIsNotNone(policy.grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)
        self.assertIsNone(rewards.grad)
        self.assertFalse(details["pg_loss"].requires_grad)

    def test_dpo_keeps_gradients_on_policy_only(self):
        chosen = torch.tensor([1.0], requires_grad=True)
        rejected = torch.tensor([0.0], requires_grad=True)
        ref_chosen = torch.tensor([0.0], requires_grad=True)
        ref_rejected = torch.tensor([0.0], requires_grad=True)

        loss, _ = DPOLoss()(
            policy_chosen_logps=chosen,
            policy_rejected_logps=rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
        )
        loss.backward()

        self.assertIsNotNone(chosen.grad)
        self.assertIsNotNone(rejected.grad)
        self.assertIsNone(ref_chosen.grad)
        self.assertIsNone(ref_rejected.grad)

    def test_grpo_requires_reference_when_kl_weight_is_positive(self):
        with self.assertRaisesRegex(ValueError, "ref_token_logps"):
            GRPOLoss(kl_beta=0.1)(
                policy_token_logps=torch.zeros(1, 2, 1),
                old_token_logps=torch.zeros(1, 2, 1),
                rewards=torch.tensor([[1.0, 2.0]]),
                response_mask=torch.ones(1, 2, 1, dtype=torch.bool),
            )

    def test_grpo_rejects_group_mask_without_tokens(self):
        with self.assertRaisesRegex(ValueError, "no response tokens"):
            GRPOLoss()(
                policy_token_logps=torch.zeros(1, 2, 1),
                old_token_logps=torch.zeros(1, 2, 1),
                rewards=torch.tensor([[1.0, 2.0]]),
                response_mask=torch.tensor([[[True], [False]]]),
                group_mask=torch.tensor([[True, True]]),
            )


class ContinuousGRPOLossTest(unittest.TestCase):
    def test_continuous_grpo_uses_step_logps_and_clipping(self):
        old = torch.zeros(1, 2, 2)
        policy = torch.log(torch.tensor([[[0.8, 0.8], [1.5, 1.5]]]))
        rewards = torch.tensor([[1.0, 3.0]])
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        loss_fn = ContinuousGRPOLoss(clip_range=0.2)

        loss, details = loss_fn(
            policy_step_logps=policy,
            old_step_logps=old,
            rewards=rewards,
            step_mask=mask,
        )

        torch.testing.assert_close(loss, torch.tensor((0.8 - 1.2) / 2))
        torch.testing.assert_close(details["pg_loss"], loss.detach())
        torch.testing.assert_close(details["clip_fraction"], torch.tensor(0.5))
        torch.testing.assert_close(details["candidate_count"], torch.tensor(2.0))
        torch.testing.assert_close(details["step_count"], torch.tensor(4.0))

    def test_continuous_grpo_adds_external_kl_values(self):
        policy = torch.zeros(1, 2, 1)
        old = torch.zeros_like(policy)
        rewards = torch.tensor([[1.0, 3.0]])
        step_mask = torch.ones_like(policy, dtype=torch.bool)
        kl_values = torch.full_like(policy, 0.25)

        loss, details = ContinuousGRPOLoss(kl_beta=0.5)(
            policy_step_logps=policy,
            old_step_logps=old,
            rewards=rewards,
            step_mask=step_mask,
            kl_values=kl_values,
        )

        torch.testing.assert_close(details["kl_loss"], torch.tensor(0.25))
        torch.testing.assert_close(loss, details["pg_loss"] + 0.5 * details["kl_loss"])

    def test_continuous_grpo_keeps_gradients_on_policy_only(self):
        policy = torch.zeros(1, 2, 1, requires_grad=True)
        old = torch.zeros(1, 2, 1, requires_grad=True)
        rewards = torch.tensor([[1.0, 3.0]], requires_grad=True)
        kl_values = torch.ones(1, 2, 1, requires_grad=True)
        step_mask = torch.ones(1, 2, 1, dtype=torch.bool)

        loss, details = ContinuousGRPOLoss(kl_beta=0.1)(
            policy_step_logps=policy,
            old_step_logps=old,
            rewards=rewards,
            step_mask=step_mask,
            kl_values=kl_values,
        )
        loss.backward()

        self.assertIsNotNone(policy.grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(rewards.grad)
        self.assertIsNone(kl_values.grad)
        self.assertFalse(details["pg_loss"].requires_grad)

    def test_continuous_grpo_requires_kl_when_weight_is_positive(self):
        with self.assertRaisesRegex(ValueError, "kl_values"):
            ContinuousGRPOLoss(kl_beta=0.1)(
                policy_step_logps=torch.zeros(1, 2, 1),
                old_step_logps=torch.zeros(1, 2, 1),
                rewards=torch.tensor([[1.0, 2.0]]),
                step_mask=torch.ones(1, 2, 1, dtype=torch.bool),
            )

    def test_continuous_grpo_rejects_group_mask_without_steps(self):
        with self.assertRaisesRegex(ValueError, "no rollout steps"):
            ContinuousGRPOLoss()(
                policy_step_logps=torch.zeros(1, 2, 1),
                old_step_logps=torch.zeros(1, 2, 1),
                rewards=torch.tensor([[1.0, 2.0]]),
                step_mask=torch.tensor([[[True], [False]]]),
                group_mask=torch.tensor([[True, True]]),
            )


class NeighborGRPOLossTest(unittest.TestCase):
    def test_neighbor_grpo_uses_surrogate_logps_and_clipping(self):
        old = torch.zeros(1, 1, 1, 2)
        policy = torch.log(torch.tensor([[[[0.8, 1.5]]]]))
        rewards = torch.tensor([[1.0, 3.0]])
        mask = torch.ones_like(policy, dtype=torch.bool)
        loss_fn = NeighborGRPOLoss(clip_range=0.2)

        loss, details = loss_fn(
            policy_neighbor_logps=policy,
            old_neighbor_logps=old,
            rewards=rewards,
            neighbor_mask=mask,
        )

        torch.testing.assert_close(loss, torch.tensor((0.8 - 1.2) / 2))
        torch.testing.assert_close(details["pg_loss"], loss.detach())
        torch.testing.assert_close(details["clip_fraction"], torch.tensor(0.5))
        torch.testing.assert_close(details["neighbor_count"], torch.tensor(2.0))
        torch.testing.assert_close(details["anchor_step_count"], torch.tensor(1.0))
        torch.testing.assert_close(details["candidate_count"], torch.tensor(2.0))

    def test_neighbor_grpo_adds_external_kl_values(self):
        policy = torch.zeros(1, 1, 1, 2)
        old = torch.zeros_like(policy)
        rewards = torch.tensor([[1.0, 3.0]])
        neighbor_mask = torch.ones_like(policy, dtype=torch.bool)
        kl_values = torch.full_like(policy, 0.25)

        loss, details = NeighborGRPOLoss(kl_beta=0.5)(
            policy_neighbor_logps=policy,
            old_neighbor_logps=old,
            rewards=rewards,
            neighbor_mask=neighbor_mask,
            kl_values=kl_values,
        )

        torch.testing.assert_close(details["kl_loss"], torch.tensor(0.25))
        torch.testing.assert_close(loss, details["pg_loss"] + 0.5 * details["kl_loss"])

    def test_neighbor_grpo_keeps_gradients_on_policy_only(self):
        policy = torch.zeros(1, 1, 1, 2, requires_grad=True)
        old = torch.zeros(1, 1, 1, 2, requires_grad=True)
        rewards = torch.tensor([[1.0, 3.0]], requires_grad=True)
        kl_values = torch.ones(1, 1, 1, 2, requires_grad=True)
        neighbor_mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)

        loss, details = NeighborGRPOLoss(kl_beta=0.1)(
            policy_neighbor_logps=policy,
            old_neighbor_logps=old,
            rewards=rewards,
            neighbor_mask=neighbor_mask,
            kl_values=kl_values,
        )
        loss.backward()

        self.assertIsNotNone(policy.grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(rewards.grad)
        self.assertIsNone(kl_values.grad)
        self.assertFalse(details["pg_loss"].requires_grad)

    def test_neighbor_grpo_supports_quasi_norm_advantages(self):
        policy = torch.zeros(1, 1, 1, 3)
        old = torch.zeros_like(policy)
        rewards = torch.tensor([[1.0, 3.0, 7.0]])
        neighbor_mask = torch.ones_like(policy, dtype=torch.bool)

        _, details = NeighborGRPOLoss(advantage_norm_p=1.0)(
            policy_neighbor_logps=policy,
            old_neighbor_logps=old,
            rewards=rewards,
            neighbor_mask=neighbor_mask,
        )

        torch.testing.assert_close(
            details["advantage_mean"],
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )

    def test_neighbor_grpo_requires_kl_when_weight_is_positive(self):
        with self.assertRaisesRegex(ValueError, "kl_values"):
            NeighborGRPOLoss(kl_beta=0.1)(
                policy_neighbor_logps=torch.zeros(1, 1, 1, 2),
                old_neighbor_logps=torch.zeros(1, 1, 1, 2),
                rewards=torch.tensor([[1.0, 2.0]]),
                neighbor_mask=torch.ones(1, 1, 1, 2, dtype=torch.bool),
            )

    def test_neighbor_grpo_rejects_group_mask_without_comparisons(self):
        with self.assertRaisesRegex(ValueError, "no neighbor comparisons"):
            NeighborGRPOLoss()(
                policy_neighbor_logps=torch.zeros(1, 1, 1, 2),
                old_neighbor_logps=torch.zeros(1, 1, 1, 2),
                rewards=torch.tensor([[1.0, 2.0]]),
                neighbor_mask=torch.tensor([[[[True, False]]]]),
                group_mask=torch.tensor([[True, True]]),
            )


if __name__ == "__main__":
    unittest.main()
