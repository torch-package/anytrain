import math
import unittest
from importlib import import_module

import torch
from torch.nn import functional as F

from anytrain.framework.rl import (
    DPOLoss,
    GRPOLoss,
    gather_token_logps,
    group_relative_advantages,
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


if __name__ == "__main__":
    unittest.main()
