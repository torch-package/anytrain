import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

import anytrain.optim as optim_api
from anytrain.optim import LoRAFactorPair, LoRAMuon
from anytrain.optim.lora_muon import _inverse_gram_root as _actual_inverse_gram_root
from anytrain.optim.lora_muon import _matrix_sign as _actual_matrix_sign
from anytrain.optim.lora_muon import _validate_finite_pair as _actual_validate_finite_pair


def _parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value.clone())


def _inverse_gram_root(factor: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(factor.mT @ factor)
    return (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.mT


def _matrix_sign(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right


class LoRAMuonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_step_matches_paper_factor_update_with_external_scale(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float64))
        right = _parameter(torch.randn(2, 4, dtype=torch.float64))
        left_before = left.detach().clone()
        right_before = right.detach().clone()
        left_grad = torch.randn_like(left)
        right_grad = torch.randn_like(right)
        left.grad = left_grad.clone()
        right.grad = right_grad.clone()

        lr = 0.08
        weight_decay = 0.3
        scale = 4.0
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right, scale=scale)],
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )

        sqrt_scale = math.sqrt(scale)
        paper_left = sqrt_scale * left_before
        paper_right = sqrt_scale * right_before.mT
        left_invroot = _inverse_gram_root(paper_left)
        right_invroot = _inverse_gram_root(paper_right)
        paper_left_grad = left_grad / sqrt_scale
        paper_right_grad = right_grad.mT / sqrt_scale
        left_increment = (
            -lr
            / 2.0
            * _matrix_sign(paper_left_grad @ right_invroot)
            @ right_invroot
        )
        right_increment = (
            -lr
            / 2.0
            * _matrix_sign(paper_right_grad @ left_invroot)
            @ left_invroot
        )
        decay_scale = math.sqrt(1.0 - lr * weight_decay)
        expected_left = (
            decay_scale * paper_left + left_increment / decay_scale
        ) / sqrt_scale
        expected_right = (
            decay_scale * paper_right + right_increment / decay_scale
        ).mT / sqrt_scale

        optimizer.step()

        torch.testing.assert_close(left, expected_left, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(right, expected_right, rtol=1e-10, atol=1e-10)

    def test_induced_weight_update_is_invariant_to_non_scalar_gauge(self):
        left_value = torch.randn(5, 2, dtype=torch.float64)
        right_value = torch.randn(2, 4, dtype=torch.float64)
        gauge = torch.tensor([[2.0, 0.3], [0.4, 1.5]], dtype=torch.float64)
        transformed_left_value = left_value @ gauge
        transformed_right_value = torch.linalg.solve(gauge, right_value)

        left = _parameter(left_value)
        right = _parameter(right_value)
        transformed_left = _parameter(transformed_left_value)
        transformed_right = _parameter(transformed_right_value)
        scale = 1.7
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right, scale=scale)],
            lr=0.05,
            momentum=0.8,
            weight_decay=0.2,
        )
        transformed_optimizer = LoRAMuon(
            [LoRAFactorPair(transformed_left, transformed_right, scale=scale)],
            lr=0.05,
            momentum=0.8,
            weight_decay=0.2,
        )

        for _ in range(3):
            ambient_grad = torch.randn(5, 4, dtype=torch.float64)
            left.grad = scale * ambient_grad @ right.detach().mT
            right.grad = scale * left.detach().mT @ ambient_grad
            transformed_left.grad = scale * ambient_grad @ transformed_right.detach().mT
            transformed_right.grad = scale * transformed_left.detach().mT @ ambient_grad
            optimizer.step()
            transformed_optimizer.step()

            torch.testing.assert_close(
                scale * left @ right,
                scale * transformed_left @ transformed_right,
                rtol=1e-9,
                atol=1e-9,
            )

    def test_momentum_is_accumulated_before_whitening(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float64))
        right = _parameter(torch.randn(2, 4, dtype=torch.float64))
        left_before = left.detach().clone()
        right_before = right.detach().clone()
        left_grad = torch.randn_like(left)
        right_grad = torch.randn_like(right)
        left.grad = left_grad.clone()
        right.grad = right_grad.clone()
        scale = 2.5
        lr = 0.06
        momentum = 0.7
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right, scale=scale)],
            lr=lr,
            momentum=momentum,
        )
        prior_left_momentum = torch.randn_like(left)
        prior_right_momentum = torch.randn_like(right)
        optimizer.state[left]["momentum_buffer"] = prior_left_momentum.clone()
        optimizer.state[right]["momentum_buffer"] = prior_right_momentum.clone()

        sqrt_scale = math.sqrt(scale)
        paper_left = sqrt_scale * left_before
        paper_right = sqrt_scale * right_before.mT
        left_invroot = _inverse_gram_root(paper_left)
        right_invroot = _inverse_gram_root(paper_right)
        expected_left_momentum = (
            momentum * prior_left_momentum
            + (1.0 - momentum) * left_grad / sqrt_scale
        )
        expected_right_momentum = (
            momentum * prior_right_momentum
            + (1.0 - momentum) * right_grad / sqrt_scale
        ).mT
        left_increment = (
            -lr
            / 2.0
            * _matrix_sign(expected_left_momentum @ right_invroot)
            @ right_invroot
        )
        right_increment = (
            -lr
            / 2.0
            * _matrix_sign(expected_right_momentum @ left_invroot)
            @ left_invroot
        )

        optimizer.step()

        torch.testing.assert_close(
            left,
            left_before + left_increment / sqrt_scale,
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            right,
            right_before + right_increment.mT / sqrt_scale,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertEqual(optimizer.state[right]["momentum_buffer"].shape, right.shape)

    def test_split_weight_decay_decays_the_product_once(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float64))
        right = _parameter(torch.randn(2, 4, dtype=torch.float64))
        left_before = left.detach().clone()
        right_before = right.detach().clone()
        left.grad = torch.zeros_like(left)
        right.grad = torch.zeros_like(right)
        lr = 0.2
        weight_decay = 0.4
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right)],
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )

        optimizer.step()

        decay_scale = math.sqrt(1.0 - lr * weight_decay)
        torch.testing.assert_close(left, decay_scale * left_before)
        torch.testing.assert_close(right, decay_scale * right_before)
        torch.testing.assert_close(
            left @ right,
            (1.0 - lr * weight_decay) * left_before @ right_before,
        )

    def test_matrix_sign_keeps_tiny_nonzero_singular_direction(self):
        left = _parameter(torch.eye(2, dtype=torch.float64))
        right = _parameter(torch.eye(2, dtype=torch.float64))
        left.grad = torch.diag(torch.tensor([1.0, 1e-18], dtype=torch.float64))
        right.grad = torch.zeros_like(right)
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right)],
            lr=0.2,
            momentum=0.0,
        )

        optimizer.step()

        torch.testing.assert_close(left, 0.9 * torch.eye(2, dtype=torch.float64))
        torch.testing.assert_close(right, torch.eye(2, dtype=torch.float64))

    def test_matrix_sign_does_not_sync_for_zero_singular_value_branch(self):
        calls = []
        original_bool = torch.Tensor.__bool__

        def tracked_bool(value):
            calls.append(value)
            return original_bool(value)

        matrix = torch.randn(5, 3, dtype=torch.float64)
        with patch.object(torch.Tensor, "__bool__", tracked_bool):
            result = _actual_matrix_sign(matrix)

        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.isfinite(result).all())

        calls.clear()
        with patch.object(torch.Tensor, "__bool__", tracked_bool):
            result = _actual_matrix_sign(torch.zeros_like(matrix))

        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(result, torch.zeros_like(matrix))

    def test_inverse_gram_root_combines_rank_predicates(self):
        calls = []
        original_bool = torch.Tensor.__bool__

        def tracked_bool(value):
            calls.append(value)
            return original_bool(value)

        factor = torch.randn(5, 3, dtype=torch.float64)
        with patch.object(torch.Tensor, "__bool__", tracked_bool):
            result = _actual_inverse_gram_root(factor, name="factor")

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(
            result @ (factor.mT @ factor) @ result,
            torch.eye(factor.shape[1], dtype=factor.dtype),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_finite_pair_validation_uses_one_scalar_conversion(self):
        calls = []
        original_bool = torch.Tensor.__bool__

        def tracked_bool(value):
            calls.append(value)
            return original_bool(value)

        with patch.object(torch.Tensor, "__bool__", tracked_bool):
            _actual_validate_finite_pair(
                torch.randn(5, 3),
                name="left",
                other=torch.randn(3, 4),
                other_name="right",
            )

        self.assertEqual(len(calls), 1)

    def test_state_dict_round_trip_preserves_next_update(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float64))
        right = _parameter(torch.randn(2, 4, dtype=torch.float64))
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right, scale=2.0)],
            lr=0.03,
            momentum=0.8,
        )
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        optimizer.step()

        restored_left = _parameter(left.detach())
        restored_right = _parameter(right.detach())
        restored_optimizer = LoRAMuon(
            [LoRAFactorPair(restored_left, restored_right, scale=2.0)],
            lr=1.0,
            momentum=0.0,
        )
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        next_left_grad = torch.randn_like(left)
        next_right_grad = torch.randn_like(right)
        left.grad = next_left_grad.clone()
        right.grad = next_right_grad.clone()
        restored_left.grad = next_left_grad.clone()
        restored_right.grad = next_right_grad.clone()

        optimizer.step()
        restored_optimizer.step()

        torch.testing.assert_close(left, restored_left, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(right, restored_right, rtol=1e-10, atol=1e-10)

    def test_state_dict_preserves_float32_momenta_for_float16_factors(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float16))
        right = _parameter(torch.randn(2, 4, dtype=torch.float16))
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right)],
            lr=0.03,
            momentum=0.83,
        )
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        optimizer.step()
        saved_state = copy.deepcopy(optimizer.state_dict())
        expected_momenta = [
            optimizer.state[left]["momentum_buffer"].clone(),
            optimizer.state[right]["momentum_buffer"].clone(),
        ]

        restored_left = _parameter(left.detach())
        restored_right = _parameter(right.detach())
        restored_optimizer = LoRAMuon(
            [LoRAFactorPair(restored_left, restored_right)],
            lr=0.03,
            momentum=0.83,
        )
        restored_optimizer.load_state_dict(saved_state)

        restored_momenta = [
            restored_optimizer.state[restored_left]["momentum_buffer"],
            restored_optimizer.state[restored_right]["momentum_buffer"],
        ]
        for actual, expected in zip(restored_momenta, expected_momenta, strict=False):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_state_dict_momentum_restore_respects_load_hooks(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float16))
        right = _parameter(torch.randn(2, 4, dtype=torch.float16))
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right)],
            lr=0.03,
            momentum=0.83,
        )
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        optimizer.step()
        saved_state = copy.deepcopy(optimizer.state_dict())
        original_momenta = [
            state["momentum_buffer"].clone()
            for state in saved_state["state"].values()
        ]

        restored_left = _parameter(left.detach())
        restored_right = _parameter(right.detach())
        restored_optimizer = LoRAMuon(
            [LoRAFactorPair(restored_left, restored_right)],
            lr=0.03,
            momentum=0.83,
        )

        def add_one_before_load(_, state_dict):
            for state in state_dict["state"].values():
                state["momentum_buffer"].add_(1.0)

        def double_after_load(loaded_optimizer):
            for state in loaded_optimizer.state.values():
                state["momentum_buffer"].mul_(2.0)

        pre_handle = restored_optimizer.register_load_state_dict_pre_hook(add_one_before_load)
        post_handle = restored_optimizer.register_load_state_dict_post_hook(double_after_load)
        try:
            restored_optimizer.load_state_dict(saved_state)
        finally:
            pre_handle.remove()
            post_handle.remove()

        restored_momenta = [
            restored_optimizer.state[restored_left]["momentum_buffer"],
            restored_optimizer.state[restored_right]["momentum_buffer"],
        ]
        for actual, original in zip(restored_momenta, original_momenta, strict=False):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, (original + 1.0) * 2.0, rtol=0.0, atol=0.0)

    def test_state_dict_rejects_unpaired_or_misshaped_momentum(self):
        left = _parameter(torch.randn(5, 2))
        right = _parameter(torch.randn(2, 4))
        optimizer = LoRAMuon([LoRAFactorPair(left, right)], lr=0.03)
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        optimizer.step()
        saved_state = copy.deepcopy(optimizer.state_dict())
        state_values = list(saved_state["state"].values())

        restored_left = _parameter(left.detach())
        restored_right = _parameter(right.detach())
        restored_optimizer = LoRAMuon(
            [LoRAFactorPair(restored_left, restored_right)],
            lr=0.03,
        )

        missing_momentum = copy.deepcopy(saved_state)
        del list(missing_momentum["state"].values())[0]["momentum_buffer"]
        with self.assertRaisesRegex(ValueError, "both momentum buffers or neither"):
            restored_optimizer.load_state_dict(missing_momentum)
        self.assertEqual(len(restored_optimizer.state), 0)

        state_values[0]["momentum_buffer"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "shape must match"):
            restored_optimizer.load_state_dict(saved_state)
        self.assertEqual(len(restored_optimizer.state), 0)

        invalid_momentum = copy.deepcopy(saved_state)
        for state in invalid_momentum["state"].values():
            state["momentum_buffer"] = "invalid"
        with self.assertRaisesRegex(TypeError, "momentum buffers must be tensors"):
            restored_optimizer.load_state_dict(invalid_momentum)
        self.assertEqual(len(restored_optimizer.state), 0)

    def test_rejects_rank_deficient_factor_at_construction(self):
        left = _parameter(torch.randn(5, 2))
        right = _parameter(torch.zeros(2, 4))

        with self.assertRaisesRegex(ValueError, "full column rank"):
            LoRAMuon([LoRAFactorPair(left, right)], lr=0.1)

    def test_rejects_factor_that_becomes_rank_deficient(self):
        left = _parameter(torch.randn(5, 2))
        right = _parameter(torch.randn(2, 4))
        optimizer = LoRAMuon([LoRAFactorPair(left, right)], lr=0.1)
        with torch.no_grad():
            right.zero_()
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)

        with self.assertRaisesRegex(ValueError, "full column rank"):
            optimizer.step()

    def test_rejects_rank_collapsing_update_before_mutating_state(self):
        left = _parameter(torch.ones(1, 1, dtype=torch.float64))
        right = _parameter(torch.ones(1, 1, dtype=torch.float64))
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right)],
            lr=2.0,
            momentum=0.0,
        )
        left.grad = torch.ones_like(left)
        right.grad = torch.zeros_like(right)

        with self.assertRaisesRegex(ValueError, "candidate left.*full column rank"):
            optimizer.step()

        torch.testing.assert_close(left, torch.ones_like(left))
        torch.testing.assert_close(right, torch.ones_like(right))
        self.assertEqual(len(optimizer.state), 0)

    def test_nonfinite_gradient_does_not_pollute_parameters_or_state(self):
        left = _parameter(torch.randn(5, 2, dtype=torch.float64))
        right = _parameter(torch.randn(2, 4, dtype=torch.float64))
        optimizer = LoRAMuon(
            [LoRAFactorPair(left, right)],
            lr=0.02,
            momentum=0.8,
        )
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        optimizer.step()
        left_before = left.detach().clone()
        right_before = right.detach().clone()
        left_momentum_before = optimizer.state[left]["momentum_buffer"].clone()
        right_momentum_before = optimizer.state[right]["momentum_buffer"].clone()

        left.grad = torch.full_like(left, float("nan"))
        right.grad = torch.randn_like(right)
        with self.assertRaisesRegex(ValueError, "left gradient.*finite"):
            optimizer.step()

        torch.testing.assert_close(left, left_before)
        torch.testing.assert_close(right, right_before)
        torch.testing.assert_close(
            optimizer.state[left]["momentum_buffer"],
            left_momentum_before,
        )
        torch.testing.assert_close(
            optimizer.state[right]["momentum_buffer"],
            right_momentum_before,
        )

        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        optimizer.step()

    def test_failure_in_later_pair_does_not_commit_earlier_pair(self):
        first_left = _parameter(torch.randn(5, 2, dtype=torch.float64))
        first_right = _parameter(torch.randn(2, 4, dtype=torch.float64))
        second_left = _parameter(torch.randn(6, 2, dtype=torch.float64))
        second_right = _parameter(torch.randn(2, 3, dtype=torch.float64))
        optimizer = LoRAMuon(
            [
                LoRAFactorPair(first_left, first_right),
                LoRAFactorPair(second_left, second_right),
            ],
            lr=0.02,
            momentum=0.8,
        )
        first_left_before = first_left.detach().clone()
        first_right_before = first_right.detach().clone()
        first_left.grad = torch.randn_like(first_left)
        first_right.grad = torch.randn_like(first_right)
        second_left.grad = torch.full_like(second_left, float("nan"))
        second_right.grad = torch.randn_like(second_right)

        with self.assertRaisesRegex(ValueError, "factor pair 1 left gradient.*finite"):
            optimizer.step()

        torch.testing.assert_close(first_left, first_left_before)
        torch.testing.assert_close(first_right, first_right_before)
        self.assertEqual(len(optimizer.state), 0)

    def test_rejects_partial_pair_gradients(self):
        left = _parameter(torch.randn(5, 2))
        right = _parameter(torch.randn(2, 4))
        optimizer = LoRAMuon([LoRAFactorPair(left, right)], lr=0.1)
        left.grad = torch.randn_like(left)

        with self.assertRaisesRegex(RuntimeError, "both factors"):
            optimizer.step()

    def test_rejects_invalid_pair_contract(self):
        left = _parameter(torch.randn(5, 2))
        wrong_rank = _parameter(torch.randn(3, 4))
        right = _parameter(torch.randn(2, 4))

        with self.assertRaisesRegex(ValueError, "incompatible rank"):
            LoRAMuon([LoRAFactorPair(left, wrong_rank)], lr=0.1)
        with self.assertRaisesRegex(ValueError, "unique"):
            LoRAMuon(
                [LoRAFactorPair(left, right), LoRAFactorPair(left, right)],
                lr=0.1,
            )
        with self.assertRaisesRegex(ValueError, "scale must be positive"):
            LoRAMuon([LoRAFactorPair(left, right, scale=0.0)], lr=0.1)

    def test_rejects_zero_rank_pair_at_contract_boundary(self):
        left = _parameter(torch.empty(5, 0))
        right = _parameter(torch.empty(0, 4))

        with self.assertRaisesRegex(ValueError, "rank dimension must be positive"):
            LoRAMuon([LoRAFactorPair(left, right)], lr=0.1)

    def test_rejects_unsafe_split_weight_decay(self):
        left = _parameter(torch.randn(5, 2))
        right = _parameter(torch.randn(2, 4))

        with self.assertRaisesRegex(ValueError, r"lr \* weight_decay < 1"):
            LoRAMuon(
                [LoRAFactorPair(left, right)],
                lr=0.5,
                weight_decay=2.0,
            )

    def test_does_not_allow_unpaired_param_groups(self):
        left = _parameter(torch.randn(5, 2))
        right = _parameter(torch.randn(2, 4))
        optimizer = LoRAMuon([LoRAFactorPair(left, right)], lr=0.1)

        with self.assertRaisesRegex(RuntimeError, "all factor pairs"):
            optimizer.add_param_group({"params": [_parameter(torch.randn(2, 2))]})

    def test_top_level_api_exports_lora_muon(self):
        self.assertIn("LoRAFactorPair", optim_api.__all__)
        self.assertIn("LoRAMuon", optim_api.__all__)


if __name__ == "__main__":
    unittest.main()
