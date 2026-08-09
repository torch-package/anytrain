import importlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from anytrain import observation
from anytrain.loss import (
    FixedWeightLossBalancer,
    InterpolatedLossBalancer,
    LossABC,
    LossBalancerABC,
    LossGroup,
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
    MeanLossBalancer,
    PackedCodebookLogits,
    TeacherLoss,
    UncertaintyLossBalancer,
    UncertaintyNormalization,
    WeightedLossBalancerABC,
    teacher_loss,
)
from anytrain.loss.codec import CodecLoss, CodecLossPreset
from anytrain.loss.spectral import (
    CompressedSpectrogramLoss,
    LogMagnitudeLoss,
    MelLoss,
    MultiScaleMelLoss,
    MultiScaleSTFTLoss,
    SpectralRMSELoss,
    STFTLoss,
)
from anytrain.loss.temporal import SDRLoss, global_sdr


class ScalarLoss(LossABC):
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        return (prediction - target).abs().mean()


class VectorLoss(LossABC):
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        return (prediction - target).abs()


class TupleLoss(LossABC):
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = (prediction - target).abs().mean()
        return loss, {"raw": loss}


class SumLossBalancer(LossBalancerABC):
    def compute_loss(self, losses):
        return sum(losses.values())


class CustomWeightedLossBalancer(WeightedLossBalancerABC):
    def __init__(self):
        super().__init__(["first", "second"])

    @property
    def weights(self):
        return torch.tensor([0.25, 0.75])

    def compute_loss(self, losses):
        return (self._stack_losses(losses) * self.weights).sum()


class LossTest(unittest.TestCase):
    def test_loss_group(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup({"l1": nn.L1Loss(), "l2": nn.MSELoss()})

        total = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(2.0)))
        self.assertTrue(total.requires_grad)
        self.assertEqual(total.ndim, 0)
        self.assertIsInstance(loss.losses, nn.ModuleDict)

    def test_loss_group_accepts_custom_balancer(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup({"l1": nn.L1Loss(), "l2": nn.MSELoss()}, balancer=SumLossBalancer())

        total = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(4.0)))

    def test_fixed_weight_loss_balancer(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup(
            {"l1": nn.L1Loss(), "l2": nn.MSELoss()},
            balancer=FixedWeightLossBalancer({"l1": 1.0, "l2": 0.5}),
        )

        total = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(2.75)))
        self.assertEqual(total.ndim, 0)

    def test_fixed_weight_loss_balancer_normalizes_explicitly(self):
        balancer = FixedWeightLossBalancer(
            {"first": 1.0, "second": 3.0},
            normalize_weights=True,
        )

        total = balancer({"first": torch.tensor(2.0), "second": torch.tensor(4.0)})

        torch.testing.assert_close(balancer.weights, torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(total, torch.tensor(3.5))

    def test_fixed_weight_loss_balancer_rejects_nonfinite_weights(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                FixedWeightLossBalancer({"loss": value})

    def test_fixed_weight_loss_balancer_does_not_coerce_weights(self):
        for value in (True, "1.0"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "real number"):
                FixedWeightLossBalancer({"loss": value})

    def test_fixed_weight_loss_balancer_rejects_dtype_overflow(self):
        value = float(torch.finfo(torch.get_default_dtype()).max) * 2

        with self.assertRaisesRegex(ValueError, "finite"):
            FixedWeightLossBalancer({"loss": value})

    def test_fixed_weight_normalization_flag_requires_bool(self):
        with self.assertRaisesRegex(TypeError, "normalize_weights must be a bool"):
            FixedWeightLossBalancer({"loss": 1.0}, normalize_weights=1)

    def test_uncertainty_normalization_requires_known_string_enum(self):
        with self.assertRaisesRegex(TypeError, "UncertaintyNormalization or string"):
            UncertaintyLossBalancer(["loss"], normalization=1)
        with self.assertRaisesRegex(ValueError, "Unknown uncertainty normalization"):
            UncertaintyLossBalancer(["loss"], normalization="mean")

    def test_uncertainty_normalization_accepts_config_string(self):
        balancer = UncertaintyLossBalancer(["loss"], normalization="none")

        self.assertEqual(balancer.normalization, UncertaintyNormalization.NONE)

    def test_observed_balancers_reject_ambiguous_observation_names(self):
        constructors = (
            lambda names: FixedWeightLossBalancer(dict.fromkeys(names, 1.0)),
            UncertaintyLossBalancer,
        )
        for constructor in constructors:
            with (
                self.subTest(constructor=constructor, names=("/",)),
                self.assertRaisesRegex(ValueError, "non-empty observation tag"),
            ):
                constructor(["/"])
            with (
                self.subTest(constructor=constructor, names=("loss", "/loss/")),
                self.assertRaisesRegex(ValueError, "normalize to the same observation tag"),
            ):
                constructor(["loss", "/loss/"])

    def test_uncertainty_loss_balancer(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup(
            {"l1": nn.L1Loss(), "l2": nn.MSELoss()},
            balancer=UncertaintyLossBalancer(["l1", "l2"]),
        )

        total = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(2.0)))
        self.assertTrue(total.requires_grad)
        self.assertEqual(total.ndim, 0)

    def test_uncertainty_loss_balancer_detached_sum_surrogate_gradient(self):
        balancer = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.DETACHED_SUM,
        ).double()
        with torch.no_grad():
            balancer.log_var.copy_(torch.tensor([0.2, -0.1], dtype=torch.float64))
        first = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        second = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)

        total = balancer({"first": first, "second": second})
        total.backward()

        log_var = torch.tensor([0.2, -0.1], dtype=torch.float64)
        raw_weights = torch.exp(-log_var)
        weight_sum = raw_weights.sum()
        expected_total = (
            raw_weights[0] * first.detach()
            + raw_weights[1] * second.detach()
            + log_var.sum()
        ) / weight_sum
        torch.testing.assert_close(total.detach(), expected_total)
        torch.testing.assert_close(first.grad, raw_weights[0] / weight_sum)
        torch.testing.assert_close(second.grad, raw_weights[1] / weight_sum)
        torch.testing.assert_close(
            balancer.log_var.grad,
            (1 - raw_weights * torch.stack((first.detach(), second.detach()))) / weight_sum,
        )

    def test_uncertainty_loss_balancer_without_normalization(self):
        balancer = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.NONE,
        ).double()
        with torch.no_grad():
            balancer.log_var.copy_(torch.tensor([0.2, -0.1], dtype=torch.float64))
        losses = {
            "first": torch.tensor(2.0, dtype=torch.float64),
            "second": torch.tensor(4.0, dtype=torch.float64),
        }

        total = balancer(losses)

        raw_weights = torch.exp(-balancer.log_var.detach())
        expected = (raw_weights * torch.stack(tuple(losses.values()))).sum()
        expected = expected + balancer.log_var.detach().sum()
        torch.testing.assert_close(total.detach(), expected)

    def test_uncertainty_loss_balancer_computes_weights_once_per_forward(self):
        balancer = UncertaintyLossBalancer(["first", "second"])

        with patch("anytrain.loss.balancer.torch.exp", wraps=torch.exp) as exp:
            balancer({"first": torch.tensor(2.0), "second": torch.tensor(4.0)})

        self.assertEqual(exp.call_count, 1)

    def test_interpolated_loss_balancer_blends_objectives_and_weights(self):
        fixed = FixedWeightLossBalancer(
            {"first": 1.0, "second": 3.0},
            normalize_weights=True,
        )
        dynamic = UncertaintyLossBalancer(["first", "second"])
        with torch.no_grad():
            dynamic.log_var.copy_(torch.tensor([0.2, -0.1]))
        balancer = InterpolatedLossBalancer(
            fixed=fixed,
            dynamic=dynamic,
            blend=0.25,
        )
        first = torch.tensor(2.0, requires_grad=True)
        second = torch.tensor(4.0, requires_grad=True)
        losses = {"first": first, "second": second}

        total = balancer(losses)
        expected = torch.lerp(fixed(losses), dynamic(losses), balancer.blend)
        total.backward()

        torch.testing.assert_close(total.detach(), expected.detach())
        torch.testing.assert_close(
            balancer.weights,
            torch.lerp(fixed.weights, dynamic.weights, balancer.blend),
        )
        if first.grad is None or second.grad is None:
            self.fail("interpolated loss gradients are unavailable")
        torch.testing.assert_close(
            torch.stack((first.grad, second.grad)),
            balancer.weights.detach(),
        )

    def test_interpolated_loss_balancer_updates_blend_explicitly(self):
        fixed = FixedWeightLossBalancer(
            {"first": 1.0, "second": 3.0},
            normalize_weights=True,
        )
        dynamic = UncertaintyLossBalancer(["first", "second"])
        balancer = InterpolatedLossBalancer(fixed=fixed, dynamic=dynamic)

        torch.testing.assert_close(balancer.weights, fixed.weights)
        balancer.set_blend(1.0)
        torch.testing.assert_close(balancer.weights, dynamic.weights)

    def test_interpolated_loss_balancer_keeps_dynamic_parameters_in_zero_blend_graph(self):
        dynamic = UncertaintyLossBalancer(["first", "second"])
        balancer = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer(
                {"first": 1.0, "second": 1.0},
                normalize_weights=True,
            ),
            dynamic=dynamic,
        )

        balancer({"first": torch.tensor(2.0), "second": torch.tensor(4.0)}).backward()

        if dynamic.log_var.grad is None:
            self.fail("dynamic parameters were unused at blend=0")
        torch.testing.assert_close(dynamic.log_var.grad, torch.zeros_like(dynamic.log_var))

    def test_interpolated_loss_balancer_validates_configuration(self):
        fixed = FixedWeightLossBalancer({"first": 1.0, "second": 1.0})
        dynamic = UncertaintyLossBalancer(["first", "second"])

        for blend in (True, torch.tensor(0.5)):
            with self.subTest(blend=blend), self.assertRaisesRegex(TypeError, "host real"):
                InterpolatedLossBalancer(fixed=fixed, dynamic=dynamic, blend=blend)
        for blend in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(blend=blend), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                InterpolatedLossBalancer(fixed=fixed, dynamic=dynamic, blend=blend)
        with self.assertRaisesRegex(ValueError, "loss_names must match"):
            InterpolatedLossBalancer(
                fixed=fixed,
                dynamic=UncertaintyLossBalancer(["second", "first"]),
            )
        with self.assertRaisesRegex(TypeError, "WeightedLossBalancerABC"):
            InterpolatedLossBalancer(fixed=fixed, dynamic=MeanLossBalancer())

    def test_interpolated_loss_balancer_computes_dynamic_weights_once_per_forward(self):
        balancer = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 1.0, "second": 1.0}),
            dynamic=UncertaintyLossBalancer(["first", "second"]),
            blend=0.5,
        )

        with patch("anytrain.loss.balancer.torch.exp", wraps=torch.exp) as exp:
            balancer({"first": torch.tensor(2.0), "second": torch.tensor(4.0)})

        self.assertEqual(exp.call_count, 1)

    def test_uncertainty_loss_balancer_rejects_name_mismatch(self):
        balancer = UncertaintyLossBalancer(["l1"])

        with self.assertRaisesRegex(ValueError, "missing"):
            balancer({})

        with self.assertRaisesRegex(ValueError, "unknown"):
            balancer({"l1": torch.tensor(1.0), "l2": torch.tensor(2.0)})

    def test_named_loss_balancer_names_are_immutable(self):
        fixed = FixedWeightLossBalancer({"first": 1.0, "second": 2.0})
        uncertainty = UncertaintyLossBalancer(["first", "second"])

        self.assertEqual(fixed.loss_names, ("first", "second"))
        self.assertEqual(uncertainty.loss_names, ("first", "second"))

    def test_weighted_loss_balancers_share_public_weights_contract(self):
        fixed = FixedWeightLossBalancer({"first": 1.0})
        dynamic = UncertaintyLossBalancer(["first"])
        interpolated = InterpolatedLossBalancer(fixed=fixed, dynamic=dynamic)

        for balancer in (fixed, dynamic, interpolated):
            with self.subTest(type=type(balancer)):
                self.assertIsInstance(balancer, WeightedLossBalancerABC)
                self.assertEqual(balancer.weights.shape, (1,))
        self.assertFalse(hasattr(fixed, "fixed_weights"))
        self.assertFalse(hasattr(dynamic, "dynamic_weights"))
        self.assertFalse(hasattr(interpolated, "fixed_weights"))
        self.assertFalse(hasattr(interpolated, "dynamic_weights"))

    def test_weighted_loss_balancer_abc_is_a_custom_extension_contract(self):
        balancer = CustomWeightedLossBalancer()

        total = balancer(
            {
                "first": torch.tensor(2.0),
                "second": torch.tensor(4.0),
            }
        )

        self.assertEqual(balancer.loss_names, ("first", "second"))
        torch.testing.assert_close(total, torch.tensor(3.5))

    def test_named_loss_balancer_checkpoint_rejects_reordered_names(self):
        cases = (
            (
                FixedWeightLossBalancer({"first": 1.0, "second": 2.0}),
                FixedWeightLossBalancer({"second": 1.0, "first": 2.0}),
            ),
            (
                UncertaintyLossBalancer(["first", "second"]),
                UncertaintyLossBalancer(["second", "first"]),
            ),
        )
        for source, target in cases:
            before = {
                name: value.clone()
                for name, value in target.state_dict().items()
                if isinstance(value, torch.Tensor)
            }
            with self.subTest(type=type(source)), self.assertRaisesRegex(
                RuntimeError,
                "loss_names do not match",
            ):
                target.load_state_dict(source.state_dict())
            after = {
                name: value
                for name, value in target.state_dict().items()
                if isinstance(value, torch.Tensor)
            }
            self.assertEqual(before.keys(), after.keys())
            for name, value in before.items():
                torch.testing.assert_close(after[name], value)

    def test_uncertainty_checkpoint_rejects_normalization_mismatch(self):
        source = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.DETACHED_SUM,
        )
        target = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.NONE,
        )
        with torch.no_grad():
            source.log_var.fill_(1.0)
            target.log_var.fill_(2.0)

        with self.assertRaisesRegex(RuntimeError, "normalization='detached_sum'"):
            target.load_state_dict(source.state_dict())
        torch.testing.assert_close(target.log_var, torch.full_like(target.log_var, 2.0))

    def test_named_loss_balancer_rejects_checkpoint_without_name_metadata(self):
        source = FixedWeightLossBalancer({"first": 1.0, "second": 2.0})
        state = source.state_dict()
        state.pop("_extra_state")
        target = FixedWeightLossBalancer({"first": 3.0, "second": 4.0})
        before = target.weights.clone()

        with self.assertRaisesRegex(RuntimeError, "missing loss-name metadata"):
            target.load_state_dict(state, strict=False)

        torch.testing.assert_close(target.weights, before)

    def test_named_loss_balancer_rejects_malformed_checkpoint_metadata(self):
        source = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.DETACHED_SUM,
        )
        cases = (("version", True, "unsupported"), ("normalization", 1, "normalization=1"))

        for name, value, message in cases:
            state = source.state_dict()
            extra_state = dict(state["_extra_state"])
            extra_state[name] = value
            state["_extra_state"] = extra_state
            target = UncertaintyLossBalancer(
                ["first", "second"],
                normalization=UncertaintyNormalization.DETACHED_SUM,
            )
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, message):
                target.load_state_dict(state)

    def test_named_loss_balancer_checkpoint_round_trip(self):
        fixed = FixedWeightLossBalancer({"first": 1.0, "second": 2.0})
        restored_fixed = FixedWeightLossBalancer({"first": 3.0, "second": 4.0})
        restored_fixed.load_state_dict(fixed.state_dict())
        torch.testing.assert_close(restored_fixed.weights, fixed.weights)

        uncertainty = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.DETACHED_SUM,
        )
        with torch.no_grad():
            uncertainty.log_var.copy_(torch.tensor([0.2, -0.1]))
        restored_uncertainty = UncertaintyLossBalancer(
            ["first", "second"],
            normalization=UncertaintyNormalization.DETACHED_SUM,
        )
        restored_uncertainty.load_state_dict(uncertainty.state_dict())
        torch.testing.assert_close(restored_uncertainty.log_var, uncertainty.log_var)

        interpolated = InterpolatedLossBalancer(
            fixed=fixed,
            dynamic=uncertainty,
            blend=0.75,
        )
        restored_interpolated = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 3.0, "second": 4.0}),
            dynamic=UncertaintyLossBalancer(
                ["first", "second"],
                normalization=UncertaintyNormalization.DETACHED_SUM,
            ),
            blend=0.1,
        )
        restored_interpolated.load_state_dict(interpolated.state_dict())
        torch.testing.assert_close(restored_interpolated.blend, interpolated.blend)
        torch.testing.assert_close(restored_interpolated.weights, interpolated.weights)

    def test_interpolated_checkpoint_rejects_dynamic_config_before_loading_blend(self):
        source = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 1.0}),
            dynamic=UncertaintyLossBalancer(
                ["first"],
                normalization=UncertaintyNormalization.DETACHED_SUM,
            ),
            blend=0.75,
        )
        target = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 9.0}),
            dynamic=UncertaintyLossBalancer(
                ["first"],
                normalization=UncertaintyNormalization.NONE,
            ),
            blend=0.25,
        )
        with torch.no_grad():
            source.dynamic.log_var.fill_(1.0)
            target.dynamic.log_var.fill_(2.0)
        before = {
            name: value.clone()
            for name, value in target.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

        with self.assertRaisesRegex(RuntimeError, "dynamic_config"):
            target.load_state_dict(source.state_dict())

        after = {
            name: value
            for name, value in target.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        self.assertEqual(before.keys(), after.keys())
        for name, value in before.items():
            torch.testing.assert_close(after[name], value)

    def test_interpolated_checkpoint_preflights_child_metadata(self):
        source = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 1.0}),
            dynamic=UncertaintyLossBalancer(["first"]),
            blend=0.75,
        )
        target = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 9.0}),
            dynamic=UncertaintyLossBalancer(["first"]),
            blend=0.25,
        )
        with torch.no_grad():
            source.dynamic.log_var.fill_(1.0)
            target.dynamic.log_var.fill_(2.0)
        state = source.state_dict()
        dynamic_state = dict(state["dynamic._extra_state"])
        dynamic_state["normalization"] = UncertaintyNormalization.NONE.value
        state["dynamic._extra_state"] = dynamic_state
        before = {
            name: value.clone()
            for name, value in target.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

        with self.assertRaisesRegex(RuntimeError, "normalization='none'"):
            target.load_state_dict(state)

        after = {
            name: value
            for name, value in target.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        self.assertEqual(before.keys(), after.keys())
        for name, value in before.items():
            torch.testing.assert_close(after[name], value)

    def test_interpolated_checkpoint_preflights_child_tensor_schema(self):
        source = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 1.0}),
            dynamic=UncertaintyLossBalancer(["first"]),
            blend=0.75,
        )
        with torch.no_grad():
            source.dynamic.log_var.fill_(1.0)

        cases = (
            ("shape", torch.zeros(2)),
            ("missing", None),
            ("unexpected", torch.tensor(0.0)),
        )
        for kind, value in cases:
            state = source.state_dict()
            if kind == "shape":
                state["dynamic.log_var"] = value
            elif kind == "missing":
                state.pop("dynamic.log_var")
            else:
                state["dynamic.extra"] = value
            target = InterpolatedLossBalancer(
                fixed=FixedWeightLossBalancer({"first": 9.0}),
                dynamic=UncertaintyLossBalancer(["first"]),
                blend=0.25,
            )
            with torch.no_grad():
                target.dynamic.log_var.fill_(2.0)
            before = {
                name: tensor.clone()
                for name, tensor in target.state_dict().items()
                if isinstance(tensor, torch.Tensor)
            }

            with self.subTest(kind=kind), self.assertRaisesRegex(
                RuntimeError,
                "checkpoint (tensor|keys)",
            ):
                target.load_state_dict(state, strict=False)

            after = {
                name: tensor
                for name, tensor in target.state_dict().items()
                if isinstance(tensor, torch.Tensor)
            }
            self.assertEqual(before.keys(), after.keys())
            for name, tensor in before.items():
                torch.testing.assert_close(after[name], tensor)

    def test_interpolated_checkpoint_rejects_missing_or_invalid_blend(self):
        source = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 1.0}),
            dynamic=UncertaintyLossBalancer(["first"]),
            blend=0.75,
        )
        with torch.no_grad():
            source.dynamic.log_var.fill_(1.0)
        cases = (
            (None, "missing blend state"),
            (torch.tensor(2.0), "finite and between 0 and 1"),
            (torch.tensor(1), "scalar floating tensor"),
        )

        for blend, message in cases:
            state = source.state_dict()
            if blend is None:
                state.pop("blend")
            else:
                state["blend"] = blend
            target = InterpolatedLossBalancer(
                fixed=FixedWeightLossBalancer({"first": 9.0}),
                dynamic=UncertaintyLossBalancer(["first"]),
                blend=0.25,
            )
            with torch.no_grad():
                target.dynamic.log_var.fill_(2.0)
            before = {
                name: value.clone()
                for name, value in target.state_dict().items()
                if isinstance(value, torch.Tensor)
            }
            with self.subTest(blend=blend), self.assertRaisesRegex(RuntimeError, message):
                target.load_state_dict(state, strict=False)
            after = {
                name: value
                for name, value in target.state_dict().items()
                if isinstance(value, torch.Tensor)
            }
            self.assertEqual(before.keys(), after.keys())
            for name, value in before.items():
                torch.testing.assert_close(after[name], value)

    def test_nested_interpolated_checkpoint_failure_preserves_balancer_state(self):
        source = LossGroup(
            {"first": nn.L1Loss()},
            balancer=InterpolatedLossBalancer(
                fixed=FixedWeightLossBalancer({"first": 1.0}),
                dynamic=UncertaintyLossBalancer(["first"]),
                blend=0.75,
            ),
        )
        target = LossGroup(
            {"first": nn.L1Loss()},
            balancer=InterpolatedLossBalancer(
                fixed=FixedWeightLossBalancer({"first": 9.0}),
                dynamic=UncertaintyLossBalancer(["first"]),
                blend=0.25,
            ),
        )
        state = source.state_dict()
        state.pop("balancer.blend")
        with torch.no_grad():
            target.balancer.dynamic.log_var.fill_(2.0)
        before = {
            name: value.clone()
            for name, value in target.balancer.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

        with self.assertRaisesRegex(RuntimeError, "missing blend state"):
            target.load_state_dict(state, strict=False)

        after = {
            name: value
            for name, value in target.balancer.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        self.assertEqual(before.keys(), after.keys())
        for name, value in before.items():
            torch.testing.assert_close(after[name], value)

    def test_loss_group_checkpoint_names_nested_balancer_state(self):
        source = LossGroup(
            {"first": nn.L1Loss(), "second": nn.MSELoss()},
            balancer=UncertaintyLossBalancer(["first", "second"]),
        )
        state = source.state_dict()
        target = LossGroup(
            {"first": nn.L1Loss(), "second": nn.MSELoss()},
            balancer=UncertaintyLossBalancer(["first", "second"]),
        )

        self.assertIn("balancer._extra_state", state)
        target.load_state_dict(state)

    def test_stateful_loss_balancers_preserve_module_state_precision(self):
        losses = {
            "first": torch.tensor(2.0, dtype=torch.bfloat16),
            "second": torch.tensor(4.0, dtype=torch.bfloat16),
        }
        balancers = (
            FixedWeightLossBalancer({"first": 1.0, "second": 2.0}),
            UncertaintyLossBalancer(["first", "second"]),
            InterpolatedLossBalancer(
                fixed=FixedWeightLossBalancer({"first": 1.0, "second": 2.0}),
                dynamic=UncertaintyLossBalancer(["first", "second"]),
                blend=0.5,
            ),
        )

        for balancer in balancers:
            with self.subTest(type=type(balancer)):
                self.assertEqual(balancer(losses).dtype, torch.float32)

    def test_interpolated_blend_allows_promoted_objective_dtype(self):
        balancer = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer({"first": 1.0}),
            dynamic=UncertaintyLossBalancer(["first"]),
            blend=0.5,
        ).bfloat16()

        total = balancer({"first": torch.tensor(2.0, dtype=torch.float32)})

        self.assertEqual(total.dtype, torch.float32)
        torch.testing.assert_close(total, torch.tensor(2.0))

    def test_stateful_loss_balancers_require_matching_state_device(self):
        losses = {"loss": torch.ones((), device="meta")}
        balancers = (
            FixedWeightLossBalancer({"loss": 1.0}),
            UncertaintyLossBalancer(["loss"]),
            InterpolatedLossBalancer(
                fixed=FixedWeightLossBalancer({"loss": 1.0}),
                dynamic=UncertaintyLossBalancer(["loss"]),
                blend=0.5,
            ),
        )

        for balancer in balancers:
            with self.subTest(type=type(balancer)), self.assertRaisesRegex(
                RuntimeError,
                "same device",
            ):
                balancer(losses)

    def test_loss_abc_accepts_only_scalar_tensor(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss_fn = ScalarLoss()

        loss = loss_fn(pred, target)

        self.assertTrue(torch.isclose(loss, torch.tensor(1.5)))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.ndim, 0)

    def test_loss_abc_rejects_non_scalar_main_loss(self):
        loss_fn = VectorLoss()

        with self.assertRaisesRegex(ValueError, "scalar tensor"):
            loss_fn(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

    def test_loss_abc_rejects_tuple_result(self):
        loss_fn = TupleLoss()

        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            loss_fn(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

    def test_loss_group_rejects_non_scalar_child_result(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss_fn = LossGroup({"tuple": TupleLoss()})

        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            loss_fn(pred, target)

    def test_loss_group_names_are_not_logging_paths(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss_fn = LossGroup({"balancer": nn.L1Loss()})

        total = loss_fn(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(1.5)))

    def test_balancers_reject_non_scalar_inputs(self):
        for balancer in (
            SumLossBalancer(),
            FixedWeightLossBalancer({"loss": 1.0}),
            UncertaintyLossBalancer(["loss"]),
        ):
            with self.subTest(type=type(balancer)), self.assertRaisesRegex(
                ValueError,
                "scalar tensor",
            ):
                balancer({"loss": torch.ones(2)})

    def test_loss_modules_declare_observations(self):
        group_specs = observation.registry.specs(LossGroup({"l1": nn.L1Loss()}))
        frame_specs = observation.registry.specs(MaskedFrameMSELoss())
        compressed_specs = observation.registry.specs(CompressedSpectrogramLoss())
        fixed = FixedWeightLossBalancer({"l1": 1.0})
        dynamic = UncertaintyLossBalancer(["l1"])
        interpolated = InterpolatedLossBalancer(fixed=fixed, dynamic=dynamic)
        fixed_specs = observation.registry.specs(fixed)
        dynamic_specs = observation.registry.specs(dynamic)
        interpolated_specs = observation.registry.specs(interpolated)

        self.assertEqual([spec.name for spec in group_specs], ["loss"])
        self.assertEqual([spec.name for spec in frame_specs], ["loss", "diagnostics"])
        self.assertEqual([spec.name for spec in compressed_specs], ["loss", "diagnostics"])
        self.assertEqual([spec.name for spec in fixed_specs], ["weights", "fixed"])
        self.assertEqual([spec.name for spec in dynamic_specs], ["weights", "dynamic"])
        self.assertEqual(
            [spec.name for spec in interpolated_specs],
            ["weights", "fixed", "dynamic", "blend"],
        )
        self.assertIsInstance(group_specs[0], observation.OutputObservation)
        self.assertIsInstance(frame_specs[1], observation.ForwardEvent)
        self.assertIsInstance(fixed_specs[0], observation.StateObservation)
        self.assertTrue(any(spec.covers_descendants for spec in interpolated_specs))

    def test_interpolated_loss_balancer_observes_effective_and_endpoint_weights(self):
        fixed = FixedWeightLossBalancer(
            {"first": 1.0, "second": 3.0},
            normalize_weights=True,
        )
        dynamic = UncertaintyLossBalancer(["first", "second"])
        balancer = InterpolatedLossBalancer(
            fixed=fixed,
            dynamic=dynamic,
            blend=0.25,
        )
        specs = {
            spec.name: spec
            for spec in observation.registry.specs(balancer)
            if isinstance(spec, observation.StateObservation)
        }

        expected = {
            "weights": balancer.weights,
            "fixed": balancer.fixed.weights,
            "dynamic": balancer.dynamic.weights,
        }
        for group, expected_weights in expected.items():
            values = specs[group].collect(balancer)
            if not isinstance(values, dict):
                self.fail(f"{group} observation did not return a mapping")
            observed_weights: list[torch.Tensor] = []
            for name in balancer.loss_names:
                value = values[name]
                if not isinstance(value, observation.Curve):
                    self.fail(f"{group}/{name} observation is not a Curve")
                observed_weights.append(value.value)
            torch.testing.assert_close(torch.stack(observed_weights), expected_weights)

        blend = specs["blend"].collect(balancer)
        if not isinstance(blend, observation.Curve):
            self.fail("blend observation is not a Curve")
        torch.testing.assert_close(blend.value, balancer.blend)

    def test_loss_balancer_weight_observation_computes_each_vector_once(self):
        dynamic = UncertaintyLossBalancer(["first", "second", "third"])
        interpolated = InterpolatedLossBalancer(
            fixed=FixedWeightLossBalancer(
                {"first": 1.0, "second": 1.0, "third": 1.0},
                normalize_weights=True,
            ),
            dynamic=dynamic,
            blend=0.5,
        )
        specs = {
            spec.name: spec
            for spec in observation.registry.specs(interpolated)
            if isinstance(spec, observation.StateObservation)
        }

        with patch("anytrain.loss.balancer.torch.exp", wraps=torch.exp) as exp:
            values = specs["weights"].collect(interpolated)
        self.assertEqual(exp.call_count, 1)
        if not isinstance(values, dict):
            self.fail("weights observation did not return a mapping")
        self.assertTrue(
            all(
                isinstance(value, observation.Curve) and not value.value.requires_grad
                for value in values.values()
            )
        )

        with patch("anytrain.loss.balancer.torch.exp", wraps=torch.exp) as exp:
            specs["dynamic"].collect(interpolated)
        self.assertEqual(exp.call_count, 1)

    def test_masked_frame_mse_loss_ignores_padding(self):
        prediction = torch.tensor([[[1.0, 2.0], [1000.0, float("nan")]]], requires_grad=True)
        target = torch.tensor([[[3.0, 4.0], [float("inf"), 1000.0]]])
        mask = torch.tensor([[True, False]])

        loss = MaskedFrameMSELoss()(prediction, target, mask)
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(4.0))
        self.assertEqual(loss.ndim, 0)
        gradient = prediction.grad
        if gradient is None:
            self.fail("prediction gradient is unavailable")
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))

    def test_masked_frame_mse_loss_weights_all_valid_frames_equally(self):
        prediction = torch.tensor(
            [
                [[2.0], [100.0], [100.0]],
                [[0.0], [2.0], [4.0]],
            ]
        )
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[True, False, False], [True, True, True]])

        loss = MaskedFrameMSELoss()(prediction, target, mask)

        torch.testing.assert_close(loss, torch.tensor(6.0))

    def test_masked_frame_losses_reject_an_empty_reduction_domain(self):
        prediction = torch.zeros(2, 3, 4)
        target = torch.zeros_like(prediction)
        mask = torch.zeros(2, 3, dtype=torch.bool)

        for loss_fn in (MaskedFrameMSELoss(), MaskedCosineAlignmentLoss(), TeacherLoss()):
            with self.subTest(type=type(loss_fn)), self.assertRaisesRegex(
                ValueError,
                "at least one frame",
            ):
                loss_fn(prediction, target, mask)

    def test_masked_cosine_alignment_loss_detaches_target(self):
        prediction = torch.tensor(
            [[[1.0, 0.0], [float("nan"), float("inf")]]],
            requires_grad=True,
        )
        target = torch.tensor([[[1.0, 0.0], [float("nan"), float("inf")]]], requires_grad=True)
        mask = torch.tensor([[True, False]])

        loss = MaskedCosineAlignmentLoss()(prediction, target, mask)
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(0.0))
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)

    def test_teacher_loss_combines_cosine_and_smooth_l1(self):
        prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
        mask = torch.tensor([[True, True]])

        loss = TeacherLoss(smooth_l1_weight=1.0)(prediction, target, mask)
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.5)
        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)

    def test_teacher_loss_ignores_padding_and_zero_smooth_weight(self):
        prediction = torch.tensor(
            [[[1.0, 0.0], [float("nan"), float("inf")]]],
            requires_grad=True,
        )
        target = torch.tensor([[[0.0, 1.0], [float("nan"), float("inf")]]], requires_grad=True)
        mask = torch.tensor([[True, False]])

        loss = TeacherLoss(smooth_l1_weight=0.0)(prediction, target, mask)
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(1.0))
        gradient = prediction.grad
        if gradient is None:
            self.fail("prediction gradient is unavailable")
        self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))

    def test_teacher_loss_rejects_nonfinite_options_from_function_and_module(self):
        prediction = torch.ones(1, 1, 2)
        target = torch.zeros_like(prediction)
        mask = torch.ones(1, 1, dtype=torch.bool)

        for name, value in (
            ("smooth_l1_weight", float("nan")),
            ("smooth_l1_weight", float("inf")),
            ("beta", float("nan")),
            ("beta", float("-inf")),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "finite"):
                teacher_loss(prediction, target, mask, **{name: value})
            with (
                self.subTest(name=f"module_{name}", value=value),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                TeacherLoss(**{name: value})

    def test_masked_codebook_cross_entropy_emits_masked_top1(self):
        labels = torch.tensor(
            [
                [[1, 2], [0, 1], [-1, -1]],
                [[2, 3], [-1, -1], [-1, -1]],
            ]
        )
        mask = torch.tensor([[True, True, False], [True, False, False]])
        predictions = (
            torch.tensor([[1, 2, 0], [2, 0, 0]]),
            torch.tensor([[2, 1, 0], [0, 0, 0]]),
        )
        logits = tuple(
            torch.nn.functional.one_hot(value, classes).to(dtype=torch.float32)
            for value, classes in zip(predictions, (3, 4), strict=False)
        )

        with patch("anytrain.loss.codebook.emit_diagnostics") as emit:
            loss = MaskedCodebookCrossEntropyLoss()(
                logits,
                labels,
                mask,
                include_top1=True,
            )

        diagnostics = emit.call_args.args[1]
        self.assertEqual(loss.ndim, 0)
        torch.testing.assert_close(diagnostics["codebook_0_top1"], torch.tensor(2 / 3))
        torch.testing.assert_close(diagnostics["codebook_1_top1"], torch.tensor(2 / 3))
        torch.testing.assert_close(diagnostics["valid_frames"], torch.tensor(3.0))

    def test_masked_codebook_cross_entropy_reduces_all_valid_frames(self):
        labels = torch.tensor([[[1], [-1]], [[0], [2]]])
        mask = torch.tensor([[True, False], [True, True]])
        logits = (torch.randn(2, 2, 3),)

        loss = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)
        expected = torch.nn.functional.cross_entropy(logits[0][mask], labels[mask, 0])

        self.assertEqual(loss.ndim, 0)
        torch.testing.assert_close(loss, expected)

    def test_masked_codebook_cross_entropy_rejects_no_valid_frames(self):
        labels = torch.full((2, 1, 1), -1)
        mask = torch.zeros(2, 1, dtype=torch.bool)
        logits = (torch.zeros(2, 1, 3),)

        with self.assertRaisesRegex(ValueError, "at least one valid frame"):
            MaskedCodebookCrossEntropyLoss()(logits, labels, mask)

    def test_masked_codebook_cross_entropy_ignores_padding_logits(self):
        labels = torch.tensor([[[1], [0], [-1]]])
        mask = torch.tensor([[True, True, False]])
        logits = (torch.randn(1, 3, 3),)
        changed = (logits[0].clone(),)
        changed[0][~mask] = 1000

        baseline = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)
        padded = MaskedCodebookCrossEntropyLoss()(changed, labels, mask)
        torch.testing.assert_close(baseline, padded)

    def test_masked_codebook_cross_entropy_only_runs_ce_for_valid_frames(self):
        labels = torch.tensor(
            [
                [[1, 2], [-1, -1], [-1, -1]],
                [[0, 1], [2, 3], [-1, -1]],
            ]
        )
        mask = torch.tensor([[True, False, False], [True, True, False]])
        logits = (
            torch.randn(2, 3, 3, requires_grad=True),
            torch.randn(2, 3, 4, requires_grad=True),
        )
        observed_shapes: list[tuple[int, ...]] = []
        cross_entropy = torch.nn.functional.cross_entropy

        def record(value, target, **kwargs):
            observed_shapes.append(tuple(value.shape))
            return cross_entropy(value, target, **kwargs)

        with patch("anytrain.loss.codebook.F.cross_entropy", side_effect=record):
            loss = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)
            loss.backward()

        self.assertEqual(observed_shapes, [(3, 3), (3, 4)])
        for value in logits:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.equal(value.grad[~mask], torch.zeros_like(value.grad[~mask])))

    def test_packed_codebook_cross_entropy_matches_masked_contract(self):
        labels = torch.tensor(
            [
                [[1, 2], [-1, -1], [-1, -1]],
                [[0, 1], [2, 3], [-1, -1]],
            ]
        )
        mask = torch.tensor([[True, False, False], [True, True, False]])
        logits = (torch.randn(2, 3, 3), torch.randn(2, 3, 4))
        frame_indices = mask.flatten().nonzero().flatten()
        packed = PackedCodebookLogits(
            logits=tuple(value.flatten(0, 1).index_select(0, frame_indices) for value in logits),
            labels=labels.flatten(0, 1).index_select(0, frame_indices),
        )

        masked = MaskedCodebookCrossEntropyLoss()(
            logits,
            labels,
            mask,
            include_top1=True,
        )
        direct = MaskedCodebookCrossEntropyLoss()(
            packed,
            include_top1=True,
        )

        torch.testing.assert_close(direct, masked)
        self.assertEqual(direct.ndim, 0)

    def test_packed_codebook_routes_through_one_output_observation_hook(self):
        labels = torch.tensor([[0], [1]])
        packed = PackedCodebookLogits((torch.randn(2, 3),), labels)
        loss_fn = MaskedCodebookCrossEntropyLoss()
        output_spec = next(
            spec
            for spec in observation.registry.specs(loss_fn)
            if isinstance(spec, observation.OutputObservation)
        )
        observed: list[observation.Curve] = []

        def collect(module, inputs, kwargs, output):
            value = output_spec.collect(module, inputs, kwargs, output)
            if not isinstance(value, observation.Curve):
                self.fail("loss output collector did not return a Curve")
            observed.append(value)

        handle = loss_fn.register_forward_hook(collect, with_kwargs=True)
        try:
            packed_loss = loss_fn(packed)
            self.assertEqual(len(observed), 1)
            torch.testing.assert_close(observed.pop().value, packed_loss)

            dense_logits = (packed.logits[0].reshape(1, 2, 3),)
            dense_labels = labels.reshape(1, 2, 1)
            dense_loss = loss_fn(
                dense_logits,
                dense_labels,
                torch.ones(1, 2, dtype=torch.bool),
            )
            self.assertEqual(len(observed), 1)
            torch.testing.assert_close(observed[0].value, dense_loss)
        finally:
            handle.remove()

    def test_masked_codebook_cross_entropy_top1_does_not_change_loss(self):
        labels = torch.tensor([[[1, 2], [0, 1], [-1, -1]]])
        mask = torch.tensor([[True, True, False]])
        logits = tuple(torch.randn(1, 3, classes, requires_grad=True) for classes in (3, 4))

        baseline = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)
        with_top1 = MaskedCodebookCrossEntropyLoss()(
            logits,
            labels,
            mask,
            include_top1=True,
        )

        torch.testing.assert_close(with_top1, baseline)

    def test_masked_codebook_cross_entropy_validation_can_be_skipped(self):
        labels = torch.tensor([[[1], [0]]])
        mask = torch.tensor([[True, False]])
        logits = (torch.randn(1, 2, 3),)

        with patch(
            "anytrain.loss.codebook._validate_inputs",
            side_effect=AssertionError("validation should be skipped"),
        ):
            loss = MaskedCodebookCrossEntropyLoss()(logits, labels, mask, validate=False)
        self.assertEqual(loss.ndim, 0)

    def test_masked_codebook_cross_entropy_validates_large_vocab_with_small_labels(self):
        labels = torch.tensor([[[127], [1]]], dtype=torch.int8)
        mask = torch.tensor([[True, True]])
        logits = (torch.randn(1, 2, 256),)

        loss = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)

        self.assertEqual(loss.ndim, 0)

    def test_masked_codebook_cross_entropy_requires_one_device(self):
        labels = torch.tensor([[[1]]])
        mask = torch.tensor([[True]])
        logits = (torch.empty((1, 1, 2), device="meta"),)

        with self.assertRaisesRegex(ValueError, "share a device"):
            MaskedCodebookCrossEntropyLoss()(logits, labels, mask)

    def test_loss_group_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(TypeError, "LossBalancerABC"):
            LossGroup({"loss": nn.L1Loss()}, balancer=nn.Identity())

        with self.assertRaisesRegex(TypeError, "mapping"):
            LossGroup([nn.L1Loss()])

    def test_sdr_loss_is_near_zero_for_identical_audio(self):
        audio = torch.randn(2, 1, 64)

        loss = SDRLoss()(audio, audio)

        self.assertLess(float(loss), 0.0)

    def test_sdr_loss_handles_mono_batch_time_layout(self):
        reference = torch.randn(4, 32)
        estimate = reference + 0.1 * torch.randn(4, 32)

        score = global_sdr(estimate, reference)
        loss = SDRLoss(scale_invariant=False)(estimate, reference)

        self.assertEqual(tuple(score.shape), (4,))
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(float(loss), float(-score.mean()), places=5)

    def test_spectral_single_losses(self):
        estimate = torch.randn(2, 8, 4, dtype=torch.complex64)
        reference = estimate + 0.1 * torch.randn(2, 8, 4, dtype=torch.complex64)

        self.assertEqual(LogMagnitudeLoss()(estimate, reference).ndim, 0)
        self.assertEqual(CompressedSpectrogramLoss()(estimate, reference).ndim, 0)
        self.assertEqual(SpectralRMSELoss()(estimate, reference).ndim, 0)

    def test_compressed_spectrogram_loss_emits_nonredundant_components(self):
        estimate = torch.randn(2, 8, 4, dtype=torch.complex64)
        reference = estimate + 0.1 * torch.randn(2, 8, 4, dtype=torch.complex64)

        with patch("anytrain.loss.spectral.single.emit_diagnostics") as emit:
            total = CompressedSpectrogramLoss()(estimate, reference)

        diagnostics = emit.call_args.args[1]
        self.assertEqual(set(diagnostics), {"magnitude", "real", "imag"})
        torch.testing.assert_close(total, sum(diagnostics.values()))

    def test_stft_and_mel_losses(self):
        estimate = torch.randn(2, 1, 64, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(2, 1, 64)

        stft_loss = STFTLoss(n_fft=16)(estimate, reference)
        mel_loss = MelLoss(sample_rate=16000, n_fft=32, n_mels=8)(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)

    def test_spectral_transforms_are_private_loss_children(self):
        spectral = importlib.import_module("anytrain.loss.spectral")
        stft_loss = STFTLoss(n_fft=64)
        mel_loss = MelLoss(
            sample_rate=16000,
            n_fft=32,
            n_mels=8,
        )

        self.assertFalse(hasattr(spectral, "STFTTransform"))
        self.assertFalse(hasattr(spectral, "MelSpectrogramTransform"))
        self.assertIsInstance(stft_loss.transform, nn.Module)
        self.assertIsInstance(mel_loss.transform, nn.Module)
        self.assertEqual(stft_loss.transform.hop_length, 16)

    def test_mel_loss_supports_mel_scale(self):
        htk_mel = MelLoss(
            sample_rate=16000,
            n_fft=32,
            n_mels=8,
            mel_scale="htk",
        )
        slaney_mel = MelLoss(
            sample_rate=16000,
            n_fft=32,
            n_mels=8,
            mel_scale="slaney",
        )

        self.assertEqual(htk_mel.transform.mel_scale, "htk")
        self.assertEqual(slaney_mel.transform.mel_scale, "slaney")
        self.assertFalse(
            torch.allclose(htk_mel.transform.mel_filter, slaney_mel.transform.mel_filter)
        )

    def test_mel_loss_forwards_mel_scale(self):
        mel_loss = MelLoss(sample_rate=16000, n_fft=32, n_mels=8, mel_scale="slaney")
        multi_mel_loss = MultiScaleMelLoss(
            sample_rate=16000,
            n_fft_list=(32,),
            n_mels_list=(8,),
            mel_scale="slaney",
        )

        self.assertEqual(mel_loss.transform.mel_scale, "slaney")
        self.assertEqual(multi_mel_loss.losses["mel_32_8"].transform.mel_scale, "slaney")

    def test_mel_loss_rejects_unknown_mel_scale(self):
        with self.assertRaisesRegex(ValueError, "mel_scale"):
            MelLoss(sample_rate=16000, mel_scale="unknown")

    def test_multi_scale_spectral_losses(self):
        estimate = torch.randn(2, 1, 64, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(2, 1, 64)
        stft_loss_fn = MultiScaleSTFTLoss(n_fft_list=(16, 8))
        mel_loss_fn = MultiScaleMelLoss(
            sample_rate=16000,
            n_fft_list=(32, 16),
            n_mels_list=(8, 4),
        )

        stft_loss = stft_loss_fn(estimate, reference)
        mel_loss = mel_loss_fn(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)
        self.assertIsInstance(stft_loss_fn, LossGroup)
        self.assertIsInstance(mel_loss_fn, LossGroup)

    def test_domain_spectral_loss_reuses_its_transform_across_terms(self):
        estimate = torch.randn(1, 1, 64)
        reference = torch.randn_like(estimate)
        loss_fn = STFTLoss(
            n_fft=16,
            losses={
                "log": LogMagnitudeLoss(),
                "complex": SpectralRMSELoss(),
            },
        )

        with patch("torch.stft", wraps=torch.stft) as stft:
            loss = loss_fn(estimate, reference)

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(stft.call_count, 2)
        self.assertNotIsInstance(loss_fn, LossGroup)

    def test_spectral_losses_do_not_replace_explicit_empty_or_coerced_config(self):
        with self.assertRaisesRegex(ValueError, "at least one loss"):
            STFTLoss(losses={})

        for scales in ((16.0,), ("16",), (True,)):
            with self.subTest(scales=scales), self.assertRaisesRegex(TypeError, "integers"):
                MultiScaleSTFTLoss(n_fft_list=scales)

        with self.assertRaisesRegex(ValueError, "positive"):
            MultiScaleSTFTLoss(n_fft_list=(0,))

    def test_codec_loss_keeps_spectral_transforms_independent(self):
        loss = CodecLoss.from_preset(
            "dynacodec",
            sample_rate=16000,
            mel_n_fft_list=(16, 8),
            mel_n_mels_list=(8, 4),
            stft_n_fft_list=(16, 8),
        )
        mel = loss.losses["multi_mel"].losses["mel_16_8"].transform.stft
        stft = loss.losses["multi_stft"].losses["stft_16"].transform

        self.assertIsNot(mel, stft)

    def test_codec_loss_presets(self):
        dac_loss = CodecLoss.from_preset(CodecLossPreset.DAC)
        dynacodec_loss = CodecLoss.from_preset("dynacodec")

        self.assertEqual(set(dac_loss.losses), {"multi_mel"})
        self.assertEqual(set(dynacodec_loss.losses), {"si_sdr", "multi_mel", "multi_stft"})
        self.assertIn("si_sdr", dynacodec_loss.balancer.loss_names)

    def test_codec_loss_task_import_is_not_provided(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("anytrain.loss.task.codec")

    def test_codec_loss_preset_returns_scalar(self):
        estimate = torch.randn(1, 1, 4096, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(1, 1, 4096)
        loss = CodecLoss.from_preset("dynacodec")

        total = loss(estimate, reference)

        self.assertEqual(total.ndim, 0)

    def test_codec_loss_routes_every_registered_loss_through_module_call(self):
        loss = CodecLoss.from_preset(
            "dynacodec",
            sample_rate=16000,
            mel_n_fft_list=(16,),
            mel_n_mels_list=(4,),
            stft_n_fft_list=(16,),
        )
        calls = {
            name or "<root>": 0
            for name, module in loss.named_modules()
            if any(
                isinstance(spec, observation.OutputObservation)
                for spec in observation.registry.specs(module)
            )
        }
        handles = []
        for name, module in loss.named_modules():
            path = name or "<root>"
            if path not in calls:
                continue

            def count(_module, _inputs, _output, *, path=path):
                calls[path] += 1

            handles.append(module.register_forward_hook(count))

        try:
            loss(torch.randn(1, 1, 64), torch.randn(1, 1, 64))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(
            set(calls),
            {
                "<root>",
                "losses.si_sdr",
                "losses.multi_mel",
                "losses.multi_mel.losses.mel_16_4",
                "losses.multi_mel.losses.mel_16_4.losses.log_magnitude",
                "losses.multi_stft",
                "losses.multi_stft.losses.stft_16",
                "losses.multi_stft.losses.stft_16.losses.log_magnitude",
            },
        )
        self.assertEqual(set(calls.values()), {1})

    def test_codec_loss_rejects_unknown_preset(self):
        with self.assertRaisesRegex(ValueError, "Unknown codec loss preset"):
            CodecLoss.from_preset("unknown")


if __name__ == "__main__":
    unittest.main()
