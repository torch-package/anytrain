import sys
import unittest
from unittest import mock

import torch
from torch import nn

from anytrain import observation
from anytrain.framework.gan import GAN, Loss, Preset, Reduction


def _run_with_observations(call):
    events = {}

    def capture(module, name, value):
        del module
        events[name] = value

    with mock.patch.object(observation, "emit", side_effect=capture):
        loss = call()
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise AssertionError("GAN objectives must return a scalar Tensor")

    diagnostics = {}
    for value in events.values():
        if isinstance(value, observation.Curve):
            diagnostics["value"] = value.value
        else:
            for name, item in value.items():
                if isinstance(item, observation.Curve):
                    diagnostics[name] = item.value
    return loss, diagnostics, events


class IdentityDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        return [[x]]


class ScaleBranchDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        return [[x], [2 * x]]


class TensorDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor):
        return [[self.linear(x).squeeze(-1)]]


class RawTensorDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class FeatureDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor):
        feature = x.unsqueeze(1)
        logits = self.linear(x).squeeze(-1)
        return [[feature, logits]]


class LogitOnlyOrFeatureDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        logits = x.mean(dim=-1)
        if x.mean().item() < 0:
            return [[logits]]
        return [[x.unsqueeze(1), logits]]


class VariableBranchDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        if x.mean().item() < 0:
            return [[x]]
        return [[x], [x]]


class LossTest(unittest.TestCase):
    def test_audio_preset_is_lazy_on_root_import(self):
        self.assertNotIn("anytrain.framework.gan.audio", sys.modules)

    def test_observation_registry_declares_outputs_events_and_state(self):
        loss_fn = Loss(
            IdentityDiscriminator(),
            feature_weight=0.5,
            gp_weight=2.0,
        )
        specs = observation.registry.specs(loss_fn)

        self.assertEqual(
            [(type(spec), spec.name) for spec in specs],
            [
                (observation.OutputObservation, "discriminator/loss"),
                (observation.ForwardEvent, "discriminator/components"),
                (observation.ForwardEvent, "discriminator/gradient_penalty"),
                (observation.OutputObservation, "generator/loss"),
                (observation.ForwardEvent, "generator/components"),
                (observation.ForwardEvent, "generator/feature_matching"),
                (observation.StateObservation, "weights"),
            ],
        )
        discriminator_output = specs[0]
        generator_output = specs[3]
        self.assertIsInstance(discriminator_output, observation.OutputObservation)
        self.assertIsInstance(generator_output, observation.OutputObservation)
        assert isinstance(discriminator_output, observation.OutputObservation)
        assert isinstance(generator_output, observation.OutputObservation)
        discriminator_curve = discriminator_output.collect(
            loss_fn,
            (),
            {"objective": "discriminator"},
            torch.tensor(1.0),
        )
        generator_curve = generator_output.collect(
            loss_fn,
            (),
            {"objective": "generator"},
            torch.tensor(2.0),
        )
        self.assertIsInstance(discriminator_curve, observation.Curve)
        self.assertIsInstance(generator_curve, observation.Curve)
        self.assertIsNone(
            discriminator_output.collect(
                loss_fn,
                (),
                {"objective": "generator"},
                torch.tensor(2.0),
            )
        )
        state = specs[-1]
        self.assertIsInstance(state, observation.StateObservation)
        assert isinstance(state, observation.StateObservation)
        weights = state.collect(loss_fn)
        self.assertIsInstance(weights, dict)
        assert isinstance(weights, dict)
        torch.testing.assert_close(weights["feature_weight"].value, torch.tensor(0.5))
        torch.testing.assert_close(weights["gp_weight"].value, torch.tensor(2.0))

    def test_compatibility_objective_methods_run_forward_hooks(self):
        loss_fn = Loss(IdentityDiscriminator())
        objectives = []

        def capture(module, inputs, kwargs, output):
            del module, inputs, output
            objectives.append(kwargs["objective"])

        handle = loss_fn.register_forward_hook(capture, with_kwargs=True)
        try:
            loss_fn.discriminator_loss(torch.zeros(2), torch.ones(2))
            loss_fn.generator_loss(torch.zeros(2))
        finally:
            handle.remove()

        self.assertEqual(objectives, ["discriminator", "generator"])

    def test_hinge_values(self):
        loss_fn = Loss(IdentityDiscriminator(), gan=GAN.Hinge)
        real = torch.tensor([2.0, 0.0])
        fake = torch.tensor([-2.0, 0.0])

        d_loss, d_diagnostics, _ = _run_with_observations(
            lambda: loss_fn.discriminator_loss(fake, real)
        )
        g_loss, g_diagnostics, _ = _run_with_observations(lambda: loss_fn.generator_loss(fake))

        self.assertTrue(torch.isclose(d_loss, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(d_diagnostics["real"], torch.tensor(0.5)))
        self.assertTrue(torch.isclose(d_diagnostics["fake"], torch.tensor(0.5)))
        self.assertTrue(torch.isclose(g_loss, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(g_diagnostics["adv"], torch.tensor(1.0)))

    def test_lsgan_and_wgan_values(self):
        logits = torch.tensor([-1.0, 1.0])

        lsgan = Loss(IdentityDiscriminator(), gan="lsgan")
        d_loss, d_diagnostics, _ = _run_with_observations(
            lambda: lsgan.discriminator_loss(logits, logits)
        )
        g_loss, g_diagnostics, _ = _run_with_observations(lambda: lsgan.generator_loss(logits))
        self.assertTrue(torch.isclose(d_loss, torch.tensor(3.0)))
        self.assertTrue(torch.isclose(d_diagnostics["fake"], torch.tensor(1.0)))
        self.assertTrue(torch.isclose(d_diagnostics["real"], torch.tensor(2.0)))
        self.assertTrue(torch.isclose(g_loss, torch.tensor(2.0)))
        self.assertTrue(torch.isclose(g_diagnostics["adv"], torch.tensor(2.0)))

        wgan = Loss(IdentityDiscriminator(), gan="wgan", gp_weight=0.0)
        d_loss, d_diagnostics, _ = _run_with_observations(
            lambda: wgan.discriminator_loss(logits, logits)
        )
        g_loss, g_diagnostics, _ = _run_with_observations(lambda: wgan.generator_loss(logits))
        self.assertTrue(torch.isclose(d_loss, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(d_diagnostics["fake"], torch.tensor(0.0)))
        self.assertTrue(torch.isclose(d_diagnostics["real"], torch.tensor(0.0)))
        self.assertTrue(torch.isclose(g_loss, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(g_diagnostics["adv"], torch.tensor(0.0)))

    def test_default_reduction_is_mean(self):
        loss_fn = Loss(ScaleBranchDiscriminator(), gan=GAN.LSGAN)
        fake = torch.zeros(2)
        real = torch.ones(2)

        d_loss, diagnostics, _ = _run_with_observations(
            lambda: loss_fn.discriminator_loss(fake, real)
        )

        self.assertTrue(torch.isclose(d_loss, torch.tensor(0.5)))
        self.assertTrue(torch.isclose(diagnostics["real"], torch.tensor(0.5)))
        self.assertTrue(torch.isclose(diagnostics["fake"], torch.tensor(0.0)))

    def test_sum_reduction(self):
        loss_fn = Loss(ScaleBranchDiscriminator(), gan=GAN.LSGAN, reduction=Reduction.Sum)
        fake = torch.zeros(2)
        real = torch.ones(2)

        d_loss, diagnostics, _ = _run_with_observations(
            lambda: loss_fn.discriminator_loss(fake, real)
        )

        self.assertTrue(torch.isclose(d_loss, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(diagnostics["real"], torch.tensor(1.0)))
        self.assertTrue(torch.isclose(diagnostics["fake"], torch.tensor(0.0)))

    def test_raw_tensor_output_is_rejected(self):
        loss_fn = Loss(RawTensorDiscriminator())
        fake = torch.randn(2, 4)
        real = torch.randn(2, 4)

        with self.assertRaisesRegex(TypeError, r"\[\[logits\]\]"):
            loss_fn.discriminator_loss(fake, real)

    def test_discriminator_loss_rejects_mismatched_branch_count(self):
        loss_fn = Loss(VariableBranchDiscriminator())

        with self.assertRaisesRegex(ValueError, "same branch count"):
            loss_fn.discriminator_loss(-torch.ones(2), torch.ones(2))

    def test_discriminator_loss_detaches_fake(self):
        discriminator = TensorDiscriminator()
        loss_fn = Loss(discriminator)
        fake = torch.randn(3, 4, requires_grad=True)
        real = torch.randn(3, 4)

        loss, diagnostics, _ = _run_with_observations(
            lambda: loss_fn.discriminator_loss(fake, real)
        )
        loss.backward()

        self.assertIn("real", diagnostics)
        self.assertIn("fake", diagnostics)
        self.assertIsNone(fake.grad)
        self.assertIsNotNone(discriminator.linear.weight.grad)
        self.assertFalse(diagnostics["real"].requires_grad)

    def test_generator_loss_keeps_fake_grad(self):
        discriminator = TensorDiscriminator()
        loss_fn = Loss(discriminator)
        fake = torch.randn(3, 4, requires_grad=True)

        loss, diagnostics, _ = _run_with_observations(lambda: loss_fn.generator_loss(fake))
        loss.backward()

        self.assertIn("adv", diagnostics)
        self.assertIsNotNone(fake.grad)

    def test_generator_feature_matching_requires_real(self):
        loss_fn = Loss(FeatureDiscriminator(), feature_weight=1.0)
        fake = torch.randn(2, 4)

        with self.assertRaisesRegex(ValueError, "real"):
            loss_fn.generator_loss(fake)

    def test_generator_feature_matching_requires_features(self):
        loss_fn = Loss(TensorDiscriminator(), feature_weight=1.0)
        fake = torch.randn(2, 4)
        real = torch.randn(2, 4)

        with self.assertRaisesRegex(ValueError, "feature maps"):
            loss_fn.generator_loss(fake, real)

    def test_generator_feature_matching_adds_weighted_feature_and_detaches_real(self):
        loss_fn = Loss(FeatureDiscriminator(), feature_weight=0.5)
        fake = torch.randn(2, 4, requires_grad=True)
        real = torch.randn(2, 4, requires_grad=True)

        loss, diagnostics, _ = _run_with_observations(lambda: loss_fn.generator_loss(fake, real))
        loss.backward()

        self.assertTrue(loss.requires_grad)
        self.assertIn("feature", diagnostics)
        torch.testing.assert_close(
            loss,
            diagnostics["adv"] + 0.5 * diagnostics["feature"],
        )
        self.assertIsNotNone(fake.grad)
        self.assertIsNone(real.grad)

    def test_feature_matching_rejects_mismatched_feature_count(self):
        loss_fn = Loss(LogitOnlyOrFeatureDiscriminator(), feature_weight=1.0)
        fake = -torch.ones(2, 4)
        real = torch.ones(2, 4)

        with self.assertRaisesRegex(ValueError, "feature count"):
            loss_fn.generator_loss(fake, real)

    def test_wgan_uses_default_gradient_penalty(self):
        discriminator = TensorDiscriminator()
        loss_fn = Loss(discriminator, gan=GAN.WGAN)
        fake = torch.randn(3, 4)
        real = torch.randn(3, 4)

        loss, diagnostics, _ = _run_with_observations(
            lambda: loss_fn.discriminator_loss(fake, real)
        )
        loss.backward()

        self.assertIn("gp", diagnostics)
        self.assertEqual(diagnostics["gp"].ndim, 0)
        self.assertIsNotNone(discriminator.linear.weight.grad)

    def test_non_wgan_does_not_use_default_gradient_penalty(self):
        loss_fn = Loss(TensorDiscriminator(), gan=GAN.Hinge)
        fake = torch.randn(3, 4)
        real = torch.randn(3, 4)

        _, diagnostics, _ = _run_with_observations(lambda: loss_fn.discriminator_loss(fake, real))

        self.assertNotIn("gp", diagnostics)

    def test_invalid_reduction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid Reduction"):
            Loss(TensorDiscriminator(), reduction="none")

    def test_from_preset_dac_returns_loss_with_feature_discriminator(self):
        dac_kwargs = {
            "in_channels": 1,
            "periods": (2,),
            "n_ffts": (16,),
            "bands": (0.25, 0.5, 0.75),
            "mpd_dim": 2,
            "mrd_dim": 2,
        }
        loss_fn = Loss.from_preset(
            Preset.DAC,
            feature_weight=0.1,
            **dac_kwargs,
        )
        fake = torch.randn(2, 1, 64, requires_grad=True)
        real = torch.randn(2, 1, 64)

        d_loss, d_diagnostics, _ = _run_with_observations(
            lambda: loss_fn.discriminator_loss(fake, real)
        )
        g_loss, g_diagnostics, _ = _run_with_observations(
            lambda: loss_fn.generator_loss(fake, real)
        )
        (d_loss + g_loss).backward()

        self.assertIn("real", d_diagnostics)
        self.assertIn("fake", d_diagnostics)
        self.assertIn("adv", g_diagnostics)
        self.assertIn("feature", g_diagnostics)
        self.assertIsNotNone(fake.grad)

    def test_dac_discriminator_output_contract(self):
        from anytrain.framework.gan.audio import DACDiscriminator

        discriminator = DACDiscriminator(
            in_channels=1,
            periods=(2,),
            n_ffts=(16,),
            bands=(0.25, 0.5, 0.75),
            mpd_dim=2,
            mrd_dim=2,
        )
        output = discriminator(torch.randn(2, 1, 64))

        self.assertEqual(len(output), 2)
        self.assertTrue(all(len(branch) > 1 for branch in output))
        self.assertTrue(
            all(isinstance(tensor, torch.Tensor) for branch in output for tensor in branch)
        )

    def test_dac_discriminator_reuses_registered_stft_window(self):
        from anytrain.framework.gan.audio import DACDiscriminator

        discriminator = DACDiscriminator(
            in_channels=1,
            periods=(),
            n_ffts=(16,),
            bands=(0.25, 0.5, 0.75),
            mrd_dim=2,
        )
        mrd = discriminator.layers[0]

        self.assertIn("window", dict(mrd.named_buffers()))
        self.assertNotIn("layers.0.window", discriminator.state_dict())
        with mock.patch("torch.hann_window", side_effect=AssertionError("rebuilt window")):
            output = discriminator(torch.randn(2, 1, 64))

        self.assertEqual(len(output), 1)

    def test_from_preset_requires_preset_enum(self):
        with self.assertRaisesRegex(TypeError, "Preset"):
            Loss.from_preset("dac")


if __name__ == "__main__":
    unittest.main()
