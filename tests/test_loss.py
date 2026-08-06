import importlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from anytrain import observation
from anytrain.loss import (
    FixedWeightLossBalancer,
    LossABC,
    LossBalancerABC,
    LossGroup,
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
    PackedCodebookLogits,
    TeacherLoss,
    UncertaintyLossBalancer,
    teacher_loss,
)
from anytrain.loss.codec import CodecLoss, CodecLossPreset
from anytrain.loss.spectral import (
    CompressedSpectrogramLoss,
    LogMagnitudeLoss,
    MelLoss,
    MelSpectrogramTransform,
    MultiScaleMelLoss,
    MultiScaleSTFTLoss,
    SpectralRMSELoss,
    STFTLoss,
    STFTTransform,
)
from anytrain.loss.temporal import SDRLoss, global_sdr


class ScalarLoss(LossABC):
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        return (prediction - target).abs().mean()


class CountingLoss(ScalarLoss):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.batch_sizes: list[int] = []

    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        self.call_count += 1
        self.batch_sizes.append(prediction.size(0))
        return super().compute_loss(prediction, target)


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


class ForwardSumLossBalancer(LossBalancerABC):
    def forward(self, losses):
        return sum(losses.values())


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

    def test_loss_group_accepts_forward_style_custom_balancer(self):
        loss = LossGroup(
            {"l1": nn.L1Loss(), "l2": nn.MSELoss()},
            balancer=ForwardSumLossBalancer(),
        )

        total = loss(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

        torch.testing.assert_close(total, torch.tensor(4.0))

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

    def test_fixed_weight_loss_balancer_rejects_nonfinite_weights(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                FixedWeightLossBalancer({"loss": value})

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

    def test_uncertainty_loss_balancer_rejects_name_mismatch(self):
        balancer = UncertaintyLossBalancer(["l1"])

        with self.assertRaisesRegex(ValueError, "missing"):
            balancer({})

        with self.assertRaisesRegex(ValueError, "unknown"):
            balancer({"l1": torch.tensor(1.0), "l2": torch.tensor(2.0)})

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
        fixed_specs = observation.registry.specs(FixedWeightLossBalancer({"l1": 1.0}))

        self.assertEqual([spec.name for spec in group_specs], ["loss"])
        self.assertEqual([spec.name for spec in frame_specs], ["loss", "diagnostics"])
        self.assertEqual([spec.name for spec in fixed_specs], ["weights"])
        self.assertIsInstance(group_specs[0], observation.OutputObservation)
        self.assertIsInstance(frame_specs[1], observation.ForwardEvent)
        self.assertIsInstance(fixed_specs[0], observation.StateObservation)

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
            for value, classes in zip(predictions, (3, 4))
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
        direct = MaskedCodebookCrossEntropyLoss().forward_packed(
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
            packed_loss = loss_fn.forward_packed(packed)
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

    def test_packed_codebook_compatibility_metadata_does_not_change_reduction(self):
        logits = (torch.randn(3, 4),)
        labels = torch.tensor([[0], [1], [2]])
        plain = PackedCodebookLogits(logits, labels)
        with_rows = PackedCodebookLogits(
            logits,
            labels,
            row_indices=torch.tensor([0, 1, 1]),
            batch_size=2,
        )
        loss_fn = MaskedCodebookCrossEntropyLoss()

        torch.testing.assert_close(
            loss_fn.forward_packed(plain),
            loss_fn.forward_packed(with_rows),
        )

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

    def test_stft_and_mel_losses(self):
        estimate = torch.randn(2, 1, 64, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(2, 1, 64)

        stft_loss = STFTLoss(n_fft=16)(estimate, reference)
        mel_loss = MelLoss(sample_rate=16000, n_fft=32, n_mels=8)(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)

    def test_spectral_transform_backend_selection(self):
        auto_stft = STFTTransform(n_fft=16)
        auto_mel = MelSpectrogramTransform(sample_rate=16000)
        torch_stft = STFTTransform(n_fft=16, backend="torch")
        torch_mel = MelSpectrogramTransform(
            sample_rate=16000,
            n_fft=16,
            n_mels=4,
            backend="torch",
        )

        self.assertIn(auto_stft.backend, {"torchaudio", "torch"})
        self.assertIn(auto_mel.backend, {"torchaudio", "torch"})
        self.assertEqual(torch_stft.backend, "torch")
        self.assertEqual(torch_mel.backend, "torch")

    def test_spectral_auto_backend_falls_back_when_torchaudio_binary_fails(self):
        module_name = "anytrain.loss.spectral.transform"
        spectral_package = sys.modules["anytrain.loss.spectral"]
        original_module = sys.modules.pop(module_name, None)
        original_package_transform = getattr(spectral_package, "transform", None)
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torchaudio.transforms" or name.startswith("torchaudio.transforms."):
                raise OSError("simulated libtorchaudio load failure")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                transform = importlib.import_module(module_name)

            self.assertEqual(transform.STFTTransform(n_fft=16, backend="auto").backend, "torch")
            self.assertEqual(
                transform.MelSpectrogramTransform(
                    sample_rate=16000,
                    n_fft=16,
                    n_mels=8,
                    backend="auto",
                ).backend,
                "torch",
            )
            with self.assertRaises(ImportError) as caught:
                transform.STFTTransform(n_fft=16, backend="torchaudio")
            self.assertIsInstance(caught.exception.__cause__, OSError)
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module
                spectral_package.transform = original_module
            elif original_package_transform is not None:
                spectral_package.transform = original_package_transform

    def test_stft_cache_key_uses_shared_default_hop_length(self):
        torch_stft = STFTTransform(n_fft=64, backend="torch")
        auto_stft = STFTTransform(n_fft=64)

        self.assertEqual(torch_stft.hop_length, 16)
        self.assertEqual(auto_stft.hop_length, 16)
        self.assertEqual(torch_stft.cache_key[2], 16)
        self.assertEqual(auto_stft.cache_key[2], 16)

    def test_mel_transform_supports_mel_scale(self):
        htk_mel = MelSpectrogramTransform(
            sample_rate=16000,
            n_fft=32,
            n_mels=8,
            mel_scale="htk",
            backend="torch",
        )
        slaney_mel = MelSpectrogramTransform(
            sample_rate=16000,
            n_fft=32,
            n_mels=8,
            mel_scale="slaney",
            backend="torch",
        )

        self.assertEqual(htk_mel.mel_scale, "htk")
        self.assertEqual(slaney_mel.mel_scale, "slaney")
        self.assertFalse(torch.allclose(htk_mel.mel_filter, slaney_mel.mel_filter))

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

    def test_mel_transform_rejects_unknown_mel_scale(self):
        with self.assertRaisesRegex(ValueError, "mel_scale"):
            MelSpectrogramTransform(sample_rate=16000, mel_scale="unknown")

    def test_spectral_loss_torch_backend_fallback(self):
        estimate = torch.randn(2, 1, 64, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(2, 1, 64)

        stft_loss = STFTLoss(n_fft=16, backend="torch")(estimate, reference)
        mel_loss = MelLoss(
            sample_rate=16000,
            n_fft=32,
            n_mels=8,
            backend="torch",
        )(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)

    def test_multi_scale_spectral_losses(self):
        estimate = torch.randn(2, 1, 64, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(2, 1, 64)

        stft_loss = MultiScaleSTFTLoss(n_fft_list=(16, 8))(estimate, reference)
        mel_loss = MultiScaleMelLoss(
            sample_rate=16000,
            n_fft_list=(32, 16),
            n_mels_list=(8, 4),
        )(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)

    def test_codec_loss_reuses_overlapping_stft_transforms_per_forward(self):
        estimate = torch.randn(1, 1, 128)
        reference = torch.randn_like(estimate)
        loss = CodecLoss.from_preset(
            "dynacodec",
            sample_rate=16000,
            mel_n_fft_list=(64, 32),
            mel_n_mels_list=(8, 4),
            stft_n_fft_list=(64, 32),
            backend="auto",
        )

        with patch("torch.stft", wraps=torch.stft) as stft:
            loss(estimate, reference)
            self.assertEqual(stft.call_count, 4)
            loss(estimate, reference)
            self.assertEqual(stft.call_count, 8)

    def test_codec_loss_spectral_cache_preserves_values_and_gradients(self):
        torch.manual_seed(7)
        estimate = torch.randn(1, 1, 64, requires_grad=True)
        reference = torch.randn_like(estimate)
        cached_loss = CodecLoss.from_preset(
            "dynacodec",
            sample_rate=16000,
            mel_n_fft_list=(16, 8),
            mel_n_mels_list=(8, 4),
            stft_n_fft_list=(16, 8),
            backend="torch",
        )
        cached_value = cached_loss(estimate, reference)

        expected_estimate = estimate.detach().clone().requires_grad_()
        expected_reference = reference.detach().clone()
        expected_mel = MultiScaleMelLoss(
            sample_rate=16000,
            n_fft_list=(16, 8),
            n_mels_list=(8, 4),
            backend="torch",
        )
        expected_stft = MultiScaleSTFTLoss(n_fft_list=(16, 8), backend="torch")
        expected_sdr = SDRLoss()
        mel_value = expected_mel(expected_estimate, expected_reference)
        stft_value = expected_stft(expected_estimate, expected_reference)
        expected_value = expected_sdr(expected_estimate, expected_reference)
        expected_value = expected_value + 10 * mel_value + 10 * stft_value

        self.assertTrue(torch.allclose(cached_value, expected_value, rtol=1e-5, atol=1e-6))
        cached_value.backward()
        expected_value.backward()
        self.assertTrue(torch.allclose(estimate.grad, expected_estimate.grad, rtol=1e-5, atol=2e-5))

    def test_codec_loss_presets(self):
        dac_loss = CodecLoss.from_preset(CodecLossPreset.DAC, backend="torch")
        dynacodec_loss = CodecLoss.from_preset("dynacodec", backend="torch")

        self.assertEqual(set(dac_loss.losses), {"multi_mel"})
        self.assertEqual(set(dynacodec_loss.losses), {"si_sdr", "multi_mel", "multi_stft"})
        self.assertIn("si_sdr", dynacodec_loss.balancer.loss_names)

    def test_codec_loss_task_import_is_not_provided(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("anytrain.loss.task.codec")

    def test_codec_loss_preset_returns_scalar(self):
        estimate = torch.randn(1, 1, 4096, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(1, 1, 4096)
        loss = CodecLoss.from_preset("dynacodec", backend="torch")

        total = loss(estimate, reference)

        self.assertEqual(total.ndim, 0)

    def test_codec_loss_supports_lengths(self):
        estimate = torch.tensor(
            [
                [[1.0, 2.0, 100.0, 100.0]],
                [[1.0, 2.0, 3.0, 4.0]],
            ],
            requires_grad=True,
        )
        reference = torch.zeros_like(estimate)
        loss = CodecLoss({"l1": nn.L1Loss()})

        total = loss(estimate, reference, lengths=torch.tensor([2, 4]))

        self.assertTrue(torch.isclose(total, torch.tensor(2.0)))
        self.assertEqual(total.ndim, 0)
        total.backward()
        self.assertEqual(float(estimate.grad[0, 0, 2]), 0.0)

    def test_codec_loss_groups_repeated_lengths(self):
        estimate = torch.randn(4, 1, 4)
        reference = torch.zeros_like(estimate)
        counting_loss = CountingLoss()
        loss = CodecLoss({"counting": counting_loss})

        total = loss(estimate, reference, lengths=[2, 4, 2, 4])

        self.assertEqual(counting_loss.call_count, 2)
        self.assertEqual(counting_loss.batch_sizes, [2, 2])
        self.assertEqual(total.ndim, 0)

    def test_codec_loss_weights_unequal_length_groups_by_sample_count(self):
        estimate = torch.tensor(
            [
                [[1.0, 1.0, 100.0, 100.0]],
                [[3.0, 3.0, 100.0, 100.0]],
                [[5.0, 5.0, 5.0, 5.0]],
            ]
        )
        reference = torch.zeros_like(estimate)
        loss = CodecLoss({"mean": nn.L1Loss(reduction="mean")})

        total = loss(estimate, reference, lengths=[2, 2, 4])

        torch.testing.assert_close(total, torch.tensor(3.0))

    def test_codec_cache_routes_every_registered_loss_through_module_call(self):
        loss = CodecLoss.from_preset(
            "dynacodec",
            sample_rate=16000,
            mel_n_fft_list=(16,),
            mel_n_mels_list=(4,),
            stft_n_fft_list=(16,),
            backend="torch",
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

    def test_codec_loss_rejects_invalid_lengths(self):
        loss = CodecLoss({"l1": nn.L1Loss()})
        audio = torch.zeros(2, 1, 4)

        with self.assertRaisesRegex(ValueError, "one value per batch"):
            loss(audio, audio, lengths=[4])

        with self.assertRaisesRegex(ValueError, "positive"):
            loss(audio, audio, lengths=[4, 0])

        with self.assertRaisesRegex(ValueError, "exceed"):
            loss(audio, audio, lengths=[4, 5])

    def test_codec_loss_rejects_non_integer_tensor_lengths(self):
        loss = CodecLoss({"l1": nn.L1Loss()})
        audio = torch.zeros(2, 1, 4)

        invalid_lengths = (
            torch.tensor([4.0, 4.0]),
            torch.tensor([True, True]),
            torch.tensor([4 + 0j, 4 + 0j]),
        )
        for lengths in invalid_lengths:
            with (
                self.subTest(dtype=lengths.dtype),
                self.assertRaisesRegex(TypeError, "integer dtype"),
            ):
                loss(audio, audio, lengths=lengths)

    def test_codec_loss_rejects_non_index_sequence_lengths(self):
        loss = CodecLoss({"l1": nn.L1Loss()})
        audio = torch.zeros(2, 1, 4)

        for lengths in ([4.0, 4.0], [True, True], [4 + 0j, 4 + 0j], ["4", "4"]):
            with (
                self.subTest(lengths=lengths),
                self.assertRaisesRegex(TypeError, "index-compatible integers"),
            ):
                loss(audio, audio, lengths=lengths)

    def test_codec_loss_rejects_bool_like_sequence_lengths(self):
        loss = CodecLoss({"l1": nn.L1Loss()})
        audio = torch.zeros(2, 1, 4)

        bool_values = (torch.tensor(True), np.bool_(True), np.array(True))
        for value in bool_values:
            with (
                self.subTest(value_type=type(value)),
                self.assertRaisesRegex(TypeError, "not bool"),
            ):
                loss(audio, audio, lengths=[value, value])

    def test_codec_loss_rejects_non_sequence_length_containers(self):
        loss = CodecLoss({"l1": nn.L1Loss()})
        audio = torch.zeros(2, 1, 4)

        invalid = ({0: 4, 1: 4}, iter([4, 4]), b"44")
        for lengths in invalid:
            with (
                self.subTest(lengths_type=type(lengths)),
                self.assertRaisesRegex(TypeError, "non-text sequence"),
            ):
                loss(audio, audio, lengths=lengths)

    def test_codec_loss_rejects_unknown_preset(self):
        with self.assertRaisesRegex(ValueError, "Unknown codec loss preset"):
            CodecLoss.from_preset("unknown")


if __name__ == "__main__":
    unittest.main()
