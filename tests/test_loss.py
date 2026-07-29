import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from anytrain.loss import (
    FixedWeightLossBalancer,
    LossABC,
    LossBalancerABC,
    LossGroup,
    LossItem,
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
    UncertaintyLossBalancer,
    combine_loss_outputs,
    iter_loss_items,
)
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
from anytrain.loss.task import CodecLoss, CodecLossPreset
from anytrain.loss.temporal import SDRLoss


class DetailLoss(LossABC):
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = (prediction - target).abs().mean()
        return loss, {"raw": loss}


class CountingDetailLoss(DetailLoss):
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


class InvalidDetailLoss(LossABC):
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = (prediction - target).abs().mean()
        return loss, {"count": 1}


class SumLossBalancer(LossBalancerABC):
    def forward(self, losses):
        return sum(losses.values())


class DetailLossBalancer(LossBalancerABC):
    def forward(self, losses):
        total = sum(losses.values())
        return total, {"scale": torch.tensor(float(len(losses)), device=total.device)}


class LossTest(unittest.TestCase):
    def test_loss_group(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup({"l1": nn.L1Loss(), "l2": nn.MSELoss()})

        total, details = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(2.0)))
        self.assertTrue(total.requires_grad)
        self.assertEqual(set(details), {"l1", "l2"})
        self.assertFalse(details["l1"].requires_grad)
        self.assertIsInstance(loss.losses, nn.ModuleDict)

    def test_loss_group_accepts_custom_balancer(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup({"l1": nn.L1Loss(), "l2": nn.MSELoss()}, balancer=SumLossBalancer())

        total, _ = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(4.0)))

    def test_fixed_weight_loss_balancer(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup(
            {"l1": nn.L1Loss(), "l2": nn.MSELoss()},
            balancer=FixedWeightLossBalancer({"l1": 1.0, "l2": 0.5}),
        )

        total, details = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(2.75)))
        self.assertIn("balancer/l1_weight", details)
        self.assertIn("balancer/l2_weight", details)

    def test_fixed_weight_loss_balancer_rejects_nonfinite_weights(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                FixedWeightLossBalancer({"loss": value})

    def test_loss_group_flattens_balancer_details(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup({"l1": nn.L1Loss(), "l2": nn.MSELoss()}, balancer=DetailLossBalancer())

        _, details = loss(pred, target)

        self.assertIn("balancer/scale", details)
        self.assertFalse(details["balancer/scale"].requires_grad)

    def test_uncertainty_loss_balancer(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = LossGroup(
            {"l1": nn.L1Loss(), "l2": nn.MSELoss()},
            balancer=UncertaintyLossBalancer(["l1", "l2"]),
        )

        total, details = loss(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(2.0)))
        self.assertTrue(total.requires_grad)
        self.assertIn("balancer/l1_uncertainty_weight", details)
        self.assertIn("balancer/l2_uncertainty_weight", details)

    def test_uncertainty_loss_balancer_rejects_name_mismatch(self):
        balancer = UncertaintyLossBalancer(["l1"])

        with self.assertRaisesRegex(ValueError, "missing"):
            balancer({})

        with self.assertRaisesRegex(ValueError, "unknown"):
            balancer({"l1": torch.tensor(1.0), "l2": torch.tensor(2.0)})

    def test_loss_abc_accepts_scalar_loss_with_optional_details(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss_fn = DetailLoss()

        loss, details = loss_fn(pred, target)

        self.assertTrue(torch.isclose(loss, torch.tensor(1.5)))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(set(details), {"raw"})
        self.assertFalse(details["raw"].requires_grad)

    def test_loss_abc_rejects_non_scalar_main_loss(self):
        loss_fn = VectorLoss()

        with self.assertRaisesRegex(ValueError, "scalar tensor"):
            loss_fn(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

    def test_loss_abc_rejects_non_scalar_detail_values(self):
        loss_fn = InvalidDetailLoss()

        with self.assertRaisesRegex(TypeError, "float or 0-d tensor"):
            loss_fn(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

    def test_loss_group_flattens_child_details(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss_fn = LossGroup({"detail": DetailLoss()})

        total, details = loss_fn(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(1.5)))
        self.assertIn("detail/raw", details)
        self.assertFalse(details["detail/raw"].requires_grad)

    def test_loss_group_allows_total_as_loss_name(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss_fn = LossGroup({"total": nn.L1Loss()})

        total, details = loss_fn(pred, target)

        self.assertTrue(torch.isclose(total, torch.tensor(1.5)))
        self.assertEqual(set(details), {"total"})

    def test_loss_item_weighted_mean(self):
        item = LossItem(
            torch.tensor([1.0, 3.0]),
            {"tokens": torch.tensor([1.0, 3.0])},
        )

        mean = item.weighted_mean(item.details["tokens"])

        torch.testing.assert_close(mean, torch.tensor(2.5))

    def test_loss_item_weighted_mean_ignores_zero_weight_nonfinite_rows(self):
        item = LossItem(
            torch.tensor([1.0, float("nan")]),
            {"tokens": torch.tensor([2.0, 0.0])},
        )

        torch.testing.assert_close(item.weighted_mean(item.details["tokens"]), torch.tensor(1.0))

    def test_loss_item_weighted_mean_rejects_invalid_weights(self):
        item = LossItem(torch.tensor([1.0, 2.0]))
        for weight in (torch.tensor([1.0, float("nan")]), torch.tensor([1.0, -1.0])):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "finite"):
                item.weighted_mean(weight)

    def test_combine_loss_outputs_uses_units_and_weights(self):
        first = {
            "loss": torch.tensor(0.0),
            "token": LossItem(
                torch.tensor([1.0]),
                {"tokens": torch.tensor([2.0])},
            ),
            "acoustic": LossItem(
                torch.tensor([5.0]),
                {"frames": torch.tensor([1.0])},
            ),
            "loss_weights": {"acoustic": 0.5},
        }
        second = {
            "loss": torch.tensor(0.0),
            "token": LossItem(
                torch.tensor([3.0]),
                {"tokens": torch.tensor([6.0])},
            ),
        }

        outputs = combine_loss_outputs(
            [first, second],
            {"token": "tokens", "acoustic": "frames"},
        )

        self.assertEqual([name for name, _ in iter_loss_items(outputs, ["acoustic", "token"])], ["acoustic", "token"])
        torch.testing.assert_close(outputs["loss"], torch.tensor(5.0))
        torch.testing.assert_close(outputs["token"].loss, torch.tensor([1.0, 3.0]))

    def test_masked_frame_mse_loss_ignores_padding(self):
        prediction = torch.tensor([[[1.0, 2.0], [1000.0, float("nan")]]], requires_grad=True)
        target = torch.tensor([[[3.0, 4.0], [float("inf"), 1000.0]]])
        mask = torch.tensor([[True, False]])

        item = MaskedFrameMSELoss()(
            prediction,
            target,
            mask,
            details={"t": torch.tensor([0.25])},
        )
        item.loss.mean().backward()

        torch.testing.assert_close(item.loss, torch.tensor([4.0]))
        self.assertIsNotNone(item.details)
        torch.testing.assert_close(item.details["frames"], torch.tensor([1.0]))
        torch.testing.assert_close(item.details["t"], torch.tensor([0.25]))
        gradient = prediction.grad
        if gradient is None:
            self.fail("prediction gradient is unavailable")
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))

    def test_masked_cosine_alignment_loss_detaches_target(self):
        prediction = torch.tensor(
            [[[1.0, 0.0], [float("nan"), float("inf")]]],
            requires_grad=True,
        )
        target = torch.tensor([[[1.0, 0.0], [float("nan"), float("inf")]]], requires_grad=True)
        mask = torch.tensor([[True, False]])

        item = MaskedCosineAlignmentLoss()(prediction, target, mask)
        item.loss.mean().backward()

        torch.testing.assert_close(item.loss, torch.tensor([0.0]))
        self.assertIsNotNone(item.details)
        torch.testing.assert_close(item.details["frames"], torch.tensor([1.0]))
        torch.testing.assert_close(item.details["cosine"], torch.tensor([1.0]))
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)

    def test_masked_codebook_cross_entropy_reports_masked_top1(self):
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

        item = MaskedCodebookCrossEntropyLoss()(
            logits,
            labels,
            mask,
            include_top1=True,
        )
        details = item.details
        self.assertIsNotNone(details)
        torch.testing.assert_close(details["codebook_0_top1"], torch.tensor([0.5, 1.0]))
        torch.testing.assert_close(details["codebook_1_top1"], torch.tensor([1.0, 0.0]))

    def test_masked_codebook_cross_entropy_preserves_zero_frame_count(self):
        labels = torch.tensor([[[1]], [[-1]]])
        mask = torch.tensor([[True], [False]])
        logits = (torch.tensor([[[0.0, 1.0]], [[0.0, 0.0]]]),)

        item = MaskedCodebookCrossEntropyLoss()(
            logits,
            labels,
            mask,
            include_top1=True,
        )

        details = item.details
        self.assertIsNotNone(details)
        torch.testing.assert_close(item.loss[1], torch.tensor(0.0))
        torch.testing.assert_close(details["frames"], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(details["codebook_0_top1"], torch.tensor([1.0, 0.0]))

    def test_masked_codebook_cross_entropy_ignores_padding_logits(self):
        labels = torch.tensor([[[1], [0], [-1]]])
        mask = torch.tensor([[True, True, False]])
        logits = (torch.randn(1, 3, 3),)
        changed = (logits[0].clone(),)
        changed[0][~mask] = 1000

        baseline = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)
        padded = MaskedCodebookCrossEntropyLoss()(changed, labels, mask)
        torch.testing.assert_close(baseline.loss, padded.loss)

    def test_masked_codebook_cross_entropy_can_skip_details(self):
        labels = torch.tensor([[[1, 2], [0, 1], [-1, -1]]])
        mask = torch.tensor([[True, True, False]])
        logits = tuple(
            torch.randn(1, 3, classes, requires_grad=True)
            for classes in (3, 4)
        )

        with_details = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)
        without_details = MaskedCodebookCrossEntropyLoss()(
            logits,
            labels,
            mask,
            include_details=False,
            include_top1=True,
        )

        torch.testing.assert_close(without_details.loss, with_details.loss)
        self.assertIsNone(without_details.details)

    def test_masked_codebook_cross_entropy_validation_can_be_skipped(self):
        labels = torch.tensor([[[1], [0]]])
        mask = torch.tensor([[True, False]])
        logits = (torch.randn(1, 2, 3),)

        with patch(
            "anytrain.loss.codebook._validate_inputs",
            side_effect=AssertionError("validation should be skipped"),
        ):
            item = MaskedCodebookCrossEntropyLoss()(logits, labels, mask, validate=False)
        self.assertEqual(item.loss.shape, (1,))

    def test_masked_codebook_cross_entropy_validates_large_vocab_with_small_labels(self):
        labels = torch.tensor([[[127], [1]]], dtype=torch.int8)
        mask = torch.tensor([[True, True]])
        logits = (torch.randn(1, 2, 256),)

        item = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)

        self.assertEqual(item.loss.shape, (1,))

    def test_loss_group_rejects_ambiguous_names(self):

        with self.assertRaisesRegex(ValueError, "reserved"):
            LossGroup({"balancer": nn.L1Loss()})

        with self.assertRaisesRegex(TypeError, "LossBalancerABC"):
            LossGroup({"loss": nn.L1Loss()}, balancer=nn.Identity())

        with self.assertRaisesRegex(TypeError, "mapping"):
            LossGroup([nn.L1Loss()])

    def test_sdr_loss_is_near_zero_for_identical_audio(self):
        audio = torch.randn(2, 1, 64)

        loss = SDRLoss()(audio, audio)

        self.assertLess(float(loss), 0.0)

    def test_spectral_single_losses(self):
        estimate = torch.randn(2, 8, 4, dtype=torch.complex64)
        reference = estimate + 0.1 * torch.randn(2, 8, 4, dtype=torch.complex64)

        self.assertEqual(LogMagnitudeLoss()(estimate, reference).ndim, 0)
        self.assertEqual(CompressedSpectrogramLoss()(estimate, reference).ndim, 0)
        self.assertEqual(SpectralRMSELoss()(estimate, reference).ndim, 0)

    def test_stft_and_mel_losses(self):
        estimate = torch.randn(2, 1, 64, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(2, 1, 64)

        stft_loss, stft_details = STFTLoss(n_fft=16)(estimate, reference)
        mel_loss, mel_details = MelLoss(sample_rate=16000, n_fft=32, n_mels=8)(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)
        self.assertIn("log_magnitude", stft_details)
        self.assertIn("log_magnitude", mel_details)

    def test_spectral_transform_backend_selection(self):
        auto_stft = STFTTransform(n_fft=16)
        auto_mel = MelSpectrogramTransform(sample_rate=16000)
        torch_stft = STFTTransform(n_fft=16, backend="torch")
        torch_mel = MelSpectrogramTransform(
            sample_rate=16000,
            n_fft=16,
            backend="torch",
        )

        self.assertIn(auto_stft.backend, {"torchaudio", "torch"})
        self.assertIn(auto_mel.backend, {"torchaudio", "torch"})
        self.assertEqual(torch_stft.backend, "torch")
        self.assertEqual(torch_mel.backend, "torch")

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

        stft_loss, _ = STFTLoss(n_fft=16, backend="torch")(estimate, reference)
        mel_loss, _ = MelLoss(
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

        stft_loss, stft_details = MultiScaleSTFTLoss(n_fft_list=(16, 8))(estimate, reference)
        mel_loss, mel_details = MultiScaleMelLoss(
            sample_rate=16000,
            n_fft_list=(32, 16),
            n_mels_list=(8, 4),
        )(estimate, reference)

        self.assertEqual(stft_loss.ndim, 0)
        self.assertEqual(mel_loss.ndim, 0)
        self.assertTrue(any(key.startswith("stft_16") for key in stft_details))
        self.assertTrue(any(key.startswith("mel_32_8") for key in mel_details))

    def test_codec_loss_presets(self):
        dac_loss = CodecLoss.from_preset(CodecLossPreset.DAC, backend="torch")
        dynacodec_loss = CodecLoss.from_preset("dynacodec", backend="torch")

        self.assertEqual(set(dac_loss.losses), {"multi_mel"})
        self.assertEqual(set(dynacodec_loss.losses), {"si_sdr", "multi_mel", "multi_stft"})
        self.assertIn("si_sdr", dynacodec_loss.balancer.loss_names)

    def test_codec_loss_preset_allows_nested_loss_details(self):
        estimate = torch.randn(1, 1, 4096, requires_grad=True)
        reference = estimate.detach() + 0.01 * torch.randn(1, 1, 4096)
        loss = CodecLoss.from_preset("dynacodec", backend="torch")

        total, details = loss(estimate, reference)

        self.assertEqual(total.ndim, 0)
        self.assertIn("multi_mel/mel_2048_320/log_magnitude", details)
        self.assertIn("multi_stft/stft_2048/log_magnitude", details)

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

        total, details = loss(estimate, reference, lengths=torch.tensor([2, 4]))

        self.assertTrue(torch.isclose(total, torch.tensor(2.0)))
        self.assertTrue(torch.isclose(details["l1"], torch.tensor(2.0)))
        total.backward()
        self.assertEqual(float(estimate.grad[0, 0, 2]), 0.0)

    def test_codec_loss_groups_repeated_lengths(self):
        estimate = torch.randn(4, 1, 4)
        reference = torch.zeros_like(estimate)
        counting_loss = CountingDetailLoss()
        loss = CodecLoss({"counting": counting_loss})

        total, details = loss(estimate, reference, lengths=[2, 4, 2, 4])

        self.assertEqual(counting_loss.call_count, 2)
        self.assertEqual(counting_loss.batch_sizes, [2, 2])
        self.assertEqual(total.ndim, 0)
        self.assertIn("counting/raw", details)

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
