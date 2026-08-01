import math
import unittest

import torch

from anytrain.module.quantization import (
    QuantizationLoss,
    QuantizeOutput,
    codebook_usage,
    quantization_loss_metrics,
    quantization_usage,
    valid_indices_by_codebook,
)


class QuantizationStatsTest(unittest.TestCase):
    def test_codebook_usage_summarizes_single_codebook_indices(self):
        usage = codebook_usage(
            torch.tensor([[0, 1, 1], [3, 3, 3]]),
            codebook_size=4,
        )

        self.assertEqual(usage.counts.tolist(), [[1.0, 2.0, 0.0, 3.0]])
        self.assertEqual(usage.observations.tolist(), [6.0])
        self.assertEqual(usage.unique_count.tolist(), [3.0])
        self.assertEqual(usage.dead_count.tolist(), [1.0])
        self.assertEqual(usage.unique_ratio.tolist(), [0.75])
        self.assertEqual(usage.dead_ratio.tolist(), [0.25])
        self.assertEqual(usage.active_ratio.tolist(), [1.0])
        self.assertGreater(usage.perplexity.item(), 0.0)

    def test_codebook_usage_ignores_inactive_residual_indices(self):
        indices = torch.tensor(
            [
                [[0, 1, -1], [1, -1, -1]],
                [[2, 2, 0], [3, 2, -1]],
            ]
        )
        active = indices >= 0

        usage = codebook_usage(
            indices,
            codebook_size=4,
            num_codebooks=3,
            active_codebook_mask=active,
        )

        self.assertEqual(
            usage.counts.tolist(),
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(usage.observations.tolist(), [4.0, 3.0, 1.0])
        self.assertEqual(usage.active_count.tolist(), [4.0, 3.0, 1.0])
        self.assertEqual(usage.total_count.tolist(), [4.0, 4.0, 4.0])
        torch.testing.assert_close(
            usage.active_ratio,
            torch.tensor([1.0, 0.75, 0.25]),
        )

    def test_quantization_usage_reads_output_mask(self):
        output = QuantizeOutput(
            quantized_latents=torch.empty(2, 3),
            indices=torch.tensor([[0, 1], [2, -1]]),
            active_codebook_mask=torch.tensor([[True, True], [True, False]]),
        )

        usage = quantization_usage(output, codebook_size=3, num_codebooks=2)

        self.assertEqual(usage.counts.tolist(), [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    def test_quantization_loss_metrics_skips_missing_loss(self):
        output = QuantizeOutput(
            quantized_latents=torch.empty(1),
            indices=torch.tensor([0]),
            loss=QuantizationLoss(
                commitment=torch.tensor(0.25),
                codebook=torch.tensor(0.75),
            ),
        )

        metrics = quantization_loss_metrics(output)

        self.assertEqual(set(metrics), {"commitment", "codebook", "total"})
        self.assertEqual(metrics["total"].item(), 1.0)
        self.assertEqual(
            quantization_loss_metrics(
                QuantizeOutput(quantized_latents=torch.empty(1), indices=torch.tensor([0]))
            ),
            {},
        )

    def test_valid_indices_by_codebook_returns_only_active_assignments(self):
        values = valid_indices_by_codebook(
            torch.tensor([[0, -1], [1, 2]]),
            codebook_size=3,
            num_codebooks=2,
            active_codebook_mask=torch.tensor([[True, False], [True, True]]),
        )

        self.assertEqual([value.tolist() for value in values], [[0, 1], [2]])

    def test_invalid_indices_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            codebook_usage(torch.tensor([0, 4]), codebook_size=4)
        with self.assertRaisesRegex(TypeError, "integer"):
            codebook_usage(torch.tensor([0.0]), codebook_size=4)

    def test_empty_usage_returns_zero_metrics(self):
        usage = codebook_usage(torch.empty(0, dtype=torch.long), codebook_size=4)

        self.assertEqual(usage.counts.tolist(), [[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(usage.observations.tolist(), [0.0])
        self.assertEqual(usage.active_ratio.tolist(), [0.0])
        self.assertEqual(usage.perplexity.tolist(), [0.0])
        self.assertFalse(math.isnan(usage.entropy.item()))


if __name__ == "__main__":
    unittest.main()
