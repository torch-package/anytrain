import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import torch

from anytrain.module.quantization import (
    AGRVQConfig,
    AutoGroupResidualVectorQuantizer,
    QuantizationLoss,
)


class AutoGroupResidualVectorQuantizerTest(unittest.TestCase):
    def test_forward_shapes_and_flat_codebook_size(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=8, num_codebooks=2, codebook_size=5, codebook_dim=4)
        )
        x = torch.randn(2, 3, 8)

        output = quantizer(x)

        self.assertEqual(quantizer.group_size, 5)
        self.assertEqual(quantizer.codebook_size, 25)
        self.assertEqual(quantizer.codebook_dim, 8)
        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (2, 3, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 3, 2, 8))
        self.assertEqual(output.latents.shape, (2, 3, 2, 8))
        self.assertEqual(output.active_codebook_mask.shape, (2, 3, 2))
        self.assertIsInstance(output.loss, QuantizationLoss)
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue((output.indices < 25).all())

    def test_num_active_codebooks_controls_output_width(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=8, num_codebooks=4, codebook_size=5)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8), num_active_codebooks=2)

        self.assertEqual(output.indices.shape, (2, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 2, 16))
        self.assertTrue(output.active_codebook_mask.all())

    def test_group_indices_round_trip(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, codebook_size=5, codebook_dim=2)
        )
        group_indices = torch.tensor([[[0, 0], [2, 4]], [[1, 3], [4, 2]]])

        indices = quantizer.group_indices_to_indices(group_indices)
        round_trip = quantizer.indices_to_group_indices(indices)

        self.assertTrue(torch.equal(indices, torch.tensor([[0, 14], [8, 22]])))
        self.assertTrue(torch.equal(round_trip, group_indices))

    def test_indices_round_trip_through_codebook_vectors(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, codebook_size=3, codebook_dim=2)
        )
        with torch.no_grad():
            for book in quantizer.quantizers:
                book.codebook_a.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
                book.codebook_b.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]))
        indices = torch.tensor([[0, 1], [5, 8]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_project_codebook_vectors_sums_projected_books(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, codebook_size=3, codebook_dim=2)
        )
        indices = torch.tensor([[0, 1], [5, 8]])
        vectors = quantizer.indices_to_codebook_vectors(indices)

        projected = quantizer.project_codebook_vectors(vectors)

        self.assertEqual(projected.shape, (2, 4))

    def test_dropout_marks_inactive_codebooks_during_training(self):
        torch.manual_seed(0)
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, codebook_size=3, dropout=1.0)
        )

        output = quantizer(torch.randn(64, 4))

        self.assertTrue((output.indices == -1).any())
        self.assertFalse(output.active_codebook_mask.all())

    def test_dropout_preserves_empty_trailing_codebooks(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=3, codebook_size=3, dropout=1.0)
        )
        active_mask = torch.tensor([[True, False, False], [True, False, False]])

        with mock.patch.object(quantizer, "_sample_active_mask", return_value=active_mask):
            output = quantizer(torch.randn(2, 4))

        self.assertTrue(torch.equal(output.active_codebook_mask, active_mask))
        self.assertTrue((output.indices[:, 0] >= 0).all())
        self.assertTrue((output.indices[:, 1:] == -1).all())
        self.assertTrue((output.codebook_vectors[:, 1:] == 0).all())
        self.assertTrue((output.latents[:, 1:] == 0).all())

    def test_sample_active_mask_dropout_boundaries(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, codebook_size=3, dropout=0.0)
        )

        mask = quantizer._sample_active_mask(16, 4, torch.device("cpu"))

        self.assertTrue(mask.all())

        quantizer.set_dropout(1.0)
        torch.manual_seed(0)
        mask = quantizer._sample_active_mask(64, 4, torch.device("cpu"))
        active_counts = mask.sum(dim=-1)

        self.assertTrue((active_counts >= 1).all())
        self.assertTrue((active_counts <= 4).all())
        self.assertTrue((active_counts < 4).any())

    def test_config_is_frozen_and_dropout_has_explicit_setter(self):
        config = AGRVQConfig(input_dim=4, num_codebooks=2, codebook_size=3)
        quantizer = AutoGroupResidualVectorQuantizer(config)

        with self.assertRaises(FrozenInstanceError):
            config.dropout = 1.0
        returned = quantizer.set_dropout(0.5)

        self.assertIs(returned, quantizer)
        self.assertEqual(quantizer.dropout, 0.5)
        self.assertIsNone(quantizer.config.dropout)

    def test_dropout_has_no_effect_with_one_active_codebook(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, codebook_size=3, dropout=1.0)
        )

        output = quantizer(torch.randn(8, 4), num_active_codebooks=1)

        self.assertEqual(output.indices.shape, (8, 1))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_eval_disables_dropout(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, codebook_size=3, dropout=1.0)
        )
        quantizer.eval()

        output = quantizer(torch.randn(8, 4))

        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_backward_reaches_input_and_codebooks(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, codebook_size=3, codebook_dim=2)
        )
        x = torch.randn(3, 4, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for book in quantizer.quantizers:
            self.assertIsNotNone(book.codebook_a.weight.grad)
            self.assertIsNotNone(book.codebook_b.weight.grad)

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "even"):
            AGRVQConfig(input_dim=3, num_codebooks=2, codebook_size=3)
        with self.assertRaisesRegex(ValueError, "num_codebooks"):
            AGRVQConfig(input_dim=4, num_codebooks=0, codebook_size=3)
        with self.assertRaisesRegex(ValueError, "dropout"):
            AGRVQConfig(input_dim=4, num_codebooks=2, codebook_size=3, dropout=2.0)


if __name__ == "__main__":
    unittest.main()
