import unittest
from dataclasses import FrozenInstanceError
from typing import Any, cast
from unittest import mock

import torch

from anytrain._compat import strict_zip
from anytrain.module.quantization import (
    AGRVQConfig,
    AutoGroupResidualVectorQuantizer,
    QuantizationLoss,
)
from anytrain.module.quantization.auto_group_residual import _AutoGroupVectorQuantizer


class AutoGroupResidualVectorQuantizerTest(unittest.TestCase):
    def test_forward_shapes_and_public_codebook_size(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=8, num_codebooks=2, group_size=5, codebook_dim=4)
        )
        x = torch.randn(2, 3, 8)

        output = quantizer(x)

        self.assertEqual(quantizer.num_groups, 2)
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

    def test_configurable_num_groups_shapes_and_public_codebook_size(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(
                input_dim=6,
                num_codebooks=2,
                group_size=4,
                codebook_dim=2,
                num_groups=3,
            )
        )
        x = torch.randn(2, 3, 6)

        output = quantizer(x)

        self.assertEqual(quantizer.num_groups, 3)
        self.assertEqual(quantizer.group_size, 4)
        self.assertEqual(quantizer.codebook_size, 64)
        self.assertEqual(quantizer.codebook_dim, 6)
        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (2, 3, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 3, 2, 6))
        self.assertEqual(output.latents.shape, (2, 3, 2, 6))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue((output.indices < 64).all())

    def test_one_group_shapes_and_public_codebook_size(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(
                input_dim=4,
                num_codebooks=2,
                group_size=7,
                codebook_dim=3,
                num_groups=1,
            )
        )
        x = torch.randn(2, 4)

        output = quantizer(x)
        group_indices = quantizer.indices_to_group_indices(output.indices)
        round_trip = quantizer.group_indices_to_indices(group_indices)

        self.assertEqual(quantizer.num_groups, 1)
        self.assertEqual(quantizer.codebook_size, 7)
        self.assertEqual(quantizer.codebook_dim, 3)
        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.codebook_vectors.shape, (2, 2, 3))
        self.assertTrue(torch.equal(round_trip, output.indices))

    def test_num_active_codebooks_controls_output_width(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=8, num_codebooks=4, group_size=5)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8), num_active_codebooks=2)

        self.assertEqual(output.indices.shape, (2, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 2, 16))
        self.assertTrue(output.active_codebook_mask.all())

    def test_group_indices_round_trip(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=5, codebook_dim=2)
        )
        group_indices = torch.tensor([[[0, 0], [2, 4]], [[1, 3], [4, 2]]])

        indices = quantizer.group_indices_to_indices(group_indices)
        round_trip = quantizer.indices_to_group_indices(indices)

        self.assertTrue(torch.equal(indices, torch.tensor([[0, 22], [16, 14]])))
        self.assertTrue(torch.equal(round_trip, group_indices))

    def test_group_indices_round_trip_with_more_groups(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(
                input_dim=6,
                num_codebooks=2,
                group_size=4,
                codebook_dim=2,
                num_groups=3,
            )
        )
        group_indices = torch.tensor([[[0, 0, 0], [2, 1, 3]], [[1, 2, 0], [3, 3, 3]]])

        indices = quantizer.group_indices_to_indices(group_indices)
        round_trip = quantizer.indices_to_group_indices(indices)

        self.assertTrue(torch.equal(indices, torch.tensor([[0, 54], [9, 63]])))
        self.assertTrue(torch.equal(round_trip, group_indices))

    def test_indices_round_trip_through_codebook_vectors(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(
                input_dim=6,
                num_codebooks=2,
                group_size=3,
                codebook_dim=2,
                num_groups=3,
            )
        )
        books: Any = quantizer.quantizers
        weights = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]),
            torch.tensor([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]),
        )
        with torch.no_grad():
            for book in books:
                for codebook, weight in strict_zip(book.codebooks, weights):
                    codebook.weight.copy_(weight)
        indices = torch.tensor([[0, 1], [5, 26]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_project_codebook_vectors_sums_projected_books(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3, codebook_dim=2)
        )
        indices = torch.tensor([[0, 1], [5, 8]])
        vectors = quantizer.indices_to_codebook_vectors(indices)

        projected = quantizer.project_codebook_vectors(vectors)

        self.assertEqual(projected.shape, (2, 4))

    def test_dropout_marks_inactive_codebooks_during_training(self):
        torch.manual_seed(0)
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, group_size=3, dropout=1.0)
        )

        output = quantizer(torch.randn(64, 4))

        self.assertTrue((output.indices == -1).any())
        self.assertFalse(output.active_codebook_mask.all())

    def test_dropout_preserves_empty_trailing_codebooks(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=3, group_size=3, dropout=1.0)
        )
        active_mask = torch.tensor([[True, False, False], [True, False, False]])

        with mock.patch.object(quantizer, "_sample_active_mask", return_value=active_mask):
            output = quantizer(torch.randn(2, 4))

        self.assertTrue(torch.equal(output.active_codebook_mask, active_mask))
        self.assertTrue((output.indices[:, 0] >= 0).all())
        self.assertTrue((output.indices[:, 1:] == -1).all())
        self.assertTrue((output.codebook_vectors[:, 1:] == 0).all())
        self.assertTrue((output.latents[:, 1:] == 0).all())

    def test_dropout_empty_stages_keep_outputs_finite_and_gradients(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=3, group_size=3, dropout=1.0)
        )
        active_mask = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [False, False, False],
                [True, True, True],
            ],
            dtype=torch.bool,
        )
        latents = torch.randn(4, 4, requires_grad=True)

        with (
            mock.patch.object(quantizer, "_sample_active_mask", return_value=active_mask),
            mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("item")),
        ):
            output = quantizer(latents)

        self.assertTrue(torch.equal(output.active_codebook_mask, active_mask))
        self.assertTrue(torch.isfinite(output.quantized_latents).all())
        self.assertTrue(torch.isfinite(output.indices.float()).all())
        self.assertTrue((output.indices[2] == -1).all())
        self.assertTrue((output.codebook_vectors[2] == 0).all())
        self.assertTrue((output.latents[2] == 0).all())
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()
        self.assertIsNotNone(latents.grad)
        assert latents.grad is not None
        self.assertTrue(torch.isfinite(latents.grad).all())

    def test_dropout_all_inactive_mask_returns_zeros_without_loss(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=3, group_size=3, dropout=1.0)
        )
        active_mask = torch.zeros(2, 3, dtype=torch.bool)

        with (
            mock.patch.object(quantizer, "_sample_active_mask", return_value=active_mask),
            mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("item")),
        ):
            output = quantizer(torch.randn(2, 4))

        self.assertTrue(torch.equal(output.active_codebook_mask, active_mask))
        self.assertTrue((output.quantized_latents == 0).all())
        self.assertTrue((output.indices == -1).all())
        self.assertTrue((output.codebook_vectors == 0).all())
        self.assertTrue((output.latents == 0).all())
        self.assertIsNone(output.loss)

    def test_indices_decode_masks_inactive_entries(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3)
        )
        indices = torch.tensor([[-1, -1], [0, -1], [2, 3]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        books = cast(tuple[_AutoGroupVectorQuantizer, ...], tuple(quantizer.quantizers))

        self.assertTrue(torch.equal(vectors[0], torch.zeros_like(vectors[0])))
        self.assertTrue(torch.equal(vectors[1, 1], torch.zeros_like(vectors[1, 1])))
        self.assertTrue(
            torch.equal(
                vectors[2, 0],
                books[0].indices_to_codebook_vectors(indices[2, 0:1])[0],
            )
        )
        self.assertTrue(
            torch.equal(
                vectors[2, 1],
                books[1].indices_to_codebook_vectors(indices[2, 1:2])[0],
            )
        )

    def test_indices_decode_preserves_instance_stage_override(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3)
        )
        stage = cast(_AutoGroupVectorQuantizer, quantizer.quantizers[0])

        with mock.patch.object(
            stage,
            "indices_to_codebook_vectors",
            wraps=stage.indices_to_codebook_vectors,
        ) as decode:
            quantizer.indices_to_codebook_vectors(torch.tensor([[0, 1], [2, 3]]))

        decode.assert_called_once()

    def test_indices_decode_preserves_subclass_stage_override(self):
        class CustomAutoGroupVectorQuantizer(_AutoGroupVectorQuantizer):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.decode_calls = 0

            def indices_to_codebook_vectors(self, indices: torch.Tensor) -> torch.Tensor:
                self.decode_calls += 1
                return super().indices_to_codebook_vectors(indices)

        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3)
        )
        base = cast(_AutoGroupVectorQuantizer, quantizer.quantizers[0])
        custom_stage = CustomAutoGroupVectorQuantizer(
            input_dim=base.input_dim,
            group_size=base.group_size,
            codebook_dim=base.group_codebook_dim,
            num_groups=base.num_groups,
            normalize_latents=base.normalize_latents,
            projection_bias=True,
            weight_norm=False,
        )
        quantizer.quantizers[0] = custom_stage

        quantizer.indices_to_codebook_vectors(torch.tensor([[0, 1], [2, 3]]))

        self.assertEqual(custom_stage.decode_calls, 1)

    def test_indices_decode_preserves_class_stage_override(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3)
        )
        original = _AutoGroupVectorQuantizer.indices_to_codebook_vectors
        calls: list[torch.Tensor] = []

        def patched(
            instance: _AutoGroupVectorQuantizer,
            indices: torch.Tensor,
        ) -> torch.Tensor:
            calls.append(indices)
            return original(instance, indices)

        with mock.patch.object(
            _AutoGroupVectorQuantizer,
            "indices_to_codebook_vectors",
            patched,
        ):
            quantizer.indices_to_codebook_vectors(torch.tensor([[0, 1], [2, 3]]))

        self.assertEqual(len(calls), 2)

    def test_sample_active_mask_dropout_boundaries(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, group_size=3, dropout=0.0)
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
        config = AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3)
        quantizer = AutoGroupResidualVectorQuantizer(config)

        with self.assertRaises(FrozenInstanceError):
            cast(Any, config).dropout = 1.0
        returned = quantizer.set_dropout(0.5)

        self.assertIs(returned, quantizer)
        self.assertEqual(quantizer.dropout, 0.5)
        self.assertIsNone(quantizer.config.dropout)

    def test_dropout_has_no_effect_with_one_active_codebook(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, group_size=3, dropout=1.0)
        )

        output = quantizer(torch.randn(8, 4), num_active_codebooks=1)

        self.assertEqual(output.indices.shape, (8, 1))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_eval_disables_dropout(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=4, group_size=3, dropout=1.0)
        )
        quantizer.eval()

        output = quantizer(torch.randn(8, 4))

        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_group_indices_shape_error_mentions_shape(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=6, num_codebooks=2, group_size=3, num_groups=3)
        )

        with self.assertRaisesRegex(ValueError, r"\(2, 2\)"):
            quantizer.group_indices_to_indices(torch.zeros(2, 2, dtype=torch.long))

    def test_backward_reaches_input_and_codebooks(self):
        quantizer = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3, codebook_dim=2)
        )
        x = torch.randn(3, 4, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())
        books: Any = quantizer.quantizers
        for book in books:
            for codebook in book.codebooks:
                self.assertIsNotNone(codebook.weight.grad)

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            AGRVQConfig(input_dim=3, num_codebooks=2, group_size=3)
        with self.assertRaisesRegex(ValueError, "num_codebooks"):
            AGRVQConfig(input_dim=4, num_codebooks=0, group_size=3)
        with self.assertRaisesRegex(ValueError, "group_size"):
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=0)
        with self.assertRaisesRegex(ValueError, "num_groups"):
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3, num_groups=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            AGRVQConfig(input_dim=4, num_codebooks=2, group_size=3, dropout=2.0)


if __name__ == "__main__":
    unittest.main()
