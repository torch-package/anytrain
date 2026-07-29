import unittest
from dataclasses import fields
from unittest import mock

import torch

from anytrain.module.quantization import (
    EmbeddingVectorQuantizer,
    QuantizationLoss,
    VQConfig,
)
from anytrain.module.quantization.lookup import nearest_codebook_indices


class EmbeddingVectorQuantizerTest(unittest.TestCase):
    def test_config_is_plain_data(self):
        config = VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)

        field_names = {field.name for field in fields(config)}

        self.assertIn("normalize_latents", field_names)
        self.assertEqual(config.codebook_dim, 4)

    def test_forward_shapes(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        x = torch.randn(3, 5, 8)

        output = quantizer(x)

        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (3, 5))
        self.assertEqual(output.codebook_vectors.shape, (3, 5, 4))
        self.assertEqual(output.latents.shape, (3, 5, 4))
        self.assertIsInstance(output.loss, QuantizationLoss)
        self.assertEqual(output.loss.commitment.ndim, 0)
        self.assertEqual(output.loss.codebook.ndim, 0)

    def test_eval_returns_no_loss(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8))

        self.assertIsNone(output.loss)
        self.assertEqual(output.indices.shape, (2,))

    def test_indices_round_trip(self):
        quantizer = EmbeddingVectorQuantizer(VQConfig(input_dim=4, codebook_size=8))
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_backward_reaches_input_and_codebook_without_ema(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        x = torch.randn(3, 8, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(quantizer.codebook.weight.grad)
        self.assertTrue(torch.isfinite(quantizer.codebook.weight.grad).all())

    def test_ema_updates_without_codebook_grad(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4, use_ema=True, decay=0.5)
        )
        before_counts = quantizer.ema_counts.clone()

        output = quantizer(torch.randn(6, 8))

        self.assertIsNone(output.loss)
        self.assertFalse(torch.allclose(quantizer.ema_counts, before_counts))
        self.assertFalse(quantizer.codebook.weight.requires_grad)

    def test_ema_update_does_not_materialize_one_hot_assignments(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4, use_ema=True)
        )

        with mock.patch("torch.nn.functional.one_hot", side_effect=AssertionError("one_hot")):
            output = quantizer(torch.randn(32, 8))

        self.assertIsNone(output.loss)
        self.assertTrue(torch.isfinite(quantizer.codebook.weight).all())

    def test_ema_update_synchronizes_counts_and_sums_when_distributed(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=2, codebook_size=4, use_ema=True, decay=0.0)
        )
        calls: list[torch.Tensor] = []

        def all_reduce(value, *, op):
            self.assertEqual(op, torch.distributed.ReduceOp.SUM)
            calls.append(value)
            value.mul_(2)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.all_reduce", side_effect=all_reduce),
        ):
            quantizer(torch.randn(8, 2))

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].shape, quantizer._ema_counts.shape)
        self.assertEqual(calls[1].shape, quantizer._ema_sums.shape)
        self.assertTrue(torch.isfinite(quantizer.codebook.weight).all())

    def test_ema_stats_stay_fp32_after_half_conversion_and_distributed_sum(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=2, codebook_size=4, use_ema=True, decay=0.0)
        ).half()
        indices = torch.zeros(40_000, dtype=torch.long)
        latents = torch.ones(40_000, 2, dtype=torch.float16)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=lambda value, *, op: value.mul_(2),
            ),
        ):
            quantizer._update_ema(latents, indices)

        self.assertEqual(quantizer._ema_counts.dtype, torch.float32)
        self.assertEqual(quantizer._ema_sums.dtype, torch.float32)
        self.assertEqual(quantizer.codebook.weight.dtype, torch.float16)
        self.assertTrue(torch.isfinite(quantizer._ema_counts).all())
        self.assertTrue(torch.isfinite(quantizer.codebook.weight).all())

    def test_empty_distributed_ema_participation_preserves_state(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=2, codebook_size=4, use_ema=True)
        )
        before_counts = quantizer._ema_counts.clone()
        before_sums = quantizer._ema_sums.clone()
        before_weight = quantizer.codebook.weight.clone()
        calls: list[torch.Tensor] = []

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=lambda value, *, op: calls.append(value.clone()),
            ),
        ):
            quantizer._update_ema_without_assignments()

        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0].any())
        self.assertFalse(calls[1].any())
        self.assertTrue(torch.equal(quantizer._ema_counts, before_counts))
        self.assertTrue(torch.equal(quantizer._ema_sums, before_sums))
        self.assertTrue(torch.equal(quantizer.codebook.weight, before_weight))

    def test_lookup_chunks_large_comparison_matrices(self):
        latents = torch.randn(7, 3)
        codebook = torch.randn(5, 3)

        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                chunked = nearest_codebook_indices(
                    latents,
                    codebook,
                    normalize=normalize,
                    max_lookup_elements=10,
                )
                dense = nearest_codebook_indices(
                    latents,
                    codebook,
                    normalize=normalize,
                    max_lookup_elements=100,
                )

                self.assertTrue(torch.equal(chunked, dense))

    def test_lookup_chunks_codebook_larger_than_comparison_limit(self):
        latents = torch.randn(3, 4)
        codebook = torch.randn(11, 4)

        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                chunked = nearest_codebook_indices(
                    latents,
                    codebook,
                    normalize=normalize,
                    max_lookup_elements=5,
                )
                dense = nearest_codebook_indices(
                    latents,
                    codebook,
                    normalize=normalize,
                    max_lookup_elements=100,
                )

                self.assertTrue(torch.equal(chunked, dense))

    def test_lookup_prefers_finite_codebook_vectors_over_nan(self):
        latents = torch.tensor([[1.0, 0.0]])
        codebook = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [float("nan"), 0.0],
                [-1.0, 0.0],
            ]
        )

        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                indices = nearest_codebook_indices(
                    latents,
                    codebook,
                    normalize=normalize,
                    max_lookup_elements=2,
                )

                self.assertTrue(torch.equal(indices, torch.tensor([0])))

    def test_non_normalized_lookup(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=2, codebook_size=2, normalize_latents=False)
        )
        with torch.no_grad():
            quantizer.codebook.weight.copy_(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))

        output = quantizer(torch.tensor([[9.0, 0.0], [1.0, 0.0]]))

        self.assertTrue(torch.equal(output.indices, torch.tensor([1, 0])))

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            VQConfig(input_dim=4, codebook_size=0)
        with self.assertRaisesRegex(ValueError, "decay"):
            VQConfig(input_dim=4, codebook_size=8, decay=1.0)


if __name__ == "__main__":
    unittest.main()
