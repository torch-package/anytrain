import unittest
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import mock

import torch

from anytrain.module.quantization import (
    EmbeddingVectorQuantizer,
    QuantizationLoss,
    VQConfig,
)
from anytrain.module.quantization.lookup import nearest_codebook_indices
from anytrain.module.quantization.output import QuantizeOutput


def _distributed_kmeans_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(0)
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(
                input_dim=2,
                codebook_size=4,
                use_ema=True,
                kmeans_init=True,
                kmeans_iters=2,
                dead_code_threshold=1.0,
            )
        )
        if rank == 0:
            _ = quantizer(torch.tensor([[2.0, 0.0], [-2.0, 0.0]]))
        else:
            quantizer._update_ema_without_assignments()

        if not bool(quantizer.ema_counts.sum().item() > 0):
            raise AssertionError("distributed k-means did not initialize EMA state.")
        with torch.no_grad():
            quantizer._ema_counts[0] = 0
            quantizer._ema_sums[0].zero_()
        before_recycle = quantizer.codebook.weight[0].clone()
        if rank == 0:
            _ = quantizer(torch.tensor([[0.0, 3.0], [0.0, -3.0]]))
        else:
            quantizer._update_ema_without_assignments()
        if torch.equal(quantizer.codebook.weight[0], before_recycle):
            raise AssertionError("distributed dead-code recycling did not replace the code.")
        for value in (
            quantizer._ema_counts,
            quantizer._ema_sums,
            quantizer.codebook.weight,
        ):
            gathered = [torch.empty_like(value) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, value)
            for peer in gathered[1:]:
                torch.testing.assert_close(gathered[0], peer)
    finally:
        torch.distributed.destroy_process_group()


class EmbeddingVectorQuantizerTest(unittest.TestCase):
    def test_forward_preserves_legacy_one_argument_quantize_override(self):
        class LegacyQuantizer(EmbeddingVectorQuantizer):
            def quantize(  # pyright: ignore[reportIncompatibleMethodOverride]
                self,
                latents: torch.Tensor,
            ) -> QuantizeOutput:
                return super().quantize(latents)

        quantizer = LegacyQuantizer(VQConfig(input_dim=2, codebook_size=4))
        output = quantizer(torch.randn(3, 2))

        self.assertEqual(output.indices.shape, (3,))

    def test_config_is_plain_data(self):
        config = VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)

        field_names = {field.name for field in fields(config)}

        self.assertIn("normalize_lookup", field_names)
        self.assertIn("lookup_in_float32", field_names)
        self.assertIn("kmeans_init", field_names)
        self.assertIn("dead_code_threshold", field_names)
        self.assertNotIn("normalize_latents", field_names)
        self.assertEqual(config.codebook_dim, 4)

    def test_config_is_immutable_and_resolves_default_dimension(self):
        config = VQConfig(input_dim=8, codebook_size=16)

        self.assertIsNone(config.codebook_dim)
        self.assertEqual(config.resolved_codebook_dim, 8)
        with self.assertRaises(AttributeError):
            config.__setattr__("codebook_size", 32)

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

    def test_indices_reject_bool_tensor(self):
        quantizer = EmbeddingVectorQuantizer(VQConfig(input_dim=4, codebook_size=8))

        with self.assertRaisesRegex(TypeError, "bool tensors are not valid indices"):
            quantizer.indices_to_codebook_vectors(torch.tensor([True, False]))

    def test_quantization_inputs_reject_non_tensors(self):
        quantizer = EmbeddingVectorQuantizer(VQConfig(input_dim=4, codebook_size=8))

        with self.assertRaisesRegex(TypeError, "latents must be a tensor"):
            quantizer.quantize(cast(Any, [[0.0, 0.0, 0.0, 0.0]]))
        with self.assertRaisesRegex(TypeError, "indices must be a tensor"):
            quantizer.indices_to_codebook_vectors(cast(Any, [[0]]))

    def test_backward_reaches_input_and_codebook_without_ema(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        x = torch.randn(3, 8, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(quantizer.codebook.weight.grad)
        assert quantizer.codebook.weight.grad is not None
        self.assertTrue(torch.isfinite(quantizer.codebook.weight.grad).all())

    def test_ema_updates_with_commitment_loss_without_codebook_grad(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4, use_ema=True, decay=0.5)
        )
        before_counts = quantizer.ema_counts.clone()

        output = quantizer(torch.randn(6, 8))

        self.assertIsInstance(output.loss, QuantizationLoss)
        self.assertEqual(output.loss.codebook.item(), 0.0)
        self.assertGreaterEqual(output.loss.commitment.item(), 0.0)
        self.assertFalse(torch.allclose(quantizer.ema_counts, before_counts))
        self.assertFalse(quantizer.codebook.weight.requires_grad)

    def test_ema_update_does_not_materialize_one_hot_assignments(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4, use_ema=True)
        )

        with mock.patch("torch.nn.functional.one_hot", side_effect=AssertionError("one_hot")):
            output = quantizer(torch.randn(32, 8))

        self.assertIsInstance(output.loss, QuantizationLoss)
        self.assertTrue(torch.isfinite(quantizer.codebook.weight).all())

    def test_kmeans_initializes_ema_codebook_from_first_training_batch(self):
        torch.manual_seed(0)
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(
                input_dim=2,
                codebook_size=2,
                use_ema=True,
                decay=0.5,
                kmeans_init=True,
                kmeans_iters=4,
            )
        )
        latents = torch.tensor([[-2.0, 0.0], [-1.9, 0.0], [1.9, 0.0], [2.0, 0.0]]).repeat(8, 1)

        self.assertFalse(quantizer.ema_counts.any())
        _ = quantizer(latents)

        centers = quantizer.codebook.weight.detach()
        ordered = centers[centers[:, 0].argsort()]
        torch.testing.assert_close(
            ordered,
            torch.tensor([[-1.95, 0.0], [1.95, 0.0]]),
            atol=0.1,
            rtol=0,
        )
        self.assertTrue(quantizer.ema_counts.all())

    def test_eval_does_not_trigger_kmeans_initialization(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(
                input_dim=2,
                codebook_size=4,
                use_ema=True,
                kmeans_init=True,
            )
        )
        before = quantizer.codebook.weight.clone()
        quantizer.eval()

        _ = quantizer(torch.randn(8, 2))

        self.assertFalse(quantizer.ema_counts.any())
        self.assertTrue(torch.equal(quantizer.codebook.weight, before))

    def test_kmeans_initialization_resumes_from_existing_ema_state(self):
        config = VQConfig(
            input_dim=2,
            codebook_size=4,
            use_ema=True,
            kmeans_init=True,
            kmeans_iters=2,
        )
        quantizer = EmbeddingVectorQuantizer(config)
        _ = quantizer(torch.randn(16, 2))
        state = quantizer.state_dict()
        restored = EmbeddingVectorQuantizer(config)
        restored.load_state_dict(state, strict=True)
        before = restored.codebook.weight.clone()

        restored._prepare_ema(torch.randn(8, 2))

        self.assertTrue(torch.equal(restored.codebook.weight, before))
        self.assertNotIn("_ema_initialized", state)

    def test_dead_codes_are_recycled_from_current_latents(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(
                input_dim=2,
                codebook_size=4,
                use_ema=True,
                dead_code_threshold=2.0,
            )
        )
        with torch.no_grad():
            quantizer._ema_counts.copy_(torch.tensor([3.0, 0.0, 3.0, 0.0]))
            quantizer._ema_sums.copy_(quantizer.codebook.weight * quantizer._ema_counts[:, None])
        candidates = torch.tensor([[10.0, 0.0], [0.0, 10.0]])

        quantizer._prepare_ema(candidates)

        dead = torch.tensor([1, 3])
        replacements = quantizer.codebook.weight[dead]
        distances = torch.cdist(replacements, candidates)
        self.assertTrue((distances.min(dim=1).values == 0).all())
        torch.testing.assert_close(quantizer.ema_counts[dead], torch.full((2,), 2.0))
        torch.testing.assert_close(
            quantizer._ema_sums[dead],
            replacements * 2.0,
        )

    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(),
        "Gloo distributed backend is unavailable",
    )
    def test_distributed_kmeans_allows_a_rank_without_assignments(self):
        with TemporaryDirectory() as tmp_dir:
            init_method = (Path(tmp_dir) / "init").as_uri()
            multiprocessing: Any = torch.multiprocessing
            multiprocessing.spawn(
                _distributed_kmeans_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )

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
        quantizer = EmbeddingVectorQuantizer(VQConfig(input_dim=2, codebook_size=4, use_ema=True))
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

    def test_lookup_can_compare_in_float32(self):
        latents = torch.tensor([[200.0, 0.0]], dtype=torch.float16)
        codebook = torch.tensor([[0.0, 0.0], [200.0, 0.0]], dtype=torch.float16)

        indices = nearest_codebook_indices(
            latents,
            codebook,
            normalize=False,
            lookup_in_float32=True,
        )

        self.assertTrue(torch.equal(indices, torch.tensor([1])))

    def test_non_normalized_lookup(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=2, codebook_size=2, normalize_lookup=False)
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
        with self.assertRaisesRegex(ValueError, "kmeans_iters"):
            VQConfig(input_dim=4, codebook_size=8, kmeans_iters=0)
        with self.assertRaisesRegex(ValueError, "dead_code_threshold"):
            VQConfig(input_dim=4, codebook_size=8, dead_code_threshold=-1.0)
        with self.assertRaisesRegex(ValueError, "require use_ema"):
            VQConfig(input_dim=4, codebook_size=8, kmeans_init=True)


if __name__ == "__main__":
    unittest.main()
