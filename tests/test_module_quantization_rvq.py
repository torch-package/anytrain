from __future__ import annotations

import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import mock

import torch

import anytrain.module.quantization as quantization_api
from anytrain._compat import strict_zip
from anytrain.module.quantization import (
    EmbeddingVectorQuantizer,
    ResidualQuantizerProtocol,
    ResidualVectorQuantizer,
    RVQConfig,
    VQConfig,
)


def _rvq_config(
    *,
    input_dim: int,
    num_codebooks: int,
    codebook_size: int,
    codebook_dim: int | None = None,
    normalize_lookup: bool = True,
    lookup_in_float32: bool = True,
    weight_norm: bool = False,
    scale_grad_by_freq: bool = False,
    use_ema: bool = False,
    decay: float = 0.99,
    eps: float = 1e-5,
    projection_bias: bool = True,
    kmeans_init: bool = False,
    kmeans_iters: int = 10,
    dead_code_threshold: float = 0.0,
    dropout: float | None = None,
) -> RVQConfig:
    return RVQConfig(
        vq_configs=tuple(
            VQConfig(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                normalize_lookup=normalize_lookup,
                lookup_in_float32=lookup_in_float32,
                weight_norm=weight_norm,
                scale_grad_by_freq=scale_grad_by_freq,
                use_ema=use_ema,
                decay=decay,
                eps=eps,
                projection_bias=projection_bias,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                dead_code_threshold=dead_code_threshold,
            )
            for _ in range(num_codebooks)
        ),
        dropout=dropout,
    )


def _distributed_ema_dropout_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(0)
        quantizer = ResidualVectorQuantizer(
            _rvq_config(
                input_dim=4,
                num_codebooks=3,
                codebook_size=8,
                codebook_dim=2,
                use_ema=True,
                kmeans_init=True,
                kmeans_iters=2,
                dead_code_threshold=1.0,
                dropout=1.0,
            )
        )
        distributed_quantizer = torch.nn.parallel.DistributedDataParallel(
            quantizer,
            find_unused_parameters=True,
        )
        books: Any = quantizer.quantizers
        before = [
            (book._ema_counts.clone(), book._ema_sums.clone(), book.codebook.weight.clone())
            for book in books
        ]
        active_mask = torch.tensor(
            [[True, rank == 1, False]],
            dtype=torch.bool,
        )
        with mock.patch.object(
            quantizer,
            "_sample_active_mask",
            return_value=active_mask,
        ):
            output = distributed_quantizer(torch.full((1, 4), float(rank + 1)))
            output.quantized_latents.square().mean().backward()

        if not torch.equal(output.active_codebook_mask, active_mask):
            raise AssertionError("RVQ did not preserve the rank-local dropout mask.")
        for index, book in enumerate(books):
            if index == 2:
                for value, initial in strict_zip(
                    (book._ema_counts, book._ema_sums, book.codebook.weight),
                    before[index],
                ):
                    if not torch.equal(value, initial):
                        raise AssertionError("A globally empty EMA stage changed state.")
            for value in (book._ema_counts, book._ema_sums, book.codebook.weight):
                gathered = [torch.empty_like(value) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, value)
                for peer in gathered[1:]:
                    torch.testing.assert_close(gathered[0], peer)
    finally:
        torch.distributed.destroy_process_group()


class ResidualVectorQuantizerTest(unittest.TestCase):
    def test_quantization_exports_residual_protocol(self):
        self.assertIn("ResidualQuantizerProtocol", quantization_api.__all__)
        self.assertIn(quantization_api.QuantizerProtocol, ResidualQuantizerProtocol.__bases__)

    def test_forward_shapes(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=8, num_codebooks=3, codebook_size=16, codebook_dim=8)
        )
        x = torch.randn(2, 5, 8)

        output = quantizer(x)

        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (2, 5, 3))
        self.assertEqual(output.codebook_vectors.shape, (2, 5, 3, 8))
        self.assertEqual(output.latents.shape, (2, 5, 3, 8))
        self.assertEqual(output.active_codebook_mask.shape, (2, 5, 3))
        self.assertIsNotNone(output.loss)

    def test_num_active_codebooks_controls_output_width(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=8, num_codebooks=4, codebook_size=16)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8), num_active_codebooks=2)

        self.assertEqual(output.indices.shape, (2, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 2, 8))
        self.assertTrue(output.active_codebook_mask.all())

    def test_indices_round_trip(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[[0, 1], [2, 3]]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_indices_decode_preserves_subclass_stage_override(self):
        class CustomEmbeddingVectorQuantizer(EmbeddingVectorQuantizer):
            def __init__(self, config: VQConfig) -> None:
                super().__init__(config)
                self.decode_calls = 0

            def indices_to_codebook_vectors(self, indices: torch.Tensor) -> torch.Tensor:
                self.decode_calls += 1
                return super().indices_to_codebook_vectors(indices)

        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        custom_stage = CustomEmbeddingVectorQuantizer(quantizer.config.vq_configs[0])
        quantizer.quantizers[0] = custom_stage

        quantizer.indices_to_codebook_vectors(torch.tensor([[0, 1], [2, 3]]))

        self.assertEqual(custom_stage.decode_calls, 1)

    def test_indices_decode_preserves_instance_stage_override(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        stage = quantizer.quantizers[0]

        with mock.patch.object(
            stage,
            "indices_to_codebook_vectors",
            wraps=stage.indices_to_codebook_vectors,
        ) as decode:
            quantizer.indices_to_codebook_vectors(torch.tensor([[0, 1], [2, 3]]))

        decode.assert_called_once()

    def test_indices_decode_preserves_class_stage_override(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        original = EmbeddingVectorQuantizer.indices_to_codebook_vectors
        calls: list[torch.Tensor] = []

        def patched(
            instance: EmbeddingVectorQuantizer,
            indices: torch.Tensor,
        ) -> torch.Tensor:
            calls.append(indices)
            return original(instance, indices)

        with mock.patch.object(
            EmbeddingVectorQuantizer,
            "indices_to_codebook_vectors",
            patched,
        ):
            quantizer.indices_to_codebook_vectors(torch.tensor([[0, 1], [2, 3]]))

        self.assertEqual(len(calls), 2)

    def test_project_codebook_vectors_sums_projected_books(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[0, 1], [2, 3]])
        vectors = quantizer.indices_to_codebook_vectors(indices)

        projected = quantizer.project_codebook_vectors(vectors)

        self.assertEqual(projected.shape, (2, 4))

    def test_latents_to_codebook_vectors_has_no_training_side_effects(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(
                input_dim=4,
                num_codebooks=3,
                codebook_size=8,
                use_ema=True,
                dropout=1.0,
            )
        )
        books: Any = quantizer.quantizers
        before_counts = [book.ema_counts.clone() for book in books]

        vectors = quantizer.latents_to_codebook_vectors(torch.randn(16, 4))

        self.assertEqual(vectors.shape, (16, 3, 4))
        for book, before in strict_zip(books, before_counts):
            self.assertTrue(torch.equal(book.ema_counts, before))

    def test_dropout_marks_inactive_codebooks_during_training(self):
        torch.manual_seed(0)
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )

        output = quantizer(torch.randn(64, 4))

        self.assertTrue((output.indices == -1).any())
        self.assertFalse(output.active_codebook_mask.all())

    def test_dropout_empty_stages_keep_outputs_finite_and_gradients(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=3, codebook_size=8, dropout=1.0)
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

    def test_dropout_loss_count_accumulates_in_wide_dtype(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(
                input_dim=1,
                num_codebooks=2,
                codebook_size=1,
                codebook_dim=1,
                normalize_lookup=False,
                projection_bias=False,
                dropout=0.5,
            )
        ).half()
        for stage in quantizer.quantizers:
            stage.codebook.weight.data.zero_()
        latents = torch.full(
            (1, 65536, 1), 0.5, dtype=torch.float16, requires_grad=True
        )
        active_mask = torch.tensor([[True, False]], dtype=torch.bool).expand(65536, 2)
        with mock.patch.object(quantizer, "_sample_active_mask", return_value=active_mask):
            output = quantizer(latents)
        assert output.loss is not None
        self.assertTrue(torch.isfinite(output.loss.commitment))
        self.assertAlmostEqual(float(output.loss.commitment.detach()), 0.25, places=3)
        self.assertTrue((output.latents[:, :, 1] == 0).all())
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()
        self.assertIsNotNone(latents.grad)
        assert latents.grad is not None
        self.assertTrue(torch.isfinite(latents.grad).all())

    def test_dropout_all_inactive_mask_returns_zeros_without_loss(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=3, codebook_size=8, dropout=1.0)
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
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[-1, -1], [0, -1], [2, 3]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        books = cast(tuple[EmbeddingVectorQuantizer, ...], tuple(quantizer.quantizers))

        self.assertTrue(torch.equal(vectors[0], torch.zeros_like(vectors[0])))
        self.assertTrue(torch.equal(vectors[1, 1], torch.zeros_like(vectors[1, 1])))
        self.assertTrue(torch.equal(vectors[2, 0], books[0].codebook.weight[2]))
        self.assertTrue(torch.equal(vectors[2, 1], books[1].codebook.weight[3]))

    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(),
        "Gloo distributed backend is unavailable",
    )
    def test_distributed_ema_dropout_keeps_collectives_aligned(self):
        with TemporaryDirectory() as tmp_dir:
            init_method = (Path(tmp_dir) / "init").as_uri()
            multiprocessing: Any = torch.multiprocessing
            multiprocessing.spawn(
                _distributed_ema_dropout_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )

    def test_empty_eval_ema_stage_does_not_join_training_collectives(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(
                input_dim=4,
                num_codebooks=2,
                codebook_size=8,
                use_ema=True,
                dropout=1.0,
            )
        )
        quantizer.quantizers[1].eval()
        active_mask = torch.tensor([[True, False]], dtype=torch.bool)
        calls: list[torch.Tensor] = []

        with (
            mock.patch.object(
                quantizer,
                "_sample_active_mask",
                return_value=active_mask,
            ),
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=lambda value, *, op: calls.append(value.clone()),
            ),
        ):
            quantizer(torch.randn(1, 4))

        self.assertEqual(len(calls), 2)

    def test_sample_active_mask_dropout_boundaries(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=4, codebook_size=8, dropout=0.0)
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

    def test_config_is_plain_data_and_dropout_has_explicit_setter(self):
        config = RVQConfig(
            vq_configs=[
                VQConfig(input_dim=4, codebook_size=8),
                VQConfig(input_dim=4, codebook_size=8),
            ],
        )
        quantizer = ResidualVectorQuantizer(config)

        self.assertIsInstance(config.vq_configs, tuple)
        config.dropout = 1.0
        self.assertEqual(config.dropout, 1.0)
        returned = quantizer.set_dropout(0.5)

        self.assertIs(returned, quantizer)
        self.assertEqual(quantizer.dropout, 0.5)
        self.assertEqual(quantizer.config.dropout, 1.0)

        with self.assertRaisesRegex(ValueError, "dropout"):
            RVQConfig(vq_configs=list(config.vq_configs), dropout=1.1)

    def test_dropout_has_no_effect_with_one_active_codebook(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )

        output = quantizer(torch.randn(8, 4), num_active_codebooks=1)

        self.assertEqual(output.indices.shape, (8, 1))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_eval_disables_dropout(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )
        quantizer.eval()

        output = quantizer(torch.randn(8, 4))

        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_backward_reaches_input(self):
        quantizer = ResidualVectorQuantizer(
            _rvq_config(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        x = torch.randn(3, 4, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            RVQConfig(vq_configs=[])
        with self.assertRaisesRegex(ValueError, "uniform codebook_dim"):
            RVQConfig(
                vq_configs=[
                    VQConfig(input_dim=4, codebook_size=8, codebook_dim=2),
                    VQConfig(input_dim=4, codebook_size=8, codebook_dim=3),
                ]
            )


if __name__ == "__main__":
    unittest.main()
