import unittest
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from anytrain._compat import strict_zip
from anytrain.module.quantization import (
    ResidualVectorQuantizer,
    RVQConfig,
    VQConfig,
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
            RVQConfig.from_kwargs(
                input_dim=4,
                num_codebooks=3,
                codebook_size=8,
                codebook_dim=2,
                use_ema=True,
                dropout=1.0,
            )
        )
        distributed_quantizer = torch.nn.parallel.DistributedDataParallel(
            quantizer,
            find_unused_parameters=True,
        )
        before = [
            (book._ema_counts.clone(), book._ema_sums.clone(), book.codebook.weight.clone())
            for book in quantizer.quantizers
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
        for index, book in enumerate(quantizer.quantizers):
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
    def test_forward_shapes(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=8, num_codebooks=3, codebook_size=16, codebook_dim=8)
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
            RVQConfig.from_kwargs(input_dim=8, num_codebooks=4, codebook_size=16)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8), num_active_codebooks=2)

        self.assertEqual(output.indices.shape, (2, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 2, 8))
        self.assertTrue(output.active_codebook_mask.all())

    def test_indices_round_trip(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[[0, 1], [2, 3]]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_project_codebook_vectors_sums_projected_books(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[0, 1], [2, 3]])
        vectors = quantizer.indices_to_codebook_vectors(indices)

        projected = quantizer.project_codebook_vectors(vectors)

        self.assertEqual(projected.shape, (2, 4))

    def test_latents_to_codebook_vectors_has_no_training_side_effects(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(
                input_dim=4,
                num_codebooks=3,
                codebook_size=8,
                use_ema=True,
                dropout=1.0,
            )
        )
        before_counts = [book.ema_counts.clone() for book in quantizer.quantizers]

        vectors = quantizer.latents_to_codebook_vectors(torch.randn(16, 4))

        self.assertEqual(vectors.shape, (16, 3, 4))
        for book, before in strict_zip(quantizer.quantizers, before_counts):
            self.assertTrue(torch.equal(book.ema_counts, before))

    def test_dropout_marks_inactive_codebooks_during_training(self):
        torch.manual_seed(0)
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )

        output = quantizer(torch.randn(64, 4))

        self.assertTrue((output.indices == -1).any())
        self.assertFalse(output.active_codebook_mask.all())

    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(),
        "Gloo distributed backend is unavailable",
    )
    def test_distributed_ema_dropout_keeps_collectives_aligned(self):
        with TemporaryDirectory() as tmp_dir:
            init_method = (Path(tmp_dir) / "init").as_uri()
            torch.multiprocessing.spawn(
                _distributed_ema_dropout_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )

    def test_empty_eval_ema_stage_does_not_join_training_collectives(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(
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
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=0.0)
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
        config = RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        quantizer = ResidualVectorQuantizer(config)

        with self.assertRaises(FrozenInstanceError):
            config.dropout = 1.0
        returned = quantizer.set_dropout(0.5)

        self.assertIs(returned, quantizer)
        self.assertEqual(quantizer.dropout, 0.5)
        self.assertIsNone(quantizer.config.dropout)

    def test_dropout_has_no_effect_with_one_active_codebook(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )

        output = quantizer(torch.randn(8, 4), num_active_codebooks=1)

        self.assertEqual(output.indices.shape, (8, 1))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_eval_disables_dropout(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )
        quantizer.eval()

        output = quantizer(torch.randn(8, 4))

        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_backward_reaches_input(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        x = torch.randn(3, 4, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
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
