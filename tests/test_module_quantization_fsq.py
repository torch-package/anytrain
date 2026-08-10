import unittest
import warnings
from dataclasses import fields

import torch

from anytrain.module.quantization import (
    DEFAULT_FSQ_LEVELS,
    FiniteScalarQuantizer,
    FSQConfig,
    QuantizeOutput,
)
from anytrain.module.quantization.finite_scalar import default_fsq_levels


class FiniteScalarQuantizerTest(unittest.TestCase):
    def test_config_is_plain_data(self):
        config = FSQConfig(input_dim=8, levels=(5, 5))

        field_names = {field.name for field in fields(config)}

        self.assertEqual(
            field_names,
            {"input_dim", "levels", "bound_scale", "eps", "projection_bias"},
        )
        self.assertEqual(config.levels, (5, 5))
        self.assertEqual(config.bound_scale, 1.0)

    def test_even_levels_warn_about_zero_friendly_grid(self):
        with self.assertWarnsRegex(UserWarning, "preferably be odd"):
            FSQConfig(input_dim=8, levels=(4, 5))

    def test_default_levels_are_odd_without_warning(self):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            config = FSQConfig(input_dim=8)

        self.assertEqual(config.levels, DEFAULT_FSQ_LEVELS)
        self.assertEqual(records, [])
        self.assertTrue(all(level % 2 == 1 for level in config.levels))

    def test_forward_shapes_and_flat_indices(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=8, levels=(5, 5)))
        x = torch.randn(3, 7, 8)

        output = quantizer(x)

        self.assertIsInstance(output, QuantizeOutput)
        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (3, 7))
        self.assertEqual(output.codebook_vectors.shape, (3, 7, 2))
        self.assertEqual(output.latents.shape, (3, 7, 2))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue((output.indices < quantizer.codebook_size).all())
        self.assertTrue(torch.isfinite(output.quantized_latents).all())

    def test_identity_projection_when_dimensions_match(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=2, levels=(3, 5)))
        x = torch.randn(2, 2)

        output = quantizer(x)

        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertTrue(torch.allclose(output.quantized_latents, output.codebook_vectors))

    def test_indices_round_trip_through_codebook_vectors(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=6, levels=(5, 5, 3)))
        indices = torch.tensor([[0, 1, 5], [7, 74, 58]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_levels_round_trip(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=6, levels=(5, 5, 3)))
        levels = torch.tensor([[[0, 0, 0], [4, 4, 2]], [[2, 1, 1], [1, 3, 0]]])

        indices = quantizer.levels_to_indices(levels)
        round_trip = quantizer.indices_to_levels(indices)

        self.assertTrue(torch.equal(round_trip, levels))

    def test_round_to_codebook_vectors_clamps_to_valid_grid(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=4, levels=(5, 5)))
        continuous = torch.tensor([[10.0, -10.0], [0.49, 0.51]])

        rounded = quantizer.round_to_codebook_vectors(continuous)
        indices = quantizer.codebook_vectors_to_indices(rounded)

        self.assertEqual(rounded.shape, continuous.shape)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < quantizer.codebook_size).all())

    def test_level_probs_and_masks(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=4, levels=(3, 5)))
        indices = torch.tensor([[0, 3]])

        probs = quantizer.indices_to_level_probs(indices)
        mask = quantizer.level_logits_mask(torch.randn(1, 2, 5))

        self.assertEqual(probs.shape, (1, 2, 2, 5))
        self.assertEqual(mask.shape, (1, 2, 5))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(1, 2, 2)))
        self.assertFalse(mask[..., 0, 4].any())

    def test_backward_reaches_projection_when_used(self):
        quantizer = FiniteScalarQuantizer(FSQConfig(input_dim=8, levels=(5, 5)))
        x = torch.randn(3, 8, requires_grad=True)

        quantizer(x).quantized_latents.sum().backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bound_scale_reduces_tanh_saturation_for_large_values(self):
        small_scale = FiniteScalarQuantizer(FSQConfig(input_dim=1, levels=(5,)))
        large_scale = FiniteScalarQuantizer(FSQConfig(input_dim=1, levels=(5,), bound_scale=8.0))
        x_small = torch.tensor([[8.0]], requires_grad=True)
        x_large = x_small.detach().clone().requires_grad_()

        small_scale(x_small).quantized_latents.sum().backward()
        large_scale(x_large).quantized_latents.sum().backward()

        self.assertIsNotNone(x_small.grad)
        self.assertIsNotNone(x_large.grad)
        assert x_small.grad is not None
        assert x_large.grad is not None
        self.assertGreater(x_large.grad.abs().item(), x_small.grad.abs().item())

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "input_dim"):
            FSQConfig(input_dim=0)
        with self.assertRaisesRegex(ValueError, "bound_scale"):
            FSQConfig(input_dim=4, bound_scale=0)
        with self.assertRaisesRegex(ValueError, "levels"):
            FSQConfig(input_dim=4, levels=())
        with self.assertRaisesRegex(TypeError, "integer"):
            FSQConfig(input_dim=4, levels=(3.7,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "integer"):
            FSQConfig(input_dim=4, levels=(True,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, ">= 2"):
            FSQConfig(input_dim=4, levels=(1,))
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            FSQConfig(input_dim=4, levels=(torch.iinfo(torch.long).max, 2))

    def test_default_levels(self):
        self.assertEqual(default_fsq_levels(8), (7, 7, 5))
        self.assertEqual(default_fsq_levels(9), (5, 5, 5, 5))
        self.assertEqual(default_fsq_levels(10), (7, 7, 7, 3))
        self.assertEqual(default_fsq_levels(12), (7, 5, 5, 5, 5))
        self.assertEqual(default_fsq_levels(14), (11, 11, 9, 5, 3))
        self.assertEqual(default_fsq_levels(16), DEFAULT_FSQ_LEVELS)
        with self.assertRaisesRegex(ValueError, "not supported"):
            default_fsq_levels(11)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
