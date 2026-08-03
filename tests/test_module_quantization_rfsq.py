import unittest
from dataclasses import fields

import torch

from anytrain.module.quantization import (
    QuantizeOutput,
    ResidualFiniteScalarQuantizer,
    RFSQConfig,
)


class ResidualFiniteScalarQuantizerTest(unittest.TestCase):
    def test_config_is_plain_data(self):
        config = RFSQConfig(
            input_dim=6,
            levels=[5, 5],  # type: ignore[arg-type]
            scales=[1, 0.25],  # type: ignore[arg-type]
            work_dtype=torch.float64,
        )

        field_names = {field.name for field in fields(config)}

        self.assertEqual(
            field_names,
            {"input_dim", "levels", "scales", "projection_bias", "work_dtype"},
        )
        self.assertEqual(config.levels, (5, 5))
        self.assertEqual(config.scales, (1.0, 0.25))
        self.assertEqual(config.work_dtype, torch.float64)
        config.input_dim = 8
        self.assertEqual(config.input_dim, 8)

    def test_forward_shapes_and_residual_indices(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=6, levels=(5, 5, 5), scales=(1.0, 0.25))
        )
        x = torch.randn(2, 4, 6)

        output = quantizer(x)

        self.assertIsInstance(output, QuantizeOutput)
        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (2, 4, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 4, 2, 3))
        self.assertEqual(output.latents.shape, (2, 4, 3))
        self.assertIsNone(output.loss)
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue((output.indices < quantizer.codebook_size).all())

    def test_identity_projection_when_dimensions_match(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=2, levels=(3, 5), scales=(1.0, 0.25))
        )
        x = torch.randn(3, 2)

        output = quantizer(x)

        self.assertTrue(isinstance(quantizer.project_in, torch.nn.Identity))
        self.assertTrue(isinstance(quantizer.project_out, torch.nn.Identity))
        self.assertTrue(torch.allclose(output.quantized_latents, output.codebook_vectors.sum(dim=-2)))

    def test_matches_reference_residual_fsq_math(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=4, levels=(5, 5), scales=(1.0, 0.25))
        )
        x = torch.randn(3, 4)

        output = quantizer(x)
        expected_vectors, expected_indices = _reference_residual_fsq(quantizer, x)

        self.assertTrue(torch.allclose(output.codebook_vectors, expected_vectors))
        self.assertTrue(torch.equal(output.indices, expected_indices))
        self.assertTrue(
            torch.allclose(
                output.quantized_latents,
                quantizer.project_out(expected_vectors.sum(dim=-2)),
            )
        )

    def test_residual_math_uses_configured_or_automatic_work_dtype(self):
        cases = (
            (torch.float16, None, torch.float32),
            (torch.bfloat16, None, torch.float32),
            (torch.float32, None, torch.float32),
            (torch.float64, None, torch.float64),
            (torch.float32, torch.float64, torch.float64),
            (torch.float64, torch.float32, torch.float32),
        )
        for input_dtype, configured_dtype, expected_dtype in cases:
            with self.subTest(input_dtype=input_dtype, configured_dtype=configured_dtype):
                quantizer = _RecordingRFSQ(
                    RFSQConfig(
                        input_dim=2,
                        levels=(3, 5),
                        scales=(1.0, 0.25),
                        work_dtype=configured_dtype,
                    )
                )
                x = torch.randn(3, 2, dtype=input_dtype, requires_grad=True)

                output = quantizer(x)
                output.quantized_latents.sum().backward()

                self.assertEqual(quantizer.residual_dtypes, [expected_dtype, expected_dtype])
                self.assertEqual(output.latents.dtype, input_dtype)
                self.assertEqual(output.codebook_vectors.dtype, input_dtype)
                self.assertEqual(output.quantized_latents.dtype, input_dtype)
                self.assertIsNotNone(x.grad)
                assert x.grad is not None
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_projection_outputs_follow_cpu_bfloat16_autocast(self):
        quantizer = _RecordingRFSQ(
            RFSQConfig(input_dim=3, levels=(3, 5), scales=(1.0, 0.25))
        )
        x = torch.randn(4, 3, requires_grad=True)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = quantizer(x)
        output.quantized_latents.sum().backward()

        self.assertIsInstance(quantizer.project_in, torch.nn.Linear)
        self.assertEqual(quantizer.residual_dtypes, [torch.float32, torch.float32])
        self.assertEqual(output.latents.dtype, torch.bfloat16)
        self.assertEqual(output.codebook_vectors.dtype, torch.float32)
        self.assertEqual(output.quantized_latents.dtype, torch.bfloat16)
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_float64_work_dtype_preserves_legacy_boundary_index(self):
        x = torch.atanh(torch.tensor([[0.25]], dtype=torch.float32))
        fast = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=1, levels=(5,), scales=(1.0,))
        )
        compatible = ResidualFiniteScalarQuantizer(
            RFSQConfig(
                input_dim=1,
                levels=(5,),
                scales=(1.0,),
                work_dtype=torch.float64,
            )
        )

        fast_output = fast(x)
        compatible_output = compatible(x)

        self.assertTrue(torch.equal(fast_output.indices, torch.tensor([[2]])))
        self.assertTrue(torch.equal(compatible_output.indices, torch.tensor([[3]])))
        self.assertTrue(
            torch.equal(compatible_output.codebook_vectors, torch.tensor([[[0.5]]]))
        )

    def test_indices_round_trip_through_codebook_vectors(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=6, levels=(3, 5, 7), scales=(1.0, 0.25))
        )
        indices = torch.tensor([[[0, 1], [104, 17]], [[9, 88], [33, 12]]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)
        projected = quantizer.project_codebook_vectors(vectors)

        self.assertTrue(torch.equal(round_trip, indices))
        self.assertEqual(vectors.shape, (2, 2, 2, 3))
        self.assertEqual(projected.shape, (2, 2, 6))

    def test_non_persistent_buffers_do_not_enter_state_dict(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=2, levels=(3, 5), scales=(1.0, 0.25))
        )

        state = quantizer.state_dict()

        self.assertNotIn("_levels", state)
        self.assertNotIn("_basis", state)
        self.assertNotIn("_dtype", state)

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "input_dim"):
            RFSQConfig(input_dim=0)
        with self.assertRaisesRegex(ValueError, "levels"):
            RFSQConfig(input_dim=4, levels=())
        with self.assertRaisesRegex(ValueError, "scales"):
            RFSQConfig(input_dim=4, scales=())
        with self.assertRaisesRegex(TypeError, "real"):
            RFSQConfig(input_dim=4, scales=(True,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "positive"):
            RFSQConfig(input_dim=4, scales=(0.0,))
        with self.assertRaisesRegex(TypeError, "torch.dtype"):
            RFSQConfig(input_dim=4, work_dtype="float64")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "torch.float32, torch.float64"):
            RFSQConfig(input_dim=4, work_dtype=torch.float16)

    def test_invalid_indices_fail_clearly(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=2, levels=(3, 5), scales=(1.0, 0.25))
        )
        with self.assertRaisesRegex(TypeError, "integer"):
            quantizer.indices_to_codebook_vectors(torch.tensor([[True, False]]))
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            quantizer.indices_to_codebook_vectors(torch.tensor([[0, quantizer.codebook_size]]))
        with self.assertRaisesRegex(ValueError, "active dimension"):
            quantizer.indices_to_codebook_vectors(torch.zeros(1, 3, dtype=torch.long))

    def test_quantize_clamps_levels_outside_scale_range(self):
        quantizer = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=1, levels=(5,), scales=(0.25,))
        )
        # tanh saturates near ±1, which is outside the scale=0.25 grid.
        x = torch.tensor([[10.0], [100.0], [-10.0]])

        output = quantizer(x)

        self.assertEqual(quantizer.codebook_size, 5)
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue((output.indices < quantizer.codebook_size).all())
        self.assertTrue(torch.equal(output.indices.view(-1), torch.tensor([4, 4, 0])))


class _RecordingRFSQ(ResidualFiniteScalarQuantizer):
    def __init__(self, config: RFSQConfig) -> None:
        super().__init__(config)
        self.residual_dtypes: list[torch.dtype] = []

    def _quantize_residual_stage(
        self,
        residual: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.residual_dtypes.append(residual.dtype)
        return super()._quantize_residual_stage(residual, scale)


def _reference_residual_fsq(
    quantizer: ResidualFiniteScalarQuantizer,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = quantizer.project_in(latents)
    work_dtype = quantizer.config.work_dtype
    if work_dtype is None:
        work_dtype = torch.float64 if projected.dtype == torch.float64 else torch.float32
    residual = torch.tanh(projected.to(work_dtype))
    levels = torch.tensor(quantizer.config.levels, dtype=torch.long, device=latents.device)
    basis = torch.cumprod(
        torch.tensor((1, *quantizer.config.levels[:-1]), dtype=torch.long, device=latents.device),
        dim=0,
    )
    vectors = []
    indices = []
    for scale in quantizer.config.scales:
        step = (2 * scale / (levels - 1)).to(device=latents.device, dtype=residual.dtype)
        raw_levels = (residual + scale) / step
        max_level = (levels - 1).to(dtype=raw_levels.dtype)
        bounded = torch.minimum(raw_levels.clamp_min(0), max_level)
        rounded = bounded + (bounded.round() - bounded).detach()
        vector = rounded * step - scale
        index = (rounded.long() * basis).sum(dim=-1)
        vectors.append(vector)
        indices.append(index)
        residual = residual - vector
    return torch.stack(vectors, dim=-2).to(dtype=latents.dtype), torch.stack(indices, dim=-1)


if __name__ == "__main__":
    unittest.main()
