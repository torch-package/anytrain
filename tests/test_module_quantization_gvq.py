import unittest
from dataclasses import fields
from unittest import mock

import torch

from anytrain.module.quantization import (
    GroupedVectorQuantizer,
    GVQConfig,
    QuantizationLoss,
)


class GroupedVectorQuantizerTest(unittest.TestCase):
    def test_config_is_plain_data(self):
        config = GVQConfig(input_dim=8, group_sizes=(90, 90), codebook_dim=4)

        field_names = {field.name for field in fields(config)}

        self.assertIn("group_sizes", field_names)
        self.assertEqual(config.group_dims, (2, 2))

    def test_forward_shapes_and_product_codebook_size(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=8, group_sizes=(90, 90), codebook_dim=4)
        )
        x = torch.randn(3, 5, 8)

        output = quantizer(x)

        self.assertEqual(quantizer.codebook_size, 8100)
        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (3, 5))
        self.assertEqual(output.codebook_vectors.shape, (3, 5, 4))
        self.assertEqual(output.latents.shape, (3, 5, 4))
        self.assertIsInstance(output.loss, QuantizationLoss)
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue((output.indices < 8100).all())

    def test_builtin_forward_and_vector_encoding_do_not_synchronize_for_validation(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )
        latents = torch.randn(3, 4)

        with mock.patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("internal group indices must not be read on the host"),
        ):
            output = quantizer(latents)
            indices = quantizer.codebook_vectors_to_indices(output.codebook_vectors)

        self.assertTrue(torch.equal(indices, output.indices))

    def test_indices_decode_does_not_repeat_group_validation(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )

        with mock.patch.object(
            quantizer,
            "_validate_group_indices",
            side_effect=AssertionError("flat indices were already validated"),
        ):
            vectors = quantizer.indices_to_codebook_vectors(torch.tensor([0, 7, 14]))

        self.assertEqual(vectors.shape, (3, 4))

    def test_public_group_index_conversion_remains_strict(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )

        with self.assertRaisesRegex(ValueError, r"group_indices\[\.\.\., 1\]"):
            quantizer.group_indices_to_indices(torch.tensor([[0, 5]]))

    def test_forward_preserves_group_index_conversion_override(self):
        class CustomGroupedVectorQuantizer(GroupedVectorQuantizer):
            def __init__(self, config: GVQConfig) -> None:
                super().__init__(config)
                self.encode_calls = 0

            def group_indices_to_indices(self, group_indices: torch.Tensor) -> torch.Tensor:
                self.encode_calls += 1
                return super().group_indices_to_indices(group_indices)

        quantizer = CustomGroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )

        quantizer(torch.randn(3, 4))

        self.assertEqual(quantizer.encode_calls, 1)

    def test_forward_preserves_instance_group_index_conversion_override(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )

        with mock.patch.object(
            quantizer,
            "group_indices_to_indices",
            wraps=quantizer.group_indices_to_indices,
        ) as encode:
            quantizer(torch.randn(3, 4))

        encode.assert_called_once()

    def test_group_indices_round_trip(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )
        group_indices = torch.tensor([[[0, 0], [2, 4]], [[1, 3], [0, 2]]])

        indices = quantizer.group_indices_to_indices(group_indices)
        round_trip = quantizer.indices_to_group_indices(indices)

        self.assertTrue(torch.equal(round_trip, group_indices))

    def test_indices_round_trip_through_codebook_vectors(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=4, group_sizes=(3, 5), group_dims=(2, 2))
        )
        indices = torch.tensor([[0, 1, 7], [14, 3, 11]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_backward_reaches_input_and_codebooks(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(input_dim=8, group_sizes=(7, 9), codebook_dim=4)
        )
        x = torch.randn(3, 8, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())
        for codebook in quantizer.codebooks:
            self.assertIsNotNone(codebook.grad)
            assert codebook.grad is not None
            self.assertTrue(torch.isfinite(codebook.grad).all())

    def test_non_normalized_group_lookup(self):
        quantizer = GroupedVectorQuantizer(
            GVQConfig(
                input_dim=4,
                group_sizes=(2, 2),
                group_dims=(2, 2),
                normalize_latents=False,
            )
        )
        with torch.no_grad():
            quantizer.codebooks[0].copy_(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))
            quantizer.codebooks[1].copy_(torch.tensor([[0.0, 0.0], [0.0, 10.0]]))

        output = quantizer(torch.tensor([[9.0, 0.0, 0.0, 9.0], [1.0, 0.0, 0.0, 1.0]]))

        self.assertTrue(torch.equal(output.indices, torch.tensor([3, 0])))
        self.assertTrue(
            torch.equal(
                quantizer.indices_to_group_indices(output.indices), torch.tensor([[1, 1], [0, 0]])
            )
        )

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "group_sizes"):
            GVQConfig(input_dim=4, group_sizes=())
        with self.assertRaisesRegex(ValueError, "same length"):
            GVQConfig(input_dim=4, group_sizes=(2, 2), group_dims=(4,))
        with self.assertRaisesRegex(ValueError, "sum"):
            GVQConfig(input_dim=4, group_sizes=(2, 2), group_dims=(3, 3))


if __name__ == "__main__":
    unittest.main()
