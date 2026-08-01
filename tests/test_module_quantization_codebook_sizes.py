import unittest

from anytrain.module.quantization import (
    AGRVQConfig,
    AutoGroupResidualVectorQuantizer,
    EmbeddingVectorQuantizer,
    FiniteScalarQuantizer,
    FSQConfig,
    GroupedVectorQuantizer,
    GVQConfig,
    ResidualFiniteScalarQuantizer,
    ResidualVectorQuantizer,
    RFSQConfig,
    RVQConfig,
    VQConfig,
)


class QuantizationCodebookSizesTest(unittest.TestCase):
    def test_single_codebook_quantizers_expose_codebook_sizes_tuple(self):
        cases = (
            (EmbeddingVectorQuantizer(VQConfig(input_dim=4, codebook_size=8)), (8,)),
            (FiniteScalarQuantizer(FSQConfig(input_dim=4, levels=(3, 5))), (15,)),
            (GroupedVectorQuantizer(GVQConfig(input_dim=4, group_sizes=(3, 5))), (15,)),
        )

        for quantizer, expected in cases:
            with self.subTest(type=type(quantizer).__name__):
                self.assertEqual(quantizer.codebook_sizes, expected)

    def test_residual_quantizers_expose_public_codebook_size_per_stage(self):
        rvq = ResidualVectorQuantizer(
            RVQConfig(vq_configs=tuple(VQConfig(input_dim=4, codebook_size=8) for _ in range(3)))
        )
        rfsq = ResidualFiniteScalarQuantizer(
            RFSQConfig(input_dim=4, levels=(3, 3), scales=(1.0, 0.5))
        )
        agrvq = AutoGroupResidualVectorQuantizer(
            AGRVQConfig(input_dim=4, num_codebooks=3, group_size=5, codebook_dim=2)
        )

        self.assertEqual(rvq.codebook_sizes, (8, 8, 8))
        self.assertEqual(rfsq.codebook_sizes, (9, 9))
        self.assertEqual(agrvq.codebook_sizes, (25, 25, 25))

    def test_agrvq_config_uses_group_size_without_flat_size_alias(self):
        config = AGRVQConfig(input_dim=6, num_codebooks=2, group_size=5, num_groups=3)

        self.assertEqual(config.group_size, 5)
        self.assertFalse(hasattr(config, "codebook_size"))
        self.assertFalse(hasattr(config, "flat_codebook_size"))


if __name__ == "__main__":
    unittest.main()
