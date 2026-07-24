import importlib.util
import unittest

import torch

from anytrain.module.qwen import build_qwen3_mlp, make_qwen3_config, qwen3

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


class Qwen3ModuleTest(unittest.TestCase):
    def test_qwen3_helpers_are_lightweight_without_transformers(self):
        self.assertTrue(callable(make_qwen3_config))
        self.assertTrue(callable(build_qwen3_mlp))

    @unittest.skipIf(TRANSFORMERS_AVAILABLE, "transformers is installed")
    def test_qwen3_helpers_report_missing_transformers(self):
        with self.assertRaisesRegex(ImportError, "transformers"):
            qwen3.make_qwen3_config(hidden_size=8, intermediate_size=16)

        with self.assertRaisesRegex(ImportError, "transformers"):
            qwen3.require_qwen3_class("Qwen3MLP")

    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is not installed")
    def test_make_qwen3_config_uses_huggingface_config(self):
        from transformers import Qwen3Config

        config = qwen3.make_qwen3_config(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=3,
            max_position_embeddings=32,
        )

        self.assertIsInstance(config, Qwen3Config)
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.num_key_value_heads, 1)

    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is not installed")
    def test_builders_return_huggingface_modules(self):
        mlp_cls = qwen3.require_qwen3_class("Qwen3MLP")
        attention_cls = qwen3.require_qwen3_class("Qwen3Attention")
        layer_cls = qwen3.require_qwen3_class("Qwen3DecoderLayer")
        rms_norm_cls = qwen3.require_qwen3_class("Qwen3RMSNorm")

        mlp = qwen3.build_qwen3_mlp(hidden_size=8, intermediate_size=16)
        attention = qwen3.build_qwen3_attention(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        layer = qwen3.build_qwen3_decoder_layer(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
        )
        norm = qwen3.build_qwen3_rms_norm(8)

        self.assertIsInstance(mlp, mlp_cls)
        self.assertIsInstance(attention, attention_cls)
        self.assertIsInstance(layer, layer_cls)
        self.assertIsInstance(norm, rms_norm_cls)

    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is not installed")
    def test_huggingface_mlp_forward(self):
        mlp = qwen3.build_qwen3_mlp(hidden_size=8, intermediate_size=16)

        y = mlp(torch.randn(2, 5, 8))

        self.assertEqual(y.shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
