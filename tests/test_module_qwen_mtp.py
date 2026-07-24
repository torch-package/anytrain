import importlib.util
import unittest

import torch

from anytrain.module.qwen import QwenMTPCodebookPredictor, top_p_filter

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


class QwenMTPModuleTest(unittest.TestCase):
    def test_top_p_filter_keeps_at_least_one_logit(self):
        logits = torch.tensor([[3.0, 2.0, 1.0]])

        filtered = top_p_filter(logits, 0.1)

        self.assertTrue(torch.isfinite(filtered[:, 0]).all())
        self.assertFalse(torch.isfinite(filtered[:, 1:]).any())

    @unittest.skipIf(TRANSFORMERS_AVAILABLE, "transformers is installed")
    def test_predictor_reports_missing_transformers(self):
        with self.assertRaisesRegex(ImportError, "transformers"):
            QwenMTPCodebookPredictor(4, 2, (5, 7), hidden_dim=4, layers=1, heads=1)

    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is not installed")
    def test_predictor_trains_and_generates(self):
        predictor = QwenMTPCodebookPredictor(
            4,
            2,
            (5, 7),
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            mtp_layers=1,
            mtp_heads=1,
        )
        condition = torch.randn(1, 3, 4)
        target = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
        mask = torch.tensor([[True, True, False]])

        logits = predictor(condition, target, mask=mask)
        loss = sum(value[mask].float().mean() for value in logits)
        loss.backward()
        generated = predictor.generate(
            condition,
            mask=mask,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertEqual([value.shape for value in logits], [(1, 3, 5), (1, 3, 7)])
        self.assertEqual(generated.shape, (1, 3, 2))
        self.assertTrue(torch.equal(generated[:, 2], torch.zeros_like(generated[:, 2])))
        self.assertTrue(bool((generated[..., 0][mask] < 5).all()))
        self.assertTrue(bool((generated[..., 1][mask] < 7).all()))


if __name__ == "__main__":
    unittest.main()
