import importlib.util
import unittest
from collections import OrderedDict
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

from anytrain.module.qwen import QwenMTPCodebookPredictor
from anytrain.utils import top_p_filter


def SimpleNamespace(**kwargs: object) -> Any:
    return _SimpleNamespace(**kwargs)


TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


class _FakeQwen3Model(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(1, hidden_size)

    def forward(self, *, inputs_embeds, use_cache=False, **kwargs):
        del kwargs
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=() if use_cache else None,
        )


class QwenMTPModuleTest(unittest.TestCase):
    def test_top_p_filter_keeps_at_least_one_logit(self):
        logits = torch.tensor([[3.0, 2.0, 1.0]])

        filtered = top_p_filter(logits, 0.1)

        self.assertTrue(torch.isfinite(filtered[:, 0]).all())
        self.assertFalse(torch.isfinite(filtered[:, 1:]).any())

    def test_top_p_filter_validates_public_contract(self):
        invalid_calls = (
            lambda: top_p_filter(torch.tensor([1, 2]), 0.5),
            lambda: top_p_filter(torch.empty(1, 0), 0.5),
            lambda: top_p_filter(torch.ones(2), cast(Any, True)),
            lambda: top_p_filter(torch.ones(2), float("nan")),
            lambda: top_p_filter(torch.ones(2), 0.0),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call), self.assertRaises((TypeError, ValueError)):
                invalid_call()

    @unittest.skipIf(TRANSFORMERS_AVAILABLE, "transformers is installed")
    def test_predictor_reports_missing_transformers(self):
        with self.assertRaisesRegex(ImportError, "transformers"):
            QwenMTPCodebookPredictor(4, 2, (5, 7), hidden_dim=4, layers=1, heads=1)

    def test_predictor_registers_only_used_residual_parameters(self):
        with patch(
            "anytrain.module.qwen.mtp.build_qwen3_model",
            side_effect=lambda **kwargs: _FakeQwen3Model(kwargs["hidden_size"]),
        ):
            predictor = QwenMTPCodebookPredictor(
                4,
                3,
                (5, 7, 9),
                codebook_embeddings=tuple(torch.randn(size, 2) for size in (5, 7, 9)),
                hidden_dim=4,
                layers=1,
                heads=1,
                ffn_ratio=2,
                mtp_layers=1,
                mtp_heads=1,
            )

        self.assertEqual(len(predictor.codebook_embeddings), 2)
        self.assertEqual(len(predictor.embedding_projections), 2)
        self.assertEqual(tuple(predictor.codebook_bos.shape), (2, 4))
        self.assertNotIn("codebook_embeddings.2.weight", predictor.state_dict())
        self.assertNotIn("embedding_projections.2.weight", predictor.state_dict())

    def test_predictor_loads_legacy_dead_parameters(self):
        with patch(
            "anytrain.module.qwen.mtp.build_qwen3_model",
            side_effect=lambda **kwargs: _FakeQwen3Model(kwargs["hidden_size"]),
        ):
            predictor = QwenMTPCodebookPredictor(
                4,
                3,
                (5, 7, 9),
                codebook_embeddings=tuple(torch.randn(size, 2) for size in (5, 7, 9)),
                hidden_dim=4,
                layers=1,
                heads=1,
                ffn_ratio=2,
                mtp_layers=1,
                mtp_heads=1,
            )

        state = predictor.state_dict()
        legacy = OrderedDict((key, value.clone()) for key, value in state.items())
        legacy["codebook_bos"] = torch.cat(
            (torch.full((1, 4), 11.0), state["codebook_bos"]),
            dim=0,
        )
        legacy["codebook_embeddings.2.weight"] = torch.randn(9, 2)
        legacy["embedding_projections.2.weight"] = torch.randn(4, 2)
        legacy["embedding_projections.2.bias"] = torch.randn(4)

        result = predictor.load_state_dict(legacy, strict=True)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        torch.testing.assert_close(predictor.codebook_bos, state["codebook_bos"])

    def test_single_codebook_keeps_temporal_embedding(self):
        with patch(
            "anytrain.module.qwen.mtp.build_qwen3_model",
            side_effect=lambda **kwargs: _FakeQwen3Model(kwargs["hidden_size"]),
        ) as build:
            predictor = QwenMTPCodebookPredictor(
                4,
                1,
                5,
                hidden_dim=4,
                layers=1,
                heads=1,
                ffn_ratio=2,
                mtp_layers=1,
                mtp_heads=1,
            )

        self.assertEqual(len(predictor.codebook_embeddings), 1)
        self.assertEqual(len(predictor.embedding_projections), 1)
        self.assertEqual(tuple(predictor.codebook_bos.shape), (0, 4))
        self.assertEqual(build.call_count, 1)
        self.assertIsNone(predictor.mtp)
        self.assertFalse(any(key.startswith("mtp.") for key in predictor.state_dict()))
        logits = predictor(
            torch.randn(1, 2, 4),
            torch.tensor([[[1], [2]]]),
            validate=False,
        )
        self.assertEqual([value.shape for value in logits], [(1, 2, 5)])

    def test_single_codebook_loads_legacy_mtp_parameters(self):
        with patch(
            "anytrain.module.qwen.mtp.build_qwen3_model",
            side_effect=lambda **kwargs: _FakeQwen3Model(kwargs["hidden_size"]),
        ):
            predictor = QwenMTPCodebookPredictor(
                4,
                1,
                5,
                hidden_dim=4,
                layers=1,
                heads=1,
                ffn_ratio=2,
                mtp_layers=1,
                mtp_heads=1,
            )

        legacy = OrderedDict((key, value.clone()) for key, value in predictor.state_dict().items())
        legacy["mtp.embed_tokens.weight"] = torch.randn(1, 4)
        result = predictor.load_state_dict(legacy, strict=True)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

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
        loss = torch.stack([value[mask].float().mean() for value in logits]).sum()
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
