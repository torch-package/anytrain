from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from anytrain._buffer import register_buffer
from anytrain.audio.codec.unicodec.assets import UniCodecAssets
from anytrain.audio.codec.unicodec.codec import UniCodec


class UniCodecCodecTest(unittest.TestCase):
    def test_from_pretrained_loads_model_from_resolved_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets: UniCodecAssets = {
                "cache_dir": root,
                "config": root / "config.yaml",
                "checkpoint": root / "unicode.ckpt",
            }

            with (
                patch(
                    "anytrain.audio.codec.unicodec.codec.ensure_unicodec_assets",
                    return_value=assets,
                ),
                patch(
                    "anytrain.audio.codec.unicodec.codec._load_unicodec_model",
                    return_value=FakeUniCodecFactory,
                ),
            ):
                codec = UniCodec.from_pretrained(device="cpu")

            self.assertIsInstance(codec.model, FakeUniCodecModel)
            self.assertEqual(FakeUniCodecFactory.config_path, str(assets["config"]))
            self.assertEqual(FakeUniCodecFactory.model_path, str(assets["checkpoint"]))
            self.assertEqual(codec.device, torch.device("cpu"))
            self.assertEqual(codec.assets, assets)
            self.assertEqual(codec.name, "unicodec")
            self.assertEqual(codec.sample_rate, 24000)
            self.assertEqual(codec.codebook_sizes, (16384,))
            self.assertTrue(codec.model.eval_called)

    def test_encode_returns_codes_and_passes_domain_contract(self):
        model = FakeUniCodecModel()
        codec = UniCodec(
            model=model,
            device=torch.device("cpu"),
            assets=_assets(),
            domain="0",
            bandwidth_id=3,
        )
        audio = torch.zeros((2, 1, 8))

        codes = codec.encode(audio, 24000)

        self.assertEqual(tuple(codes.shape), (2, 4, 1))
        self.assertEqual(model.domains, ("0", "0"))
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([3])))

    def test_encode_features_exposes_upstream_feature_boundary(self):
        model = FakeUniCodecModel()
        codec = UniCodec(model=model, device=torch.device("cpu"), assets=_assets())
        audio = torch.zeros((2, 1, 8))

        features, codes = codec.encode_features(audio, 24000)

        self.assertEqual(tuple(features.shape), (2, 4, 3))
        self.assertEqual(tuple(codes.shape), (2, 4, 1))
        self.assertEqual(model.domains, ("0", "0"))
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([0])))

    def test_decode_converts_codes_before_decode(self):
        model = FakeUniCodecModel()
        codec = UniCodec(
            model=model,
            device=torch.device("cpu"),
            assets=_assets(),
            bandwidth_id=2,
        )
        codes = torch.ones((2, 4, 1), dtype=torch.long)

        audio = codec.decode(codes)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertTrue(model.codes_to_features_called)
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([2])))

    def test_decode_features_uses_continuous_feature_boundary(self):
        model = FakeUniCodecModel()
        codec = UniCodec(
            model=model,
            device=torch.device("cpu"),
            assets=_assets(),
            bandwidth_id=2,
        )
        features = torch.ones((2, 4, 3))

        audio = codec.decode_features(features)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertTrue(model.decode_called)
        self.assertFalse(model.codes_to_features_called)
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([2])))

    def test_encode_rejects_unknown_domain(self):
        with self.assertRaisesRegex(ValueError, "domain"):
            UniCodec(
                model=FakeUniCodecModel(),
                device=torch.device("cpu"),
                assets=_assets(),
                domain="speech",
            )

    def test_to_moves_model_and_public_device_together(self):
        codec = UniCodec(
            model=FakeUniCodecModel(),
            device=torch.device("cpu"),
            assets=_assets(),
        )

        codec.to("meta")

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)
        self.assertNotIn("_device", codec.state_dict())

    def test_constructor_moves_backend_to_requested_device(self):
        codec = UniCodec(
            model=FakeUniCodecModel(),
            device=torch.device("meta"),
            assets=_assets(),
        )

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)


class FakeUniCodecFactory:
    config_path: str | None = None
    model_path: str | None = None

    @classmethod
    def from_pretrained0802(cls, config_path, model_path):
        cls.config_path = config_path
        cls.model_path = model_path
        return FakeUniCodecModel()


class FakeUniCodecModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        self.feature_extractor = SimpleNamespace(
            encodec=SimpleNamespace(quantizer=SimpleNamespace(n_q=1, bins=16384))
        )
        self.domains: tuple[str, ...] | None = None
        self.bandwidth_id: torch.Tensor | None = None
        self.decode_called = False
        self.codes_to_features_called = False
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()

    def encode_infer(self, audio: torch.Tensor, domain, *, bandwidth_id: torch.Tensor):
        self.domains = tuple(domain)
        self.bandwidth_id = bandwidth_id
        batch = audio.size(0)
        return torch.ones((batch, 3, 4)), torch.ones((1, batch, 4), dtype=torch.long)

    def decode(self, features: torch.Tensor, *, bandwidth_id: torch.Tensor):
        self.decode_called = True
        self.bandwidth_id = bandwidth_id
        return torch.ones((features.size(0), 16))

    def codes_to_features(self, codes: torch.Tensor):
        self.codes_to_features_called = True
        return torch.ones((codes.size(1), 3, codes.size(2)))


def _assets() -> UniCodecAssets:
    return {
        "cache_dir": Path(),
        "config": Path(),
        "checkpoint": Path(),
    }


if __name__ == "__main__":
    unittest.main()
