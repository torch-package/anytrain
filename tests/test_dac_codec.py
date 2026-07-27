from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from anytrain._buffer import register_buffer
from anytrain.codec.dac.assets import DACAssets
from anytrain.codec.dac.codec import DAC


class DACCodecTest(unittest.TestCase):
    def test_from_pretrained_loads_resolved_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "weights.pth"
            checkpoint.touch()
            assets: DACAssets = {
                "cache_dir": root,
                "checkpoint": checkpoint,
                "model_type": "44khz",
                "model_bitrate": "8kbps",
                "tag": "0.0.1",
            }

            with (
                patch(
                    "anytrain.codec.dac.codec.ensure_dac_assets",
                    return_value=assets,
                ),
                patch(
                    "anytrain.codec.dac.codec._load_dac_model",
                    return_value=FakeDACFactory,
                ),
            ):
                codec = DAC.from_pretrained(device="cpu", n_quantizers=2)

            self.assertEqual(FakeDACFactory.checkpoint, checkpoint)
            self.assertEqual(codec.checkpoint, checkpoint)
            self.assertEqual(codec.assets, assets)
            self.assertEqual(codec.name, "dac")
            self.assertEqual(codec.sample_rate, 24000)
            self.assertEqual(codec.codebook_sizes, (1024, 1024))
            self.assertTrue(codec.model.eval_called)

    def test_from_checkpoint_exposes_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            DAC.from_checkpoint("missing.pth")

    def test_encode_returns_time_major_codes_and_features(self):
        model = FakeDACModel()
        codec = _codec(model, n_quantizers=2)
        audio = torch.zeros((2, 1, 8))

        features, codes = codec.encode_features(audio, 24000)

        self.assertEqual(tuple(features.shape), (2, 3, 6))
        self.assertEqual(tuple(codes.shape), (2, 3, 2))
        self.assertTrue(torch.equal(codes[..., 0], torch.zeros((2, 3), dtype=torch.long)))
        self.assertTrue(torch.equal(codes[..., 1], torch.ones((2, 3), dtype=torch.long)))
        self.assertEqual(model.preprocess_sample_rate, 24000)
        self.assertEqual(model.encoded_quantizers, 2)

    def test_encode_resamples_input(self):
        model = FakeDACModel()
        codec = _codec(model)
        audio = torch.zeros((1, 1, 8))

        with patch(
            "anytrain.codec.dac.codec.resample",
            return_value=audio,
        ) as resample:
            codec.encode(audio, 16000)

        resample.assert_called_once_with(audio, 16000, 24000)

    def test_decode_converts_codes_to_quantized_features(self):
        model = FakeDACModel()
        codec = _codec(model, n_quantizers=2)
        codes = torch.arange(6).reshape(1, 3, 2)

        audio = codec.decode(codes)

        self.assertEqual(tuple(audio.shape), (1, 1, 12))
        self.assertTrue(torch.equal(model.quantizer.codes, codes.transpose(1, 2)))
        self.assertEqual(tuple(model.decoded_features.shape), (1, 6, 3))

    def test_n_quantizers_must_fit_loaded_model(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            _codec(FakeDACModel(), n_quantizers=5)

    def test_encode_rejects_non_mono_audio(self):
        codec = _codec(FakeDACModel())

        with self.assertRaisesRegex(ValueError, "mono"):
            codec.encode(torch.zeros((1, 2, 8)), 24000)

    def test_decode_rejects_wrong_codebook_count(self):
        codec = _codec(FakeDACModel(), n_quantizers=2)

        with self.assertRaisesRegex(ValueError, "2 aligned codebooks"):
            codec.decode(torch.zeros((1, 3, 3), dtype=torch.long))

    def test_to_moves_model_and_public_device_together(self):
        codec = _codec(FakeDACModel())

        codec.to("meta")

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)
        self.assertNotIn("_device", codec.state_dict())

    def test_constructor_moves_backend_to_requested_device(self):
        codec = DAC(
            model=FakeDACModel(),
            device=torch.device("meta"),
            checkpoint=Path("weights.pth"),
        )

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)

    def test_load_state_dict_rejects_assign(self):
        codec = _codec(FakeDACModel())

        with self.assertRaisesRegex(ValueError, "assign=True"):
            codec.load_state_dict(codec.state_dict(), assign=True)

        parent = nn.Module()
        parent.codec = codec
        with self.assertRaisesRegex(ValueError, "assign=True"):
            parent.load_state_dict(parent.state_dict(), assign=True)


class FakeDACFactory:
    checkpoint: Path | None = None

    @classmethod
    def load(cls, checkpoint: Path):
        cls.checkpoint = checkpoint
        return FakeDACModel()


class FakeDACQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codes: torch.Tensor | None = None

    def from_codes(self, codes: torch.Tensor):
        self.codes = codes
        batch, _, frames = codes.shape
        features = torch.ones((batch, 6, frames))
        latents = torch.ones((batch, 4, frames))
        return features, latents, codes


class FakeDACModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        self.sample_rate = 24000
        self.n_codebooks = 4
        self.codebook_size = 1024
        self.quantizer = FakeDACQuantizer()
        self.eval_called = False
        self.preprocess_sample_rate: int | None = None
        self.encoded_quantizers: int | None = None
        self.decoded_features: torch.Tensor | None = None

    def eval(self):
        self.eval_called = True
        return super().eval()

    def preprocess(self, audio: torch.Tensor, sample_rate: int):
        self.preprocess_sample_rate = sample_rate
        return audio

    def encode(self, audio: torch.Tensor, *, n_quantizers: int):
        self.encoded_quantizers = n_quantizers
        batch = audio.size(0)
        features = torch.ones((batch, 6, 3))
        codes = torch.arange(n_quantizers).reshape(1, n_quantizers, 1)
        codes = codes.expand(batch, -1, 3).contiguous()
        latents = torch.ones((batch, n_quantizers * 2, 3))
        return features, codes, latents, torch.tensor(0.0), torch.tensor(0.0)

    def decode(self, features: torch.Tensor):
        self.decoded_features = features
        return torch.ones((features.size(0), 1, features.size(-1) * 4))


def _codec(model: FakeDACModel, *, n_quantizers: int | None = None) -> DAC:
    return DAC(
        model=model,
        device=torch.device("cpu"),
        checkpoint=Path("weights.pth"),
        n_quantizers=n_quantizers,
    )


if __name__ == "__main__":
    unittest.main()
