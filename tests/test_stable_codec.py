import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from anytrain._buffer import register_buffer
from anytrain.codec.stable_codec import (
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_POSTHOC_BOTTLENECK,
    DEFAULT_PRETRAINED_MODEL,
    StableCodec,
)


class StableCodecTest(unittest.TestCase):
    def test_from_pretrained_loads_default_model(self):
        with patch(
            "anytrain.codec.stable_codec.codec._load_stable_codec_model",
            return_value=FakeStableCodecBackend,
        ):
            codec = StableCodec.from_pretrained(device="cpu")

        self.assertIsInstance(codec.model, FakeStableCodecBackend)
        self.assertEqual(FakeStableCodecBackend.kwargs["pretrained_model"], DEFAULT_PRETRAINED_MODEL)
        self.assertEqual(FakeStableCodecBackend.kwargs["device"], torch.device("cpu"))
        self.assertEqual(codec.device, torch.device("cpu"))
        self.assertEqual(codec.name, "stable_codec")
        self.assertEqual(codec.sample_rate, 16000)
        self.assertEqual(DEFAULT_CODEBOOK_SIZE, 17**6)
        self.assertEqual(DEFAULT_POSTHOC_BOTTLENECK, "1x46656_400bps")
        self.assertEqual(codec.codebook_sizes, (46656,))
        self.assertEqual(codec.model.posthoc_stages, DEFAULT_POSTHOC_BOTTLENECK)

    def test_from_pretrained_sets_posthoc_bottleneck(self):
        with patch(
            "anytrain.codec.stable_codec.codec._load_stable_codec_model",
            return_value=FakeStableCodecBackend,
        ):
            codec = StableCodec.from_pretrained(
                device="cpu",
                posthoc_bottleneck="2x15625_700bps",
            )

        self.assertTrue(codec.posthoc_bottleneck)
        self.assertEqual(codec.codebook_sizes, (15625, 15625))
        self.assertEqual(codec.model.posthoc_stages, "2x15625_700bps")

    def test_from_config_loads_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            checkpoint = root / "model.ckpt"

            with patch(
                "anytrain.codec.stable_codec.codec._load_stable_codec_model",
                return_value=FakeStableCodecBackend,
            ):
                codec = StableCodec.from_config(
                    config,
                    ckpt_path=checkpoint,
                    device="cpu",
                )

        self.assertIsInstance(codec.model, FakeStableCodecBackend)
        self.assertEqual(FakeStableCodecBackend.kwargs["model_config_path"], str(config))
        self.assertEqual(FakeStableCodecBackend.kwargs["ckpt_path"], str(checkpoint))

    def test_encode_returns_tokens(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(
            model=model,
            device=torch.device("cpu"),
            posthoc_bottleneck=None,
            normalize=False,
        )
        audio = torch.zeros((2, 1, 16))

        tokens = codec.encode(audio, 16000)

        self.assertEqual(tuple(tokens.shape), (2, 5, 1))
        self.assertFalse(model.normalize)
        self.assertFalse(model.posthoc)

    def test_encode_latents_exposes_upstream_latent_boundary(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)
        audio = torch.zeros((2, 1, 16))

        latents, tokens = codec.encode_latents(audio, 16000)

        self.assertEqual(tuple(latents.shape), (2, 4, 5))
        self.assertEqual(tuple(tokens.shape), (2, 5, 1))

    def test_posthoc_encode_joins_backend_codebooks(self):
        with patch(
            "anytrain.codec.stable_codec.codec._load_stable_codec_model",
            return_value=FakeStableCodecBackend,
        ):
            codec = StableCodec.from_pretrained(
                device="cpu",
                posthoc_bottleneck="2x15625_700bps",
            )

        tokens = codec.encode(torch.zeros((2, 1, 16)), 16000)

        self.assertEqual(tuple(tokens.shape), (2, 5, 2))
        torch.testing.assert_close(tokens[..., 0], torch.zeros((2, 5), dtype=torch.long))
        torch.testing.assert_close(tokens[..., 1], torch.ones((2, 5), dtype=torch.long))

    def test_posthoc_encode_rejects_non_tensor_codebook(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        model.set_posthoc_bottleneck("1x46656_400bps")
        model.posthoc_tokens = [None]
        codec = StableCodec(
            model=model,
            device=torch.device("cpu"),
            posthoc_bottleneck="1x46656_400bps",
        )

        with self.assertRaisesRegex(TypeError, "list of Tensors"):
            codec.encode(torch.zeros((2, 1, 16)), 16000)

    def test_posthoc_encode_rejects_codebook_axis_larger_than_one(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        model.set_posthoc_bottleneck("1x46656_400bps")
        model.posthoc_tokens = [torch.zeros((2, 5, 2), dtype=torch.long)]
        codec = StableCodec(
            model=model,
            device=torch.device("cpu"),
            posthoc_bottleneck="1x46656_400bps",
        )

        with self.assertRaisesRegex(ValueError, r"\[batch, time, 1\]"):
            codec.encode(torch.zeros((2, 1, 16)), 16000)

    def test_posthoc_encode_rejects_misaligned_codebooks(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        model.set_posthoc_bottleneck("2x15625_700bps")
        model.posthoc_tokens = [
            torch.zeros((2, 5, 1), dtype=torch.long),
            torch.zeros((2, 4, 1), dtype=torch.long),
        ]
        codec = StableCodec(
            model=model,
            device=torch.device("cpu"),
            posthoc_bottleneck="2x15625_700bps",
        )

        with self.assertRaisesRegex(ValueError, "align on batch and time"):
            codec.encode(torch.zeros((2, 1, 16)), 16000)

    def test_encode_resamples_from_input_sample_rate(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)
        audio = torch.zeros((2, 1, 8))

        with patch(
            "anytrain.codec.stable_codec.codec.resample",
            return_value=audio,
        ) as resample:
            codec.encode(audio, 8000)

        resample.assert_called_once_with(audio, 8000, 16000)

    def test_frame_rate_reads_nested_downsampling_ratio(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        model.model.downsampling_ratio = 640
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)

        self.assertEqual(codec.frame_rate, 25.0)

    def test_decode_uses_tokens(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)
        tokens = torch.ones((2, 5, 1), dtype=torch.long)

        audio = codec.decode(tokens)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertFalse(model.posthoc)
        self.assertIs(model.decoded_tokens, tokens)

    def test_posthoc_decode_splits_public_codebook_axis(self):
        with patch(
            "anytrain.codec.stable_codec.codec._load_stable_codec_model",
            return_value=FakeStableCodecBackend,
        ):
            codec = StableCodec.from_pretrained(
                device="cpu",
                posthoc_bottleneck="2x15625_700bps",
            )
        tokens = torch.stack(
            (
                torch.zeros((2, 5), dtype=torch.long),
                torch.ones((2, 5), dtype=torch.long),
            ),
            dim=-1,
        )

        audio = codec.decode(tokens)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertIsInstance(codec.model.decoded_tokens, list)
        self.assertEqual(len(codec.model.decoded_tokens), 2)
        torch.testing.assert_close(codec.model.decoded_tokens[0], tokens[..., :1])
        torch.testing.assert_close(codec.model.decoded_tokens[1], tokens[..., 1:])

    def test_reconstruct_roundtrips_tokens(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)
        audio = torch.zeros((2, 1, 16))

        reconstructed = codec.reconstruct(audio, 16000)

        self.assertEqual(tuple(reconstructed.shape), (2, 1, 16))
        self.assertTrue(model.decode_called)

    def test_encode_rejects_non_mono_audio(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)

        with self.assertRaisesRegex(ValueError, "mono"):
            codec.encode(torch.zeros((2, 2, 16)), 16000)

    def test_encode_rejects_missing_channel_axis(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)

        with self.assertRaisesRegex(ValueError, "shape"):
            codec.encode(torch.zeros((2, 16)), 16000)

    def test_codebook_sizes_reads_nested_backend_bottleneck(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        model.model.bottleneck.quantizer.codebook_size = 123
        model.model.bottleneck.quantizer.num_codebooks = 2

        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)

        self.assertEqual(codec.codebook_sizes, (123, 123))

    def test_to_moves_model_and_public_device_together(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"), posthoc_bottleneck=None)

        codec.to("meta")

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)
        self.assertNotIn("_device", codec.state_dict())

    def test_constructor_moves_backend_to_requested_device(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("meta"), posthoc_bottleneck=None)

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)


class FakeStableCodecBackend(nn.Module):
    kwargs: dict[str, object] = {}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        type(self).kwargs = kwargs
        self.sample_rate = 16000
        self.posthoc_stages = None
        self.posthoc = False
        self.normalize = True
        self.decode_called = False
        self.decoded_tokens = None
        self.posthoc_tokens = None
        self.model = FakeAudioAutoencoder()

    def set_posthoc_bottleneck(self, stages):
        self.posthoc_stages = stages

    def encode(self, audio: torch.Tensor, *, posthoc_bottleneck: bool, normalize: bool, **kwargs):
        self.posthoc = posthoc_bottleneck
        self.normalize = normalize
        batch = audio.size(0)
        latents = torch.ones((batch, 4, 5))
        if posthoc_bottleneck:
            if self.posthoc_tokens is not None:
                return latents, self.posthoc_tokens
            count = {
                "1x46656_400bps": 1,
                "2x15625_700bps": 2,
                "4x729_1000bps": 4,
            }[self.posthoc_stages]
            tokens = [
                torch.full((batch, 5, 1), index, dtype=torch.long)
                for index in range(count)
            ]
            return latents, tokens
        return latents, torch.ones((batch, 5, 1), dtype=torch.long)

    def decode(self, tokens, *, posthoc_bottleneck: bool, **kwargs):
        self.decode_called = True
        self.posthoc = posthoc_bottleneck
        self.decoded_tokens = tokens
        first = tokens[0] if isinstance(tokens, list) else tokens
        return torch.ones((first.size(0), 1, 16))


class FakeAudioAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bottleneck = FakeBottleneck()


class FakeBottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = FakeQuantizer()


class FakeQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codebook_size = 17**6
        self.num_codebooks = 1


if __name__ == "__main__":
    unittest.main()
