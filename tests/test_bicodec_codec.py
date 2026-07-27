from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from anytrain._buffer import register_buffer
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.codec.bicodec.assets import BiCodecAssets
from anytrain.codec.bicodec.codec import BICODEC_INSTALL_COMMAND, BiCodec


class BiCodecCodecTest(unittest.TestCase):
    def test_install_command_is_copyable_and_uses_the_project_fork(self):
        self.assertEqual(
            BICODEC_INSTALL_COMMAND,
            'python -m pip install "spark-tts @ '
            "git+https://github.com/rookiejune/Spark-TTS.git@"
            'cf5302b0da797aeffc18af172de4dda0bb6ab718"',
        )
        subprocess.run(
            ["/bin/sh", "-n"],
            input=BICODEC_INSTALL_COMMAND,
            text=True,
            check=True,
        )

    def test_missing_spark_tts_error_includes_the_install_command(self):
        with (
            patch.dict(sys.modules, {"sparktts": None}),
            patch("anytrain.codec.bicodec.codec.ensure_bicodec_assets") as assets,
            self.assertRaises(ImportError) as raised,
        ):
            BiCodec.from_pretrained(device="cpu")

        self.assertTrue(str(raised.exception).endswith(BICODEC_INSTALL_COMMAND))
        assets.assert_not_called()

    def test_from_pretrained_loads_resolved_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets: BiCodecAssets = {
                "cache_dir": root,
                "model_dir": root,
            }

            with (
                patch(
                    "anytrain.codec.bicodec.codec.ensure_bicodec_assets",
                    return_value=assets,
                ),
                patch(
                    "anytrain.codec.bicodec.codec._load_bicodec_model",
                    return_value=FakeBiCodecFactory,
                ),
                patch(
                    "anytrain.codec.bicodec.codec._load_wav2vec2_classes",
                    return_value=(FakeProcessor, FakeFeatureExtractor),
                ),
                patch(
                    "anytrain.codec.bicodec.codec._load_config",
                    return_value={
                        "sample_rate": 16000,
                        "ref_segment_duration": 0.5,
                        "latent_hop_length": 320,
                    },
                ),
            ):
                codec = BiCodec.from_pretrained(device="cpu")

        self.assertIsInstance(codec.model, FakeBiCodecModel)
        self.assertIsInstance(codec.processor, FakeProcessor)
        self.assertIsInstance(codec.feature_extractor, FakeFeatureExtractor)
        self.assertEqual(FakeBiCodecFactory.model_dir, root / "BiCodec")
        self.assertEqual(FakeProcessor.path, str(root / "wav2vec2-large-xlsr-53"))
        self.assertEqual(FakeFeatureExtractor.path, str(root / "wav2vec2-large-xlsr-53"))
        self.assertEqual(codec.assets, assets)
        self.assertEqual(codec.device, torch.device("cpu"))
        self.assertEqual(codec.sample_rate, 16000)
        self.assertEqual(codec.ref_segment_length, 8000)
        self.assertEqual(codec.name, "bicodec")
        self.assertEqual(codec.acoustic_layout, AcousticLayout.FIXED_LENGTH)
        self.assertEqual(codec.semantic_codebook_sizes, (4096,))
        self.assertEqual(codec.acoustic_codebook_sizes, (8192,))
        self.assertTrue(codec.feature_extractor.config.output_hidden_states)
        self.assertTrue(codec.model.eval_called)

    def test_tokenize_returns_canonical_semantic_and_acoustic_codes(self):
        model = FakeBiCodecModel()
        processor = FakeProcessor()
        feature_extractor = FakeFeatureExtractor()
        codec = BiCodec(
            model=model,
            processor=processor,
            feature_extractor=feature_extractor,
            device=torch.device("cpu"),
            config={
                "sample_rate": 16000,
                "ref_segment_duration": 0.5,
                "latent_hop_length": 320,
            },
        )
        audio = torch.zeros((2, 1, 4000))

        codes = codec.tokenize(audio, 16000)

        self.assertIsInstance(codes, SemanticAcousticCodes)
        self.assertEqual(tuple(codes.semantic.shape), (2, 5, 1))
        self.assertEqual(tuple(codes.acoustic.shape), (2, 3, 1))
        expected = torch.arange(6, dtype=torch.long).reshape(2, 1, 3).transpose(1, 2)
        self.assertTrue(torch.equal(codes.acoustic, expected))
        self.assertEqual(tuple(model.batch["wav"].shape), (2, 4000))
        self.assertEqual(tuple(model.batch["ref_wav"].shape), (2, 8000))
        self.assertEqual(tuple(model.batch["feat"].shape), (2, 4, 3))
        self.assertEqual(processor.sample_rate, 16000)

    def test_bicodec_exposes_only_structured_token_methods(self):
        codec = _codec(FakeBiCodecModel())

        self.assertTrue(hasattr(codec, "tokenize"))
        self.assertTrue(hasattr(codec, "detokenize"))
        self.assertFalse(hasattr(codec, "encode"))
        self.assertFalse(hasattr(codec, "decode"))
        self.assertFalse(hasattr(codec, "global_codebook_sizes"))

    def test_tokenize_accepts_separate_reference_audio(self):
        model = FakeBiCodecModel()
        codec = _codec(model)
        audio = torch.zeros((2, 1, 4000))
        ref_audio = torch.ones((2, 1, 2000))

        with patch(
            "anytrain.codec.bicodec.codec.resample",
            side_effect=[audio, ref_audio],
        ) as resample:
            codec.tokenize(
                audio,
                8000,
                ref_audio=ref_audio,
                ref_sample_rate=24000,
            )

        self.assertEqual(resample.call_args_list[0].args, (audio, 8000, 16000))
        self.assertEqual(resample.call_args_list[1].args, (ref_audio, 24000, 16000))
        self.assertEqual(tuple(model.batch["ref_wav"].shape), (2, 8000))
        self.assertTrue(torch.equal(model.batch["ref_wav"][:, :2000], torch.ones((2, 2000))))

    def test_extract_features_rejects_missing_hidden_states(self):
        feature_extractor = FakeFeatureExtractor()
        feature_extractor.hidden_states = None
        codec = BiCodec(
            model=FakeBiCodecModel(),
            processor=FakeProcessor(),
            feature_extractor=feature_extractor,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(ValueError, "hidden states"):
            codec.extract_features(torch.zeros((1, 16)))

    def test_detokenize_restores_backend_semantic_and_acoustic_layouts(self):
        model = FakeBiCodecModel()
        codec = _codec(model)
        codes = SemanticAcousticCodes(
            semantic=torch.arange(10, dtype=torch.long).reshape(2, 5, 1),
            acoustic=torch.arange(6, dtype=torch.long).reshape(2, 3, 1),
        )

        audio = codec.detokenize(codes)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertTrue(torch.equal(model.semantic, codes.semantic[..., 0]))
        self.assertTrue(torch.equal(model.acoustic_tokens, codes.acoustic.transpose(1, 2)))

    def test_detokenize_adds_channel_axis_for_backend_2d_audio(self):
        model = FakeBiCodecModel()
        model.detokenized_audio = torch.ones((2, 16))
        codec = _codec(model)

        audio = codec.detokenize(
            SemanticAcousticCodes(
                semantic=torch.ones((2, 5, 1), dtype=torch.long),
                acoustic=torch.ones((2, 3, 1), dtype=torch.long),
            )
        )

        self.assertEqual(tuple(audio.shape), (2, 1, 16))

    def test_reconstruct_roundtrips_tokens(self):
        model = FakeBiCodecModel()
        codec = _codec(model)

        audio = codec.reconstruct(torch.zeros((2, 1, 4000)), 16000)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertIsNotNone(model.semantic)
        self.assertIsNotNone(model.acoustic_tokens)

    def test_tokenize_rejects_non_mono_audio(self):
        codec = _codec(FakeBiCodecModel())

        with self.assertRaisesRegex(ValueError, "mono"):
            codec.tokenize(torch.zeros((2, 2, 16)), 16000)

    def test_detokenize_rejects_batch_mismatch(self):
        codec = _codec(FakeBiCodecModel())

        with self.assertRaisesRegex(ValueError, "align on batch"):
            codec.detokenize(
                SemanticAcousticCodes(
                    semantic=torch.ones((2, 5, 1), dtype=torch.long),
                    acoustic=torch.ones((1, 3, 1), dtype=torch.long),
                )
            )

    def test_to_moves_model_feature_extractor_and_public_device_together(self):
        codec = _codec(FakeBiCodecModel())

        codec.to("meta")

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.model.device_probe.device, codec.device)
        self.assertEqual(codec.feature_extractor.device_probe.device, codec.device)
        self.assertNotIn("_device", codec.state_dict())

    def test_load_state_dict_rejects_assign(self):
        codec = _codec(FakeBiCodecModel())

        with self.assertRaisesRegex(ValueError, "assign=True"):
            codec.load_state_dict(codec.state_dict(), assign=True)

    def test_fixed_acoustic_features_have_independent_32_slot_axis(self):
        codec = _feature_codec()

        self.assertEqual(codec.acoustic_unit_length, 32)
        self.assertEqual(codec.acoustic_feature_dim, 3)
        self.assertEqual(tuple(codec.semantic_codebook.shape), (4, 2))

        codes = torch.zeros((2, 32, 1), dtype=torch.long)
        features = codec.acoustic_codes_to_features(codes)
        decoded = codec.decode_features(
            torch.zeros((2, 5, 1), dtype=torch.long),
            features,
        )

        self.assertEqual(tuple(features.shape), (2, 32, 3))
        self.assertEqual(tuple(decoded.shape), (2, 1, 5))

    def test_fixed_acoustic_features_reject_wrong_slot_length(self):
        codec = _feature_codec()

        with self.assertRaisesRegex(ValueError, "fixed acoustic unit length"):
            codec.acoustic_codes_to_features(torch.zeros((1, 31, 1), dtype=torch.long))


class FakeBiCodecFactory:
    model_dir: Path | None = None

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path):
        cls.model_dir = model_dir
        return FakeBiCodecModel()


class FakeBiCodecModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        self.quantizer = SimpleNamespace(codebook_size=4096)
        self.speaker_encoder = SimpleNamespace(
            quantizer=SimpleNamespace(codebook_size=8192, num_quantizers=1),
            perceiver_sampler=SimpleNamespace(latents=torch.zeros(3, 128)),
        )
        self.eval_called = False
        self.batch = None
        self.semantic = None
        self.acoustic_tokens = None
        self.detokenized_audio = None

    def eval(self):
        self.eval_called = True
        return super().eval()

    def tokenize(self, batch):
        self.batch = batch
        batch_size = batch["wav"].size(0)
        semantic = torch.ones((batch_size, 5), dtype=torch.long)
        acoustic_tokens = torch.arange(batch_size * 3, dtype=torch.long).reshape(batch_size, 1, 3)
        return semantic, acoustic_tokens

    def detokenize(self, semantic, acoustic_tokens):
        self.semantic = semantic
        self.acoustic_tokens = acoustic_tokens
        if self.detokenized_audio is not None:
            return self.detokenized_audio
        return torch.ones((semantic.size(0), 1, 16))


class FakeProcessor:
    path: str | None = None

    def __init__(self) -> None:
        self.sample_rate = None

    @classmethod
    def from_pretrained(cls, path):
        cls.path = path
        return cls()

    def __call__(
        self,
        wav,
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: bool,
        output_hidden_states: bool,
    ):
        self.sample_rate = sampling_rate
        return SimpleNamespace(input_values=torch.stack([torch.as_tensor(item) for item in wav]))


class FakeFeatureExtractor(nn.Module):
    path: str | None = None

    def __init__(self) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        self.config = SimpleNamespace(output_hidden_states=False)
        self.eval_called = False
        self.hidden_states = [torch.full((2, 4, 3), float(index)) for index in range(17)]

    @classmethod
    def from_pretrained(cls, path):
        cls.path = path
        return cls()

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden_states = self.hidden_states
        if hidden_states is not None:
            hidden_states = [state[:batch_size] for state in hidden_states]
        return SimpleNamespace(hidden_states=hidden_states)


def _codec(model: FakeBiCodecModel) -> BiCodec:
    return BiCodec(
        model=model,
        processor=FakeProcessor(),
        feature_extractor=FakeFeatureExtractor(),
        device=torch.device("cpu"),
        config={
            "sample_rate": 16000,
            "ref_segment_duration": 0.5,
            "latent_hop_length": 320,
        },
    )


class FeatureQuantizer(nn.Module):
    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.codebook_size = size
        self.codebook_dim = dim
        self.codebook = nn.Embedding(size, dim)
        self.project_out = nn.Identity()

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices).transpose(1, 2)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices[..., 0])


class FeatureSpeakerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = FeatureQuantizer(8, 3)
        self.perceiver_sampler = SimpleNamespace(latents=torch.zeros(32, 3))
        self.project = nn.Linear(32 * 3, 2)


class FeatureBiCodecModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = FeatureQuantizer(4, 2)
        self.speaker_encoder = FeatureSpeakerEncoder()
        self.prenet = lambda z, condition: z
        self.decoder = lambda value: value[:, :1]


def _feature_codec() -> BiCodec:
    model = FeatureBiCodecModel()
    return BiCodec(
        model=model,
        processor=FakeProcessor(),
        feature_extractor=FakeFeatureExtractor(),
        device=torch.device("cpu"),
        config={"sample_rate": 16000, "latent_hop_length": 320},
    )


if __name__ == "__main__":
    unittest.main()
