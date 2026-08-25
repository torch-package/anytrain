from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

from anytrain._buffer import register_buffer
from anytrain.audio.codec.longcat.assets import LongCatAssets, LongCatConfigPaths
from anytrain.audio.codec.longcat.codec import LongCat
from anytrain.audio.codec.longcat.tokenizer import represent_tokenizer
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec, AudioCodeStreamSpec
from anytrain.audio.tokenizer import PaddedAudioTokenizer


class LongCatCodecTest(unittest.TestCase):
    def test_from_pretrained_sets_longcat_checkpoint_env_while_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = LongCatConfigPaths(
                encoder=root / "encoder.yaml",
                decoder_16k_4codebooks=root / "decoder16.yaml",
                decoder_24k_2codebooks=root / "decoder24_2.yaml",
                decoder_24k_4codebooks=root / "decoder24_4.yaml",
            )
            assets = LongCatAssets(
                cache_dir=root,
                ckpt_dir=root / "ckpts",
                configs=configs,
                checkpoints={},
            )

            def fake_load_encoder(config_path, device):
                self.assertEqual(config_path, str(configs.encoder))
                self.assertEqual(os.environ["LONGCAT_AUDIO_CODEC_CKPT_DIR"], str(assets.ckpt_dir))
                return nn.Identity()

            def fake_load_decoder(config_path, device):
                self.assertEqual(config_path, str(configs.decoder_16k_4codebooks))
                self.assertEqual(os.environ["LONGCAT_AUDIO_CODEC_CKPT_DIR"], str(assets.ckpt_dir))
                return nn.Identity()

            def fake_ensure_assets(**kwargs):
                self.assertEqual(kwargs["decoders"], ("16k_4codebooks",))
                return assets

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LONGCAT_AUDIO_CODEC_CKPT_DIR", None)
                with (
                    patch(
                        "anytrain.audio.codec.longcat.codec.ensure_longcat_assets",
                        side_effect=fake_ensure_assets,
                    ),
                    patch(
                        "anytrain.audio.codec.longcat.codec._load_longcat_loaders",
                        return_value=(fake_load_encoder, fake_load_decoder),
                    ),
                ):
                    codec = LongCat.from_pretrained(device="cpu")

                self.assertIsInstance(codec.encoder, nn.Identity)
                self.assertIsInstance(codec.decoders["16k_4codebooks"], nn.Identity)
                self.assertEqual(codec.sample_rate, 16000)
                self.assertEqual(codec.name, "longcat")
                self.assertEqual(codec.codebook_sizes, (8192, 8100, 8100, 8100))
                self.assertEqual(codec.semantic_codebook_sizes, (8192,))
                self.assertEqual(codec.acoustic_codebook_sizes, (8100, 8100, 8100))
                self.assertNotIn("LONGCAT_AUDIO_CODEC_CKPT_DIR", os.environ)

    def test_encode_returns_time_major_aligned_codebooks(self):
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        codes = codec.encode(torch.zeros((2, 1, 12)), 16000)

        self.assertEqual(tuple(codes.shape), (2, 2, 4))
        self.assertTrue(torch.equal(codes[..., 0], torch.tensor([[7, 8], [7, 8]])))
        self.assertTrue(torch.equal(codes[..., 1:], torch.ones((2, 2, 3), dtype=torch.long)))

    def test_padded_tokenizer_reports_named_frame_lengths(self):
        codec = _codec(FakeLongCatEncoder())
        tokenizer = represent_tokenizer(codec, _code_spec(), AudioBackendIdentity("longcat"))

        self.assertIsInstance(tokenizer, PaddedAudioTokenizer)
        output = tokenizer.tokenize_padded(
            torch.zeros((2, 1, 12)),
            16_000,
            torch.tensor([12, 5]),
        )

        self.assertEqual(tuple(output.codes), ("semantic", "rvq_0", "rvq_1", "rvq_2"))
        for value in output.lengths.values():
            self.assertTrue(torch.equal(value, torch.tensor([2, 1])))

    def test_padded_tokenizer_uses_acoustic_length_when_semantic_has_tail_frame(self):
        codec = _codec(FakeLongCatEncoder(semantic_tail=True))
        tokenizer = represent_tokenizer(codec, _code_spec(), AudioBackendIdentity("longcat"))

        output = tokenizer.tokenize_padded(
            torch.zeros((2, 1, 12)),
            16_000,
            torch.tensor([12, 5]),
        )

        self.assertEqual(tuple(output.codes["semantic"].shape), (2, 3))
        self.assertTrue(torch.equal(output.lengths["semantic"], torch.tensor([2, 1])))

    def test_padded_encode_requires_length_aware_encoder_and_native_sample_rate(self):
        codec = _codec(FakeLongCatEncoder())
        with self.assertRaisesRegex(ValueError, "pre-resampled to 16000 Hz"):
            codec.encode_padded(
                torch.zeros((2, 1, 12)),
                48_000,
                torch.tensor([12, 7]),
            )

        encoder = nn.Identity()
        cast(Any, encoder).input_sample_rate = 16_000
        with self.assertRaisesRegex(TypeError, "length-aware encode methods"):
            _codec(encoder).encode_padded(
                torch.zeros((2, 1, 12)),
                16_000,
                torch.tensor([12, 7]),
            )

    def test_encode_debug_boundary_writes_shape_and_elapsed_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "longcat.jsonl"
            codec = LongCat(
                encoder=FakeLongCatEncoder(),
                decoders={"16k_4codebooks": FakeLongCatDecoder()},
                device=torch.device("cpu"),
                assets=_assets(),
            )
            with patch.dict(
                os.environ,
                {"ANYTRAIN_LONGCAT_DEBUG_LOG": str(trace_path)},
            ):
                codec.encode(torch.zeros((2, 1, 12)), 16_000)

            record = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(record["event"], "longcat_encode_boundary")
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["audio_shape"], [2, 1, 12])
            self.assertEqual(record["audio_seconds"], 0.0015)
            self.assertEqual(record["encoder_output_shapes"], [[2, 2], [2, 3, 2]])
            self.assertIn("elapsed_seconds", record)

    def test_encode_debug_boundary_records_invalid_backend_output_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "longcat.jsonl"
            encoder = FakeLongCatEncoder()
            codec = _codec(encoder)
            with (
                patch.dict(os.environ, {"ANYTRAIN_LONGCAT_DEBUG_LOG": str(trace_path)}),
                patch.object(
                    encoder,
                    "forward",
                    return_value=(
                        torch.zeros((2, 2)),
                        torch.ones((2, 3, 2), dtype=torch.long),
                    ),
                ),
                self.assertRaisesRegex(TypeError, "semantic codes"),
            ):
                codec.encode(torch.zeros((2, 1, 12)), 16_000)

            record = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["error_type"], "TypeError")

    def test_encode_validates_inputs_before_calling_the_backend(self):
        encoder = FakeLongCatEncoder()
        codec = _codec(encoder)
        with (
            patch.object(encoder, "forward") as forward,
            self.assertRaisesRegex(TypeError, "audio to be a Tensor"),
        ):
            codec.encode(cast(Any, [0.0]), 16_000)
        forward.assert_not_called()

        with (
            patch.object(encoder, "forward") as forward,
            self.assertRaisesRegex(ValueError, "audio must not be empty"),
        ):
            codec.encode(torch.zeros((1, 1, 0)), 16_000)
        forward.assert_not_called()

        with (
            patch.object(encoder, "forward") as forward,
            self.assertRaisesRegex(TypeError, "sample_rate"),
        ):
            codec.encode(torch.zeros((1, 1, 12)), cast(Any, True))
        forward.assert_not_called()

    def test_semantic_feature_dim_matches_semantic_codebook(self):
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        self.assertEqual(codec.semantic_feature_dim, 3)

    def test_decode_splits_time_major_codebooks_for_backend(self):
        decoder = FakeLongCatDecoder()
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        codes = torch.arange(16).reshape(2, 2, 4)

        audio = codec.decode(codes)

        self.assertEqual(tuple(audio.shape), (2, 1, 6))
        assert decoder.semantic_codes is not None
        assert decoder.acoustic is not None
        self.assertTrue(torch.equal(decoder.semantic_codes, codes[..., 0]))
        self.assertTrue(torch.equal(decoder.acoustic, codes[..., 1:].transpose(1, 2)))

    def test_acoustic_codebook_to_features_exposes_decoder_latents_as_time_major(self):
        decoder = FakeLongCatDecoder()
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        with self.assertRaisesRegex(TypeError, "acoustic quantizer"):
            codec.acoustic_codebook_codes_to_features(0, torch.tensor([[1, 8099]]))

    def test_acoustic_codes_to_features_requires_time_major_full_codebooks(self):
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "outside the codebook"):
            codec.acoustic_codebook_codes_to_features(0, torch.full((1, 3), 8100))

    def test_decode_features_feeds_longcat_decoder_with_channel_major_latents(self):
        decoder = FakeLongCatDecoder()
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        semantic_features = torch.zeros((1, 2, 3))
        acoustic_features = torch.arange(6, dtype=torch.float).reshape(1, 2, 3)

        audio = codec.decode_acoustic_features(semantic_features, acoustic_features)

        self.assertEqual(tuple(audio.shape), (1, 1, 6))
        self.assertIsNone(decoder.semantic_codes)
        assert decoder.acoustic is not None
        self.assertTrue(torch.equal(decoder.acoustic[:, 3:], acoustic_features.transpose(1, 2)))

    def test_decode_features_requires_semantic_and_acoustic_time_alignment(self):
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "align on batch and time"):
            codec.decode_acoustic_features(
                torch.zeros((1, 2, 3)),
                torch.zeros((1, 3, 4)),
            )

    def test_to_moves_encoder_decoder_and_public_device_together(self):
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        codec.to("meta")

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.encoder.device_probe.device, codec.device)
        self.assertEqual(codec.decoders["16k_4codebooks"].device_probe.device, codec.device)
        self.assertNotIn("_device", codec.state_dict())

    def test_constructor_moves_backends_to_requested_device(self):
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("meta"),
            assets=_assets(),
        )

        self.assertEqual(codec.device, torch.device("meta"))
        self.assertEqual(codec.encoder.device_probe.device, codec.device)
        self.assertEqual(codec.decoders["16k_4codebooks"].device_probe.device, codec.device)

    def test_semantic_frame_rate_accepts_numpy_integer_attrs(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed.")

        encoder = FakeLongCatEncoder()
        encoder.input_sample_rate = np.int64(16000)
        encoder.hop_length = np.int64(960)
        codec = LongCat(
            encoder=encoder,
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        self.assertAlmostEqual(codec.semantic_frame_rate, 16000 / 960)

    def test_factor_capability_hides_upstream_quantizer_layout(self):
        decoder = FakeFactorLongCatDecoder()
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        self.assertEqual(codec.acoustic_factor_sizes, ((90, 90),) * 3)
        self.assertEqual(len(codec.acoustic_factor_codebooks), 6)
        projected = codec.project_acoustic_factor_features(
            torch.ones((1, 2, 4)),
            active_codebooks=1,
        )
        self.assertEqual(tuple(projected.shape), (1, 2, 4))

        decoder.acoustic_quantizer.stage(1).retarget_code = 2 * 90 + 3
        residual_codes = codec.quantize_acoustic_residual(
            torch.zeros((1, 2, 4)),
            codebook=1,
        )
        self.assertTrue(torch.equal(residual_codes, torch.tensor([[[2, 3], [2, 3]]])))

        native = codec.native_acoustic_features(
            torch.zeros((1, 2, 1), dtype=torch.long),
            active_codebooks=1,
        )
        self.assertEqual(tuple(native.shape), (1, 2, 4))

    def test_factor_projection_requires_positive_active_codebooks(self):
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": FakeFactorLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "active_codebooks"):
            codec.project_acoustic_factor_features(
                torch.zeros((1, 2, 4)),
                active_codebooks=0,
            )


class FakeLongCatDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        self.n_codebooks = 3
        self.acoustic_codebook_size = 90
        self.latent_dim = 3
        self.codes: torch.Tensor | None = None
        self.features: torch.Tensor | None = None
        self.semantic_codes: torch.Tensor | None = None
        self.acoustic: torch.Tensor | None = None

    def acoustic_codes_to_latents(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        self.codes = acoustic_codes
        self.features = torch.arange(6, dtype=torch.float).reshape(1, 3, 2)
        return self.features

    def forward(self, semantic_codes: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
        self.semantic_codes = semantic_codes
        self.acoustic = acoustic
        return torch.ones((semantic_codes.size(0), 1, acoustic.size(-1) * 3))

    def decoder(self, features: torch.Tensor) -> torch.Tensor:
        self.acoustic = features
        return torch.ones((features.size(0), 1, features.size(-1) * 3))


class FakeFactorStage(nn.Module):
    codebook_size_a: int
    codebook_size_b: int
    codebook_a: nn.Embedding
    codebook_b: nn.Embedding
    out_proj_a: nn.Conv1d
    out_proj_b: nn.Conv1d
    retarget_code: int

    def __init__(self) -> None:
        super().__init__()
        self.codebook_size_a = 90
        self.codebook_size_b = 90
        self.codebook_a = nn.Embedding(90, 2)
        self.codebook_b = nn.Embedding(90, 2)
        self.out_proj_a = nn.Conv1d(2, 2, kernel_size=1)
        self.out_proj_b = nn.Conv1d(2, 2, kernel_size=1)
        self.retarget_code = 0

    def forward(self, residual: torch.Tensor):
        codes = torch.full(
            residual.shape[:1] + residual.shape[2:],
            self.retarget_code,
            dtype=torch.long,
            device=residual.device,
        )
        empty = residual.new_zeros(residual.size(0))
        return residual, empty, empty, codes


class FakeFactorQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList([FakeFactorStage() for _ in range(3)])

    def stage(self, index: int) -> FakeFactorStage:
        return cast(FakeFactorStage, self.quantizers[index])

    def from_codes(self, codes: torch.Tensor):
        projected: torch.Tensor | None = None
        for index in range(codes.size(1)):
            stage = self.stage(index)
            composite = codes[:, index]
            code_a = torch.div(composite, stage.codebook_size_b, rounding_mode="floor")
            code_b = composite.remainder(stage.codebook_size_b)
            value = torch.cat(
                (
                    stage.out_proj_a(stage.codebook_a(code_a).transpose(1, 2)),
                    stage.out_proj_b(stage.codebook_b(code_b).transpose(1, 2)),
                ),
                dim=1,
            )
            projected = value if projected is None else projected + value
        if projected is None:
            raise AssertionError("factor test quantizer requires at least one codebook.")
        return projected, projected, codes


class FakeFactorLongCatDecoder(FakeLongCatDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 4
        self.acoustic_quantizer = FakeFactorQuantizer()


class FakeLongCatEncoder(nn.Module):
    def __init__(self, *, semantic_tail: bool = False) -> None:
        super().__init__()
        register_buffer(self, "device_probe", torch.empty(0), persistent=False)
        self.input_sample_rate: Any = 16000
        self.hop_length: Any = 960
        self.semantic_tail = semantic_tail

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        *,
        n_acoustic_codebooks: int,
    ):
        batch = audio.size(0)
        semantic = torch.tensor([[7, 8]]).expand(batch, -1)
        acoustic = torch.ones((batch, n_acoustic_codebooks, 2), dtype=torch.long)
        return semantic, acoustic

    def preprocess(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.input_sample_rate:
            raise ValueError("unexpected sample rate")
        return audio

    def get_semantic_codes_with_lengths(
        self,
        audio: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = audio.size(0)
        codes = torch.tensor([[7, 8, 9]], device=audio.device).expand(batch, -1)
        code_lengths = torch.div(lengths + 5, 6, rounding_mode="floor")
        if self.semantic_tail:
            code_lengths = code_lengths + 1
        return codes[:, : 3 if self.semantic_tail else 2], code_lengths

    def get_acoustic_codes_with_lengths(
        self,
        audio: torch.Tensor,
        lengths: torch.Tensor,
        *,
        n_acoustic_codebooks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = audio.size(0)
        code_time = 3 if self.semantic_tail else 2
        codes = torch.ones(
            (batch, n_acoustic_codebooks, code_time),
            dtype=torch.long,
            device=audio.device,
        )
        return codes, torch.div(lengths + 5, 6, rounding_mode="floor")

    def get_semantic_codebook(self) -> torch.Tensor:
        return torch.zeros((3, 8192))


def _assets() -> LongCatAssets:
    return LongCatAssets(
        cache_dir=Path(),
        ckpt_dir=Path(),
        configs=LongCatConfigPaths(
            encoder=Path(),
            decoder_16k_4codebooks=Path(),
            decoder_24k_2codebooks=Path(),
            decoder_24k_4codebooks=Path(),
        ),
        checkpoints={},
    )


def _code_spec() -> AudioCodeSpec:
    return AudioCodeSpec(
        sample_rate=16_000,
        streams=(
            AudioCodeStreamSpec("semantic", 8192, frame_rate=16_000 / 960),
            AudioCodeStreamSpec("rvq_0", 8100, frame_rate=16_000 / 960),
            AudioCodeStreamSpec("rvq_1", 8100, frame_rate=16_000 / 960),
            AudioCodeStreamSpec("rvq_2", 8100, frame_rate=16_000 / 960),
        ),
    )


def _codec(encoder: nn.Module) -> LongCat:
    return LongCat(
        encoder=encoder,
        decoders={"16k_4codebooks": FakeLongCatDecoder()},
        device=torch.device("cpu"),
        assets=_assets(),
    )


if __name__ == "__main__":
    unittest.main()
