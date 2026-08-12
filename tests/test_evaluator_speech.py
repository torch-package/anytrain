from __future__ import annotations

import os
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch

import torch

import anytrain.evaluator.speech as speech
from anytrain.env import (
    ANYTRAIN_HOME_ENV,
    TORCH_HOME_ENV,
    WHISPER_ROOT_ENV,
)
from anytrain.evaluator.speech import (
    SpeechEvaluator,
    SpeechEvaluatorConfig,
    SpeechEvaluatorLoadOptions,
    UTMOSEvaluator,
    UTMOSLoadOptions,
    UTMOSSemanticConfig,
    WhisperASREvaluator,
    WhisperLoadOptions,
    WhisperSemanticConfig,
)
from anytrain.evaluator.speech._config import (
    DEFAULT_UTMOS_REPOSITORY,
    DEFAULT_UTMOS_REVISION,
    DEFAULT_UTMOS_WEIGHTS_SHA256,
    OPENAI_WHISPER_VERSION,
    WHISPER_MODEL_SHA256,
)
from anytrain.evaluator.speech.audio import load_wave_batch
from anytrain.evaluator.speech.utmos import TorchHubUTMOSBackend


class FakeUTMOSBackend:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def score(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        return self.output


class FakeCallableUTMOSBackend:
    def __init__(self, output: Callable[[torch.Tensor, int], torch.Tensor]):
        self.output = output
        self.calls: list[tuple[torch.Tensor, int]] = []

    def score(self, audio: Any, sample_rate: int) -> torch.Tensor:
        if not isinstance(audio, torch.Tensor):
            raise TypeError("fake UTMOS audio must be a Tensor.")
        self.calls.append((audio, sample_rate))
        return self.output(audio, sample_rate)


class FakeWhisperModule(ModuleType):
    def __init__(self, model):
        super().__init__("whisper")
        self.__version__ = OPENAI_WHISPER_VERSION
        self._MODELS = {
            name: f"https://models.example/{digest}/{name}.pt"
            for name, digest in WHISPER_MODEL_SHA256.items()
        }
        self.model = model
        self.calls = []
        self.pad_calls = []
        self.mel_calls = []
        self.DecodingOptions = FakeDecodingOptions

    def load_model(self, model_name, **load_options):
        self.calls.append((model_name, load_options))
        return self.model

    def pad_or_trim(self, audio, length=480000, *, axis=-1):
        self.pad_calls.append((audio.shape, length, axis))
        if axis != -1:
            raise ValueError("FakeWhisperModule only pads the last axis.")
        if audio.shape[axis] > length:
            return audio.narrow(axis, 0, length)
        if audio.shape[axis] < length:
            return torch.nn.functional.pad(audio, (0, length - audio.shape[axis]))
        return audio

    def log_mel_spectrogram(self, audio, n_mels=80, padding=0, device=None):
        self.mel_calls.append((audio.shape, n_mels, padding, device))
        return torch.zeros(audio.shape[0], n_mels, 3000, device=device)


@dataclass(frozen=True)
class FakeDecodingOptions:
    task: str = "transcribe"
    language: str | None = None
    temperature: float = 0.0
    sample_len: int | None = None
    best_of: int | None = None
    beam_size: int | None = None
    patience: float | None = None
    length_penalty: float | None = None
    prompt: str | list[int] | None = None
    prefix: str | list[int] | None = None
    suppress_tokens: str | list[int] | None = "-1"
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float | None = 1.0
    fp16: bool = True


@dataclass(frozen=True)
class FakeDecodingResult:
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    temperature: float = 0.0


@dataclass(frozen=True)
class FakeWhisperDims:
    n_mels: int = 80


class FakeWhisperModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = []
        self.decode_calls = []
        self.dims = FakeWhisperDims()
        self.weight = torch.nn.Parameter(torch.ones(()))

    @property
    def device(self):
        return self.weight.device

    def transcribe(self, audio, **decode_options):
        self.calls.append(
            (
                audio,
                decode_options,
                self.training,
                self.weight.requires_grad,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        )
        return self.output

    def decode(self, mel, options):
        self.decode_calls.append(
            (
                mel.shape,
                options,
                self.training,
                self.weight.requires_grad,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        )
        if callable(self.output):
            return self.output(mel, options)
        return self.output


class FakeWhisperASREvaluator(WhisperASREvaluator):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        self.calls = []

    def evaluate(
        self,
        audio,
        sample_rate,
        reference_text=None,
        *,
        target_text=None,
        **decode_options,
    ):
        self.calls.append((audio, sample_rate, reference_text, target_text, decode_options))
        return dict(self.metrics)


class FakeUTMOSModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = []
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def forward(self, audio, sample_rate):
        self.calls.append(
            (
                audio,
                sample_rate,
                self.training,
                self.weight.requires_grad,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        )
        return self.output


class SpeechEvaluatorTest(unittest.TestCase):
    def test_speech_top_level_exports_only_evaluators(self):
        self.assertEqual(
            set(speech.__all__),
            {
                "SpeechEvaluator",
                "SpeechEvaluatorConfig",
                "SpeechEvaluatorLoadOptions",
                "UTMOSEvaluator",
                "UTMOSLoadOptions",
                "UTMOSSemanticConfig",
                "WhisperASREvaluator",
                "WhisperLoadOptions",
                "WhisperSemanticConfig",
            },
        )
        self.assertFalse(hasattr(speech, "OpenAIWhisperBackend"))
        self.assertFalse(hasattr(speech, "TorchHubUTMOSBackend"))

    def test_semantic_configs_are_stable_and_exclude_load_options(self):
        whisper = WhisperSemanticConfig(decode_options=(("language", "en"),))
        utmos = UTMOSSemanticConfig()
        config = SpeechEvaluatorConfig(whisper=whisper, utmos=utmos)

        first = config.evaluator(
            SpeechEvaluatorLoadOptions(
                whisper=WhisperLoadOptions(
                    device="cpu",
                    download_root="first-cache",
                    local_files_only=True,
                ),
                utmos=UTMOSLoadOptions(
                    device="cpu",
                    allow_remote_code=True,
                    local_files_only=True,
                ),
            )
        )
        second = config.evaluator(
            SpeechEvaluatorLoadOptions(
                whisper=WhisperLoadOptions(device="cuda:0", download_root="second-cache"),
                utmos=UTMOSLoadOptions(device="cuda:0", allow_remote_code=True),
            )
        )

        self.assertEqual(first.semantic_id(), config.semantic_id())
        self.assertEqual(first.semantic_id(), second.semantic_id())
        self.assertIsNone(first.asr._backend.model)
        self.assertIsNone(cast(Any, first.utmos.backend).model)
        self.assertNotEqual(
            config.semantic_id(),
            SpeechEvaluatorConfig(
                whisper=WhisperSemanticConfig(
                    decode_options=(("language", "zh"),)
                ),
                utmos=utmos,
            ).semantic_id(),
        )

    def test_default_semantics_pin_whisper_artifact_and_utmos_commit(self):
        whisper = WhisperSemanticConfig()
        utmos = UTMOSSemanticConfig()

        self.assertEqual(
            whisper.artifact_sha256,
            "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
        )
        self.assertEqual(utmos.revision, DEFAULT_UTMOS_REVISION)
        self.assertEqual(len(utmos.revision), 40)
        self.assertEqual(
            utmos.weights_sha256,
            DEFAULT_UTMOS_WEIGHTS_SHA256,
        )
        self.assertEqual(
            whisper.semantic_payload()["artifact_sha256"],
            whisper.artifact_sha256,
        )

    def test_custom_utmos_backend_has_no_implicit_semantic_identity(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend(4.0))

        with self.assertRaisesRegex(RuntimeError, "pinned built-in backend"):
            evaluator.semantic_id()

    def test_speech_evaluator_returns_asr_text_and_utmos_metrics(self):
        audio = torch.zeros(2, 16000)
        asr = FakeWhisperASREvaluator({"bleu": 99.0, "wer": 0.25, "chrf": 88.0})
        utmos = UTMOSEvaluator(backend=FakeUTMOSBackend([4.0, 5.0]))
        evaluator = SpeechEvaluator(asr=asr, utmos=utmos)

        metrics = evaluator(
            audio,
            16000,
            reference_text=["hello world", "good morning"],
            temperature=0.0,
        )

        self.assertEqual(metrics, {"bleu": 99.0, "wer": 0.25, "chrf": 88.0, "utmos": 4.5})
        self.assertEqual(
            asr.calls,
            [
                (
                    audio,
                    16000,
                    ["hello world", "good morning"],
                    None,
                    {"temperature": 0.0},
                )
            ],
        )
        self.assertEqual(cast(Any, utmos.backend).calls, [(audio, 16000)])

    def test_speech_evaluator_validates_children(self):
        with self.assertRaisesRegex(TypeError, "asr"):
            SpeechEvaluator(asr=cast(Any, UTMOSEvaluator(backend=FakeUTMOSBackend(4.0))))
        with self.assertRaisesRegex(TypeError, "utmos"):
            SpeechEvaluator(
                asr=FakeWhisperASREvaluator({"bleu": 99.0, "wer": 0.0, "chrf": 99.0}),
                utmos=cast(
                    Any,
                    FakeWhisperASREvaluator({"bleu": 99.0, "wer": 0.0, "chrf": 99.0}),
                ),
            )

    def test_whisper_asr_loads_openai_whisper_and_returns_text_metrics(self):
        audio = torch.zeros(2, 16000)
        model = FakeWhisperModel(
            [
                FakeDecodingResult("hello world"),
                FakeDecodingResult("hello world"),
            ]
        )
        whisper = FakeWhisperModule(model)
        with patch.dict(os.environ, {ANYTRAIN_HOME_ENV: "/tmp/anytrain"}, clear=True):
            evaluator = WhisperASREvaluator(
                model_name="tiny",
                device="cpu",
                decode_options={"language": "en"},
                load_options={"in_memory": True},
            )
        with (
            patch.dict(os.environ, {ANYTRAIN_HOME_ENV: "/tmp/anytrain"}, clear=True),
            patch.dict("sys.modules", {"whisper": whisper}),
        ):
            metrics = evaluator(
                audio,
                16000,
                reference_text=["hello world", "hello world"],
                temperature=0.0,
            )

        self.assertEqual(set(metrics), {"bleu", "wer", "chrf"})
        self.assertEqual(metrics["bleu"], 100.0)
        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)
        self.assertEqual(
            whisper.calls,
            [
                (
                    "tiny",
                    {
                        "in_memory": True,
                        "device": "cpu",
                        "download_root": "/tmp/anytrain/whisper",
                    },
                )
            ],
        )
        self.assertEqual(model.calls, [])
        self.assertEqual(len(model.decode_calls), 1)
        self.assertEqual(model.decode_calls[0][0], torch.Size([2, 80, 3000]))
        self.assertEqual(
            model.decode_calls[0][1],
            FakeDecodingOptions(language="en", temperature=0.0, fp16=False),
        )
        self.assertFalse(model.decode_calls[0][2])
        self.assertFalse(model.decode_calls[0][3])
        self.assertFalse(model.decode_calls[0][4])
        self.assertTrue(model.decode_calls[0][5])
        self.assertEqual(whisper.pad_calls[0], (torch.Size([2, 16000]), 480000, -1))
        self.assertEqual(
            whisper.mel_calls,
            [(torch.Size([2, 480000]), 80, 0, torch.device("cpu"))],
        )

    def test_whisper_asr_requires_reference_text(self):
        evaluator = WhisperASREvaluator()

        with self.assertRaisesRegex(ValueError, "reference_text is required"):
            evaluator(torch.zeros(16000), 16000)

        self.assertIsNone(evaluator._backend.model)

    def test_whisper_asr_rejects_prediction_reference_count_mismatch(self):
        model = FakeWhisperModel({"text": "hello world"})
        whisper = FakeWhisperModule(model)
        evaluator = WhisperASREvaluator()

        with (
            patch.dict("sys.modules", {"whisper": whisper}),
            self.assertRaisesRegex(ValueError, "counts must match"),
        ):
            evaluator(torch.zeros(16000), 16000, reference_text=["hello", "world"])

    def test_whisper_asr_builds_default_loader_without_loading_model(self):
        with patch.dict(os.environ, {ANYTRAIN_HOME_ENV: "/tmp/anytrain"}, clear=True):
            evaluator = WhisperASREvaluator(model_name="tiny")
            self.assertEqual(os.environ[WHISPER_ROOT_ENV], "/tmp/anytrain/whisper")

        self.assertEqual(evaluator.model_name, "tiny")
        self.assertIsNone(evaluator._backend.model)

    def test_whisper_asr_default_model_is_large_v3(self):
        evaluator = WhisperASREvaluator()

        self.assertEqual(evaluator.model_name, "large-v3")
        self.assertIsNone(evaluator._backend.model)
        self.assertIn("large-v3", evaluator.supported_model_names)

    def test_whisper_asr_rejects_backend_or_artifact_drift(self):
        model = FakeWhisperModel({"text": "hello world"})
        wrong_version = FakeWhisperModule(model)
        wrong_version.__version__ = "future"
        evaluator = WhisperASREvaluator(model_name="tiny", device="cpu")

        with (
            patch.dict("sys.modules", {"whisper": wrong_version}),
            self.assertRaisesRegex(RuntimeError, "requires openai-whisper"),
        ):
            evaluator.transcribe(torch.zeros(16000), 16000)

        wrong_artifact = FakeWhisperModule(model)
        wrong_artifact._MODELS["tiny"] = "https://models.example/wrong/tiny.pt"
        evaluator = WhisperASREvaluator(model_name="tiny", device="cpu")
        with (
            patch.dict("sys.modules", {"whisper": wrong_artifact}),
            self.assertRaisesRegex(RuntimeError, "pinned artifact"),
        ):
            evaluator.transcribe(torch.zeros(16000), 16000)

    def test_whisper_local_only_fails_before_model_loader(self):
        model = FakeWhisperModel({"text": "hello world"})
        whisper = FakeWhisperModule(model)
        with TemporaryDirectory() as root:
            evaluator = WhisperASREvaluator(
                model_name="tiny",
                device="cpu",
                download_root=root,
                local_files_only=True,
            )
            with (
                patch.dict("sys.modules", {"whisper": whisper}),
                self.assertRaisesRegex(FileNotFoundError, "Local-only Whisper"),
            ):
                evaluator.transcribe(torch.zeros(16000), 16000)

        self.assertEqual(whisper.calls, [])

    def test_whisper_asr_rejects_unknown_model_name(self):
        with self.assertRaisesRegex(ValueError, "model_name must be one of"):
            WhisperASREvaluator(model_name="/tmp/custom-whisper.pt")
        with self.assertRaisesRegex(ValueError, "model_name must be one of"):
            WhisperASREvaluator(model_name="future-whisper")

    def test_whisper_asr_rejects_backend_or_model_injection(self):
        with self.assertRaisesRegex(TypeError, "backend"):
            cast(Any, WhisperASREvaluator)(backend=object())
        with self.assertRaisesRegex(TypeError, "model"):
            cast(Any, WhisperASREvaluator)(model=object())

    def test_whisper_asr_transcribes_in_inference_mode(self):
        model = FakeWhisperModel({"text": "hello world"})
        whisper = FakeWhisperModule(model)
        evaluator = WhisperASREvaluator(model_name="tiny", device="cpu")

        with patch.dict("sys.modules", {"whisper": whisper}):
            prediction = evaluator.transcribe(torch.zeros(16000), 16000, language="en")

        self.assertEqual(prediction, "hello world")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].shape, (16000,))
        self.assertEqual(model.calls[0][1], {"language": "en"})
        self.assertFalse(model.calls[0][2])
        self.assertFalse(model.calls[0][3])
        self.assertFalse(model.calls[0][4])
        self.assertTrue(model.calls[0][5])

    def test_whisper_asr_falls_back_to_transcribe_for_long_batch(self):
        audio = torch.zeros(2, 480001)
        model = FakeWhisperModel({"text": "hello world"})
        whisper = FakeWhisperModule(model)
        evaluator = WhisperASREvaluator(model_name="tiny", device="cpu")

        with patch.dict("sys.modules", {"whisper": whisper}):
            prediction = evaluator.transcribe(audio, 16000, language="en")

        self.assertEqual(prediction, ["hello world", "hello world"])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.decode_calls, [])

    def test_whisper_asr_falls_back_to_transcribe_for_word_timestamps(self):
        audio = torch.zeros(2, 16000)
        model = FakeWhisperModel({"text": "hello world"})
        whisper = FakeWhisperModule(model)
        evaluator = WhisperASREvaluator(model_name="tiny", device="cpu")

        with patch.dict("sys.modules", {"whisper": whisper}):
            prediction = evaluator.transcribe(audio, 16000, language="en", word_timestamps=True)

        self.assertEqual(prediction, ["hello world", "hello world"])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.decode_calls, [])

    def test_whisper_asr_short_batch_decode_uses_temperature_fallback(self):
        audio = torch.zeros(2, 16000)

        def decode(mel, options):
            if options.temperature == 0.0:
                return [
                    FakeDecodingResult("first", compression_ratio=1.0, temperature=0.0),
                    FakeDecodingResult("bad", compression_ratio=9.0, temperature=0.0),
                ]
            return [FakeDecodingResult("second", compression_ratio=1.0, temperature=0.2)]

        model = FakeWhisperModel(decode)
        whisper = FakeWhisperModule(model)
        evaluator = WhisperASREvaluator(
            model_name="tiny",
            device="cpu",
            decode_options={"language": "en"},
        )

        with patch.dict("sys.modules", {"whisper": whisper}):
            prediction = evaluator.transcribe(audio, 16000)

        self.assertEqual(prediction, ["first", "second"])
        self.assertEqual(
            [call[0] for call in model.decode_calls],
            [torch.Size([2, 80, 3000]), torch.Size([1, 80, 3000])],
        )
        self.assertEqual([call[1].temperature for call in model.decode_calls], [0.0, 0.2])
        self.assertEqual(model.calls, [])

    def test_speech_evaluator_call_batch_returns_per_clip_metrics(self):
        texts = ("hello world", "good morning")

        def decode(mel, _options):
            return [FakeDecodingResult(text) for text in texts[: mel.shape[0]]]

        model = FakeWhisperModel(decode)
        whisper = FakeWhisperModule(model)
        asr = WhisperASREvaluator(model_name="tiny", device="cpu")
        utmos_backend = FakeCallableUTMOSBackend(
            lambda audio, _sample_rate: audio.mean(dim=-1).mul(10.0)
        )
        evaluator = SpeechEvaluator(
            asr=asr,
            utmos=UTMOSEvaluator(backend=utmos_backend),
        )
        audios = (
            torch.full((1, 8000), 0.4),
            torch.full((1, 16000), 0.5),
        )

        with patch.dict("sys.modules", {"whisper": whisper}):
            metrics = evaluator.call_batch(
                audios,
                (16000, 16000),
                texts,
                temperature=0.0,
            )

        self.assertEqual(
            [
                {name: value for name, value in item.items() if name != "utmos"}
                for item in metrics
            ],
            [
                {"bleu": 100.0, "wer": 0.0, "chrf": 100.0},
                {"bleu": 100.0, "wer": 0.0, "chrf": 100.0},
            ],
        )
        torch.testing.assert_close(
            torch.tensor([item["utmos"] for item in metrics]),
            torch.tensor([4.0, 5.0]),
        )
        self.assertEqual(len(model.decode_calls), 1)
        self.assertEqual(model.decode_calls[0][0], torch.Size([2, 80, 3000]))
        self.assertEqual(model.calls, [])
        self.assertEqual(
            [call[0].shape for call in utmos_backend.calls],
            [torch.Size([1, 8000]), torch.Size([1, 16000])],
        )

    def test_utmos_call_batch_groups_equal_lengths_and_restores_order(self):
        backend = FakeCallableUTMOSBackend(
            lambda audio, _sample_rate: audio.mean(dim=-1).mul(10.0)
        )
        evaluator = UTMOSEvaluator(backend=backend)

        metrics = evaluator.call_batch(
            (
                torch.full((1, 4), 0.3),
                torch.full((1, 6), 0.4),
                torch.full((1, 4), 0.5),
            ),
            (16000, 16000, 16000),
        )

        torch.testing.assert_close(
            torch.tensor([item["utmos"] for item in metrics]),
            torch.tensor([3.0, 4.0, 5.0]),
        )
        self.assertEqual(
            [call[0].shape for call in backend.calls],
            [torch.Size([2, 4]), torch.Size([1, 6])],
        )

    def test_utmos_returns_batch_mean(self):
        audio = torch.zeros(3, 16000)
        backend = FakeUTMOSBackend([3.0, 4.0, 5.0])
        evaluator = UTMOSEvaluator(backend=backend)

        metrics = evaluator(audio, 16000)

        self.assertEqual(metrics, {"utmos": 4.0})
        self.assertEqual(backend.calls, [(audio, 16000)])

    def test_utmos_accepts_integer_scores_from_backend(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend([3, 5]))

        metrics = evaluator(torch.zeros(2, 16000), 16000)

        self.assertEqual(metrics, {"utmos": 4.0})

    def test_utmos_accepts_tensor_scores_from_backend(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend(torch.tensor([3.0, 5.0])))

        metrics = evaluator(torch.zeros(2, 16000), 16000)

        self.assertEqual(metrics, {"utmos": 4.0})

    def test_utmos_rejects_nonfinite_scalar_sequence_and_tensor_scores(self):
        for nonfinite in (float("nan"), float("inf")):
            cases = {
                "scalar": nonfinite,
                "sequence": [3.0, nonfinite],
                "tensor": torch.tensor([3.0, nonfinite]),
            }
            for name, score in cases.items():
                with (
                    self.subTest(name=name, nonfinite=nonfinite),
                    self.assertRaisesRegex(
                        ValueError,
                        "finite",
                    ),
                ):
                    UTMOSEvaluator(backend=FakeUTMOSBackend(score))(
                        torch.zeros(16000),
                        16000,
                    )

    def test_utmos_rejects_boolean_scores(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend([True]))

        with self.assertRaisesRegex(TypeError, r"score\[0\]"):
            evaluator(torch.zeros(16000), 16000)

    def test_utmos_rejects_boolean_tensor_scores(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend(torch.tensor([True])))

        with self.assertRaisesRegex(TypeError, "score tensor"):
            evaluator(torch.zeros(16000), 16000)

    def test_audio_path_loader_falls_back_to_soundfile(self):
        with (
            patch(
                "anytrain.evaluator.speech.audio._load_audio_file_with_torchaudio",
                side_effect=RuntimeError("broken torchaudio"),
            ),
            patch(
                "anytrain.evaluator.speech.audio._load_audio_file_with_soundfile",
                return_value=(torch.zeros(1, 8), 48000),
            ),
        ):
            wave, sample_rate = load_wave_batch(Path("sample.flac"), 16000)

        self.assertEqual(wave.shape, torch.Size([1, 8]))
        self.assertEqual(sample_rate, 48000)

    def test_utmos_builds_default_backend_without_loading_model(self):
        evaluator = UTMOSEvaluator(device="cpu")

        self.assertIsInstance(evaluator.backend, TorchHubUTMOSBackend)
        backend: Any = evaluator.backend
        self.assertEqual(backend.model_name, "utmos22_strong")
        self.assertIsNone(backend.model)
        self.assertFalse(backend.allow_remote_code)

    def test_torch_hub_utmos_backend_requires_remote_code_opt_in(self):
        backend = TorchHubUTMOSBackend(device="cpu")

        with (
            patch("torch.hub.load") as load,
            self.assertRaisesRegex(ValueError, "allow_remote_code"),
        ):
            backend._load_model()

        load.assert_not_called()

    def test_torch_hub_utmos_backend_scores_tensor_audio(self):
        model = FakeUTMOSModel(torch.tensor([3.0, 5.0]))
        backend = TorchHubUTMOSBackend(device="cpu", allow_remote_code=True)

        with (
            patch.dict(os.environ, {ANYTRAIN_HOME_ENV: "/tmp/anytrain"}, clear=True),
            patch("torch.hub.load", return_value=model) as load,
            patch.object(backend, "_validate_cached_weights") as validate_weights,
        ):
            score = backend.score(torch.zeros(2, 16000), 16000)
            self.assertEqual(os.environ[TORCH_HOME_ENV], "/tmp/anytrain/torch")

        load.assert_called_once_with(
            f"{DEFAULT_UTMOS_REPOSITORY}:{DEFAULT_UTMOS_REVISION}",
            "utmos22_strong",
            trust_repo=True,
        )
        self.assertTrue(torch.equal(score, torch.tensor([3.0, 5.0])))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].shape, torch.Size([2, 16000]))
        self.assertEqual(model.calls[0][1], 16000)
        self.assertFalse(model.calls[0][2])
        self.assertFalse(model.calls[0][3])
        self.assertFalse(model.calls[0][4])
        self.assertTrue(model.calls[0][5])
        self.assertEqual(model.eval_calls, 1)
        validate_weights.assert_called_once_with()

    def test_torch_hub_utmos_backend_preserves_explicit_trust_repo_option(self):
        model = FakeUTMOSModel(torch.tensor([4.0]))
        backend = TorchHubUTMOSBackend(
            device="cpu",
            allow_remote_code=True,
            load_options={"trust_repo": False},
        )

        with (
            patch("torch.hub.load", return_value=model) as load,
            patch.object(backend, "_validate_cached_weights") as validate_weights,
        ):
            backend.score(torch.zeros(16000), 16000)

        load.assert_called_once_with(
            f"{DEFAULT_UTMOS_REPOSITORY}:{DEFAULT_UTMOS_REVISION}",
            "utmos22_strong",
            trust_repo=False,
        )
        validate_weights.assert_called_once_with()

    def test_torch_hub_utmos_local_only_uses_pinned_cache(self):
        model = FakeUTMOSModel(torch.tensor([4.0]))
        backend = TorchHubUTMOSBackend(
            device="cpu",
            allow_remote_code=True,
            local_files_only=True,
            weights_sha256=(
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855"
            ),
        )
        with TemporaryDirectory() as root:
            hub = Path(root)
            repository = (
                hub
                / f"tarepan_SpeechMOS_{DEFAULT_UTMOS_REVISION}"
            )
            repository.mkdir()
            checkpoints = hub / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "utmos22_strong_step7459_v1.pt").touch()
            with (
                patch("torch.hub.get_dir", return_value=str(hub)),
                patch("torch.hub.load", return_value=model) as load,
            ):
                backend.score(torch.zeros(16000), 16000)

        load.assert_called_once_with(
            str(repository),
            "utmos22_strong",
            source="local",
            trust_repo=True,
        )

    def test_torch_hub_utmos_rejects_cached_weight_drift(self):
        backend = TorchHubUTMOSBackend()
        with TemporaryDirectory() as root:
            checkpoints = Path(root) / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "utmos22_strong_step7459_v1.pt").write_bytes(b"drift")
            with (
                patch("torch.hub.get_dir", return_value=root),
                self.assertRaisesRegex(RuntimeError, "pinned SHA-256"),
            ):
                backend._validate_cached_weights()


if __name__ == "__main__":
    unittest.main()
