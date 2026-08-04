import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from anytrain.tts import (
    EncodedAudioReference,
    TTSOptions,
    TTSOutput,
    WaveformReference,
)
from anytrain.tts.moss import DEFAULT_CODEC_MODEL, MossTTS, MossTTSConfig


class TTSProtocolTest(unittest.TestCase):
    def test_output_requires_channel_time_waveform(self):
        with self.assertRaisesRegex(ValueError, "channels, time"):
            TTSOutput(
                waveform=torch.zeros(1, 1, 8),
                sample_rate=24000,
                duration=1.0,
            )

    def test_options_validate_sampling_parameters(self):
        with self.assertRaisesRegex(ValueError, "top_p"):
            TTSOptions(top_p=1.2)

        with self.assertRaisesRegex(ValueError, "temperature"):
            TTSOptions(temperature=0.0)

    def test_options_merge_preserves_explicit_fields_and_none(self):
        defaults = TTSOptions(speaker="default", language="Chinese", seed=7)
        configured = TTSOptions(language="English").merged(TTSOptions(speaker=None))

        merged = defaults.merged(configured)

        self.assertIsNone(merged.speaker)
        self.assertEqual(merged.language, "English")
        self.assertEqual(merged.seed, 7)

    def test_waveform_reference_requires_nonempty_channel_time_audio(self):
        with self.assertRaisesRegex(ValueError, "channels, time"):
            WaveformReference(torch.zeros(8), 24000)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            WaveformReference(torch.zeros(1, 0), 24000)
        with self.assertRaisesRegex(TypeError, "sample_rate"):
            WaveformReference(torch.zeros(1, 8), True)

    def test_encoded_reference_validates_one_suffix_and_hides_bytes(self):
        reference = EncodedAudioReference(b"secret-audio", ".flac")

        self.assertNotIn("secret-audio", repr(reference))
        with self.assertRaisesRegex(ValueError, "one filename suffix|file suffix"):
            EncodedAudioReference(b"audio", ".tar.gz")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            EncodedAudioReference(b"", ".wav")


class MossTTSImportTest(unittest.TestCase):
    def test_default_model_uses_moss_tts_v15_checkpoint(self):
        self.assertEqual(
            MossTTSConfig().model,
            "OpenMOSS-Team/MOSS-TTS-v1.5",
        )
        self.assertEqual(MossTTSConfig().codec_model, DEFAULT_CODEC_MODEL)

    def test_import_tts_does_not_import_transformers_dependency(self):
        sys.modules.pop("transformers", None)

        import anytrain.tts

        self.assertIn("TTSOutput", anytrain.tts.__all__)
        self.assertNotIn("transformers", sys.modules)

    def test_from_pretrained_requires_optional_dependency_when_missing(self):
        with (
            patch(
                "anytrain.tts.moss._loading.load_transformers_auto_model_class",
                side_effect=ImportError("transformers missing"),
            ),
            self.assertRaisesRegex(
                ImportError,
                r'pip install "torchaudio>=2.0" transformers',
            ),
        ):
            MossTTS.from_pretrained(local_files_only=True)


class MossTTSAdapterTest(unittest.TestCase):
    def test_from_pretrained_loads_v15_model_and_processor(self):
        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                return FakeMossModel(load_args=(model, kwargs))

        class FakeAutoProcessor:
            calls = []

            @classmethod
            def from_pretrained(cls, model, **kwargs):
                cls.calls.append((model, kwargs))
                return FakeMossProcessor()

        with (
            patch(
                "anytrain.tts.moss._loading.load_transformers_auto_model_class",
                return_value=FakeAutoModel,
            ),
            patch(
                "anytrain.tts.moss._loading.load_transformers_auto_processor_class",
                return_value=FakeAutoProcessor,
            ),
            patch(
                "anytrain.tts.moss._loading.pretrained_source",
                side_effect=lambda source, **kwargs: str(source),
            ),
        ):
            tts = MossTTS.from_pretrained(
                "OpenMOSS-Team/MOSS-TTS-v1.5",
                cache_dir="storage/moss",
                codec_model="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
                device="cpu",
                local_files_only=True,
                sample_rate=16000,
                language="Chinese",
                revision="main",
                trust_remote_code=True,
                runtime_kwargs={"do_sample": False},
                torch_dtype="bfloat16",
            )

        self.assertEqual(tts.config.model, "OpenMOSS-Team/MOSS-TTS-v1.5")
        self.assertEqual(tts.config.codec_model, "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
        self.assertEqual(tts.config.sample_rate, 16000)
        self.assertEqual(tts.sample_rate, 24000)
        self.assertEqual(tts.config.runtime_kwargs, {"do_sample": False})
        self.assertEqual(tts.model_loaded_from, "transformers")
        self.assertEqual(
            tts.model.load_args,
            (
                "OpenMOSS-Team/MOSS-TTS-v1.5",
                {
                    "cache_dir": "storage/moss",
                    "local_files_only": True,
                    "revision": "main",
                    "torch_dtype": "bfloat16",
                    "trust_remote_code": True,
                },
            ),
        )
        self.assertEqual(
            FakeAutoProcessor.calls,
            [
                (
                    "OpenMOSS-Team/MOSS-TTS-v1.5",
                    {
                        "codec_path": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
                        "revision": "main",
                        "trust_remote_code": True,
                    },
                )
            ],
        )

    def test_synthesize_uses_processor_generation(self):
        model = FakeMossModel()
        processor = FakeMossProcessor()
        tts = MossTTS(
            model=model,
            processor=processor,
            config=MossTTSConfig(sample_rate=24000, language="Chinese"),
            device="cpu",
        )

        output = tts.synthesize(
            "hello",
            TTSOptions(
                max_new_tokens=13,
                temperature=0.8,
                top_p=0.95,
                seed=7,
            ),
            reference_audio="assets/audio/zh_1.wav",
        )

        self.assertEqual(tuple(output.waveform.shape), (1, 5))
        self.assertEqual(output.sample_rate, 24000)
        self.assertEqual(output.meta["backend"], "OpenMOSS-Team/MOSS-TTS-v1.5")
        self.assertEqual(
            processor.messages,
            [
                {
                    "language": "Chinese",
                    "reference": ["assets/audio/zh_1.wav"],
                    "text": "hello",
                }
            ],
        )
        self.assertEqual(processor.batch_modes, ["generation"])
        self.assertEqual(model.calls[0]["max_new_tokens"], 13)
        self.assertEqual(model.calls[0]["temperature"], 0.8)
        self.assertEqual(model.calls[0]["top_p"], 0.95)
        self.assertEqual(model.calls[0]["do_sample"], True)
        self.assertTrue(torch.equal(model.calls[0]["input_ids"], torch.tensor([[1, 2]])))

    def test_synthesize_accepts_text_batch(self):
        model = FakeBatchMossModel()
        processor = FakeMossProcessor()
        tts = MossTTS(
            model=model,
            processor=processor,
            config=MossTTSConfig(sample_rate=24000, language="Chinese"),
            device="cpu",
        )

        outputs = tts.synthesize(
            ["hello", "world"],
            TTSOptions(max_new_tokens=13),
            reference_audios=["assets/source.wav", "assets/target.wav"],
        )

        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.equal(outputs[0].waveform, torch.tensor([[0.0, 1.0, 2.0, 3.0]])))
        self.assertTrue(torch.equal(outputs[1].waveform, torch.tensor([[4.0, 5.0, 6.0, 7.0]])))
        self.assertEqual([message["text"] for message in processor.messages], ["hello", "world"])
        self.assertEqual(
            [message["reference"] for message in processor.messages],
            [["assets/source.wav"], ["assets/target.wav"]],
        )
        self.assertEqual(processor.batch_modes, ["generation"])
        self.assertEqual(tuple(model.calls[0]["input_ids"].shape), (2, 2))
        self.assertEqual(model.calls[0]["max_new_tokens"], 13)

    def test_synthesize_batch_maps_legacy_positional_references_with_warning(self):
        model = FakeBatchMossModel()
        processor = FakeMossProcessor()
        tts = MossTTS(model=model, processor=processor)

        with self.assertWarnsRegex(DeprecationWarning, "reference_audios"):
            tts.synthesize(["hello", "world"], None, ["assets/one.wav", "assets/two.wav"])

        self.assertEqual(
            [message["reference"] for message in processor.messages],
            [["assets/one.wav"], ["assets/two.wav"]],
        )

    def test_synthesize_rejects_empty_text_batch(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "text batch"):
            tts.synthesize([])

    def test_synthesize_batch_rejects_decode_length_mismatch(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "text batch length"):
            tts.synthesize(["hello", "world"])

    def test_synthesize_one_rejects_extra_decoded_outputs(self):
        class ExtraOutputProcessor(FakeMossProcessor):
            def decode(self, value):
                message = FakeProcessorMessage(torch.zeros(4))
                return [message, message]

        tts = MossTTS(model=FakeMossModel(), processor=ExtraOutputProcessor())

        with self.assertRaisesRegex(ValueError, "text batch length"):
            tts.synthesize("hello")

    def test_synthesize_batch_rejects_reference_length_mismatch(self):
        tts = MossTTS(model=FakeBatchMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "reference_audios length"):
            tts.synthesize(["hello", "world"], reference_audios=["assets/one.wav"])

    def test_path_alias_warns_and_conflicts_with_canonical_reference(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertWarnsRegex(DeprecationWarning, "reference_audio_path"):
            tts.synthesize("hello", reference_audio_path="assets/prompt.wav")
        with self.assertRaisesRegex(TypeError, "cannot both be passed"):
            tts.synthesize(
                "hello",
                reference_audio="assets/canonical.wav",
                reference_audio_path="assets/compat.wav",
            )

    def test_encoded_reference_exists_through_decode_then_is_removed(self):
        processor = ReferenceCheckingProcessor(expected=b"encoded-audio", suffix=".flac")
        model = ReferenceCheckingModel(processor.reference_paths)
        tts = MossTTS(model=model, processor=processor)

        tts.synthesize(
            "hello",
            reference_audio=EncodedAudioReference(b"encoded-audio", ".flac"),
        )

        self.assertEqual(len(processor.reference_paths), 1)
        self.assertFalse(processor.reference_paths[0].exists())

    def test_waveform_reference_is_saved_as_clamped_cpu_float32(self):
        processor = ReferenceCheckingProcessor(expected=b"waveform", suffix=".wav")
        model = ReferenceCheckingModel(processor.reference_paths)
        tts = MossTTS(model=model, processor=processor)
        saved: dict[str, object] = {}

        def save(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
            saved["waveform"] = waveform
            saved["sample_rate"] = sample_rate
            Path(path).write_bytes(b"waveform")

        torchaudio = SimpleNamespace(save=save)
        with patch.dict(sys.modules, {"torchaudio": torchaudio}):
            tts.synthesize(
                "hello",
                reference_audio=WaveformReference(
                    torch.tensor([[-2.0, 0.5, 2.0]], dtype=torch.bfloat16),
                    16000,
                ),
            )

        waveform = saved["waveform"]
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(waveform.device.type, "cpu")
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(torch.equal(waveform, torch.tensor([[-1.0, 0.5, 1.0]])))
        self.assertEqual(saved["sample_rate"], 16000)
        self.assertFalse(processor.reference_paths[0].exists())

    def test_reference_temp_file_is_removed_when_generation_fails(self):
        processor = ReferenceCheckingProcessor(expected=b"encoded-audio", suffix=".wav")
        tts = MossTTS(model=FailingReferenceModel(processor.reference_paths), processor=processor)

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            tts.synthesize(
                "hello",
                reference_audio=EncodedAudioReference(b"encoded-audio", ".wav"),
            )

        self.assertFalse(processor.reference_paths[0].exists())

    def test_runtime_kwargs_do_not_leak_load_kwargs(self):
        model = FakeMossModel()
        processor = FakeMossProcessor()
        tts = MossTTS(
            model=model,
            processor=processor,
            config=MossTTSConfig(
                load_kwargs={"torch_dtype": "float32"},
                runtime_kwargs={"do_sample": False},
            ),
        )

        tts.synthesize("hello", reference_audio_path="assets/prompt.wav")

        self.assertEqual(processor.messages[0]["reference"], ["assets/prompt.wav"])
        self.assertEqual(model.calls[0]["do_sample"], False)
        self.assertNotIn("torch_dtype", model.calls[0])

    def test_explicit_extra_overrides_config_runtime_kwargs(self):
        model = FakeMossModel()
        processor = FakeMossProcessor()
        tts = MossTTS(
            model=model,
            processor=processor,
            config=MossTTSConfig(runtime_kwargs={"do_sample": False, "max_new_tokens": 12}),
        )

        tts.synthesize(
            "hello",
            TTSOptions(extra={}),
            reference_audio_path="assets/prompt.wav",
        )

        self.assertNotIn("do_sample", model.calls[0])
        self.assertNotIn("max_new_tokens", model.calls[0])
        self.assertEqual(processor.messages[0]["reference"], ["assets/prompt.wav"])

    def test_v15_rejects_speaker_ids(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "speaker"):
            tts.synthesize("hello", TTSOptions(speaker="alice"))

    def test_v15_rejects_reference_runtime_kwargs(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(TypeError, "unsupported"):
            tts.synthesize("hello", prompt_audio_path="assets/prompt.wav")

    def test_seeded_generation_restores_global_rng_state(self):
        model = FakeRandomMossModel()
        tts = MossTTS(model=model, processor=FakeMossProcessor())
        torch.manual_seed(1234)
        state = torch.random.get_rng_state()

        first = tts.synthesize("hello", TTSOptions(seed=7))
        after_first = torch.random.get_rng_state()
        second = tts.synthesize("hello", TTSOptions(seed=7))

        self.assertTrue(torch.equal(state, after_first))
        self.assertTrue(torch.equal(first.waveform, second.waveform))

    def test_config_hash_is_stable_and_uses_generation_identity(self):
        first = MossTTS(
            model=FakeMossModel(),
            processor=FakeMossProcessor(),
            config=MossTTSConfig(language="Chinese"),
        )
        second = MossTTS(
            model=FakeMossModel(),
            processor=FakeMossProcessor(),
            config=MossTTSConfig(language="Chinese"),
        )
        changed = MossTTS(
            model=FakeMossModel(),
            processor=FakeMossProcessor(),
            config=MossTTSConfig(language="English"),
        )

        self.assertEqual(first.config_hash(), second.config_hash())
        self.assertNotEqual(first.config_hash(), changed.config_hash())

    def test_processor_outputs_reject_none_message(self):
        from anytrain.tts.moss._output import processor_outputs

        with self.assertRaisesRegex(RuntimeError, "empty output at batch index 1"):
            processor_outputs(
                [FakeProcessorMessage(torch.arange(3, dtype=torch.float32)), None],
                expected_count=2,
                sample_rate=24000,
                backend="moss",
            )


class FakeMossModel:
    def __init__(self, load_args=None):
        self.load_args = load_args
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[9, 10]])


class FakeRandomMossModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.rand(1, 4)


class FakeBatchMossModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        return torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4)


class FakeProcessorMessage:
    def __init__(self, audio: torch.Tensor) -> None:
        self.audio_codes_list = [audio]


class FakeProcessorConfig:
    sampling_rate = 24000


class FakeMossProcessor:
    model_config = FakeProcessorConfig()

    def __init__(self):
        self.messages = []
        self.batch_modes = []

    def build_user_message(self, **kwargs):
        self.messages.append(kwargs)
        return kwargs

    def __call__(self, conversations, **kwargs):
        self.batch_modes.append(kwargs["mode"])
        batch_size = len(conversations)
        return {
            "input_ids": torch.arange(1, batch_size * 2 + 1).reshape(batch_size, 2),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }

    def decode(self, value):
        self.decoded = value
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[-1] == 4:
            return [FakeProcessorMessage(row) for row in value]
        return [FakeProcessorMessage(torch.arange(5, dtype=torch.float32))]


class ReferenceCheckingProcessor(FakeMossProcessor):
    def __init__(self, *, expected: bytes, suffix: str) -> None:
        super().__init__()
        self.expected = expected
        self.suffix = suffix
        self.reference_paths: list[Path] = []

    def build_user_message(self, **kwargs):
        path = Path(kwargs["reference"][0])
        self._assert_reference(path)
        self.reference_paths.append(path)
        return super().build_user_message(**kwargs)

    def __call__(self, conversations, **kwargs):
        for path in self.reference_paths:
            self._assert_reference(path)
        return super().__call__(conversations, **kwargs)

    def decode(self, value):
        for path in self.reference_paths:
            self._assert_reference(path)
        return super().decode(value)

    def _assert_reference(self, path: Path) -> None:
        if path.suffix != self.suffix:
            raise AssertionError(f"unexpected reference suffix: {path.suffix}")
        if path.read_bytes() != self.expected:
            raise AssertionError("unexpected reference contents")


class ReferenceCheckingModel(FakeMossModel):
    def __init__(self, reference_paths: list[Path]) -> None:
        super().__init__()
        self.reference_paths = reference_paths

    def generate(self, **kwargs):
        if not self.reference_paths or not all(path.is_file() for path in self.reference_paths):
            raise AssertionError("reference file did not survive through generate")
        return super().generate(**kwargs)


class FailingReferenceModel(ReferenceCheckingModel):
    def generate(self, **kwargs):
        super().generate(**kwargs)
        raise RuntimeError("generation failed")


if __name__ == "__main__":
    unittest.main()
