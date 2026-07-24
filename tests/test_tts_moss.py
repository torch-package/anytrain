import sys
import unittest
from unittest.mock import patch

import torch

from anytrain.tts import TTSOptions, TTSOutput, TTSTokens
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

    def test_tokens_build_model_inputs(self):
        tokens = TTSTokens(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
            extra_tensors={"prompt_ids": torch.tensor([[3]])},
        )

        self.assertEqual(set(tokens.model_inputs()), {"attention_mask", "input_ids", "prompt_ids"})


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
                "anytrain.tts.moss.tts.load_transformers_auto_model_class",
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
                "anytrain.tts.moss.tts.load_transformers_auto_model_class",
                return_value=FakeAutoModel,
            ),
            patch(
                "anytrain.tts.moss.tts.load_transformers_auto_processor_class",
                return_value=FakeAutoProcessor,
            ),
            patch(
                "anytrain.tts.moss.tts._pretrained_source",
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
        self.assertEqual(tts.sample_rate, 16000)
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
            reference_audio_path="assets/audio/zh_1.wav",
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
            reference_audio_paths=["assets/source.wav", "assets/target.wav"],
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

    def test_synthesize_rejects_empty_text_batch(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "text batch"):
            tts.synthesize([])

    def test_synthesize_batch_rejects_decode_length_mismatch(self):
        tts = MossTTS(model=FakeMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "text batch length"):
            tts.synthesize(["hello", "world"])

    def test_synthesize_batch_rejects_reference_length_mismatch(self):
        tts = MossTTS(model=FakeBatchMossModel(), processor=FakeMossProcessor())

        with self.assertRaisesRegex(ValueError, "reference_audio_paths length"):
            tts.synthesize(["hello", "world"], reference_audio_paths=["assets/one.wav"])

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


if __name__ == "__main__":
    unittest.main()
