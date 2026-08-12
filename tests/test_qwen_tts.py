import pickle
import sys
import types
import unittest
from pathlib import Path
from typing import Any, ClassVar, cast

import torch

from anytrain.tts import TTSOptions, TTSVoiceMode
from anytrain.tts.qwen import (
    CUSTOM_VOICE_PROMPT_PROTOCOL,
    QwenCustomVoicePromptEstimator,
    QwenCustomVoiceTTS,
    QwenCustomVoiceTTSConfig,
)

_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"


class QwenTTSAdapterTest(unittest.TestCase):
    def test_capabilities_expose_speaker_ids_only(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained("fake-qwen")

        self.assertTrue(tts.capabilities.supports(TTSVoiceMode.SPEAKER_ID))
        self.assertFalse(tts.capabilities.supports(TTSVoiceMode.REFERENCE_AUDIO))
        self.assertEqual(tts.capabilities.speaker_ids, ("Vivian", "Ryan"))

    def test_from_pretrained_loads_qwen_tts_model(self):
        FakeQwen3TTSModel.loaded = None
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained(
                "fake-qwen",
                device="cpu",
                runtime_kwargs={"top_k": 5},
            )

        self.assertEqual(tts.config.model, "fake-qwen")
        self.assertEqual(tts.config.runtime_kwargs, {"top_k": 5})
        self.assertEqual(FakeQwen3TTSModel.loaded.load_args, ("fake-qwen", {"device": "cpu"}))

    def test_synthesize_custom_voice_accepts_per_sample_speakers(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained("fake-qwen")

        outputs = tts.synthesize_custom_voice(
            ["hello", "world"],
            speakers=["Vivian", "Ryan"],
            languages=["English", "Chinese"],
            options=TTSOptions(max_new_tokens=128),
        )
        assert isinstance(outputs, list)

        self.assertEqual(
            FakeQwen3TTSModel.loaded.generate_calls,
            [
                {
                    "text": ["hello", "world"],
                    "language": ["English", "Chinese"],
                    "speaker": ["Vivian", "Ryan"],
                    "instruct": None,
                    "max_new_tokens": 128,
                }
            ],
        )
        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.equal(outputs[0].waveform, torch.tensor([[0.0, 1.0]])))
        self.assertTrue(torch.equal(outputs[1].waveform, torch.tensor([[2.0, 3.0]])))
        self.assertEqual(outputs[0].meta["speaker"], "Vivian")
        self.assertEqual(outputs[1].meta["speaker"], "Ryan")

    def test_synthesize_merges_options_and_allows_empty_extra_to_clear_defaults(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained(
                "fake-qwen",
                speaker="Vivian",
                runtime_kwargs={"top_k": 5},
            )

        tts.synthesize("hello", TTSOptions(max_new_tokens=128, extra={}))

        self.assertEqual(
            FakeQwen3TTSModel.loaded.generate_calls,
            [
                {
                    "text": "hello",
                    "language": "Auto",
                    "speaker": "Vivian",
                    "instruct": None,
                    "max_new_tokens": 128,
                }
            ],
        )

    def test_synthesize_rejects_unsupported_seed_and_sample_rate(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained("fake-qwen", speaker="Vivian")

        with self.assertRaisesRegex(ValueError, "does not support seed"):
            tts.synthesize("hello", TTSOptions(seed=7))
        with self.assertRaisesRegex(ValueError, "does not support sample_rate"):
            tts.synthesize("hello", TTSOptions(sample_rate=16000))

    def test_synthesize_reference_is_keyword_only(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained("fake-qwen", speaker="Vivian")

        with self.assertRaises(TypeError):
            cast(Any, tts).synthesize("hello", None, "assets/reference.wav")

    def test_rejects_unsupported_speaker_id(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained("fake-qwen")

        with self.assertRaisesRegex(ValueError, "unsupported Qwen speaker ids"):
            tts.synthesize_custom_voice("hello", speakers="Unknown")

    def test_rejects_extra_generated_waveforms(self):
        with _fake_qwen_tts():
            tts = QwenCustomVoiceTTS.from_pretrained("fake-qwen")

        tts.model.generate_custom_voice = lambda **kwargs: (
            [torch.zeros(2), torch.ones(2), torch.full((2,), 2.0)],
            16000,
        )
        with self.assertRaisesRegex(RuntimeError, "output count"):
            tts.synthesize_custom_voice(
                ["hello", "world"],
                speakers=["Vivian", "Ryan"],
            )


class QwenTTSConfigTest(unittest.TestCase):
    def test_default_model_is_customvoice(self):
        self.assertIn("CustomVoice", QwenCustomVoiceTTSConfig().model)

    def test_config_hash_is_stable_for_nested_mapping_order(self):
        first = QwenCustomVoiceTTS(
            model=object(),
            config=QwenCustomVoiceTTSConfig(
                load_kwargs={"nested": {"left": 1, "right": 2}},
            ),
        )
        second = QwenCustomVoiceTTS(
            model=object(),
            config=QwenCustomVoiceTTSConfig(
                load_kwargs={"nested": {"right": 2, "left": 1}},
            ),
        )

        self.assertEqual(first.config_hash(), second.config_hash())

    def test_config_hash_ignores_replica_device_but_keeps_semantic_load_options(self):
        first = QwenCustomVoiceTTS(
            model=object(),
            config=QwenCustomVoiceTTSConfig(
                load_kwargs={
                    "device": "cuda:0",
                    "device_map": "cuda:0",
                    "dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                },
            ),
        )
        replica = QwenCustomVoiceTTS(
            model=object(),
            config=QwenCustomVoiceTTSConfig(
                load_kwargs={
                    "device": "cuda:1",
                    "device_map": "cuda:1",
                    "dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                },
            ),
        )
        changed_dtype = QwenCustomVoiceTTS(
            model=object(),
            config=QwenCustomVoiceTTSConfig(
                load_kwargs={
                    "device": "cuda:1",
                    "device_map": "cuda:1",
                    "dtype": "float32",
                    "attn_implementation": "sdpa",
                },
            ),
        )
        changed_attention = QwenCustomVoiceTTS(
            model=object(),
            config=QwenCustomVoiceTTSConfig(
                load_kwargs={
                    "device": "cuda:1",
                    "device_map": "cuda:1",
                    "dtype": "bfloat16",
                    "attn_implementation": "flash_attention_2",
                },
            ),
        )

        self.assertEqual(first.config_hash(), replica.config_hash())
        self.assertNotEqual(first.config_hash(), changed_dtype.config_hash())
        self.assertNotEqual(first.config_hash(), changed_attention.config_hash())


class QwenCustomVoicePromptEstimatorTest(unittest.TestCase):
    def setUp(self):
        FakeAutoProcessor.loaded = None
        FakeAutoProcessor.output = None

    def test_loads_only_the_processor_and_uses_the_exact_upstream_prompt(self):
        estimator = QwenCustomVoicePromptEstimator(
            model="fake-qwen",
            revision=_REVISION,
            cache_dir=Path("storage/hf"),
            local_files_only=True,
        )

        self.assertFalse(estimator.processor_loaded)
        with _fake_transformers():
            count = estimator.estimate("hello")

        self.assertTrue(estimator.processor_loaded)
        self.assertEqual(count, 6)
        assert FakeAutoProcessor.loaded is not None
        self.assertEqual(
            FakeAutoProcessor.loaded.load_args,
            (
                "fake-qwen",
                {
                    "cache_dir": "storage/hf",
                    "fix_mistral_regex": True,
                    "local_files_only": True,
                    "revision": _REVISION,
                },
            ),
        )
        self.assertEqual(
            FakeAutoProcessor.loaded.calls,
            [
                {
                    "text": ("<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"),
                    "return_tensors": "pt",
                    "padding": True,
                }
            ],
        )

    def test_identity_excludes_physical_loading_policy(self):
        first = QwenCustomVoicePromptEstimator(
            model="fake-qwen",
            revision=_REVISION,
            cache_dir="first",
            local_files_only=False,
        )
        replica = QwenCustomVoicePromptEstimator(
            model="fake-qwen",
            revision=_REVISION,
            cache_dir="second",
            local_files_only=True,
        )
        changed_model = QwenCustomVoicePromptEstimator(
            model="other-qwen",
            revision=_REVISION,
        )
        changed_revision = QwenCustomVoicePromptEstimator(
            model="fake-qwen",
            revision="1" * 40,
        )

        self.assertEqual(first.prompt_protocol, CUSTOM_VOICE_PROMPT_PROTOCOL)
        self.assertEqual(first.identity, replica.identity)
        self.assertNotEqual(first.identity, changed_model.identity)
        self.assertNotEqual(first.identity, changed_revision.identity)

    def test_pickle_discards_a_loaded_processor_and_remains_lazy(self):
        estimator = QwenCustomVoicePromptEstimator(
            model="fake-qwen",
            revision=_REVISION,
        )
        restored_before_load = pickle.loads(pickle.dumps(estimator))
        self.assertFalse(restored_before_load.processor_loaded)
        self.assertEqual(restored_before_load.identity, estimator.identity)

        with _fake_transformers():
            estimator.estimate("hello")
        self.assertTrue(estimator.processor_loaded)

        restored_after_load = pickle.loads(pickle.dumps(estimator))
        self.assertFalse(restored_after_load.processor_loaded)
        self.assertEqual(restored_after_load.identity, estimator.identity)
        with _fake_transformers():
            self.assertEqual(restored_after_load.estimate("hello"), 6)

    def test_rejects_unpinned_or_physical_semantics_and_bad_input(self):
        with self.assertRaisesRegex(ValueError, "40-character commit"):
            QwenCustomVoicePromptEstimator(revision="main")
        with self.assertRaisesRegex(ValueError, "logical artifact id"):
            QwenCustomVoicePromptEstimator(
                model="/models/qwen",
                revision=_REVISION,
            )
        with self.assertRaisesRegex(ValueError, "unsupported.*prompt protocol"):
            QwenCustomVoicePromptEstimator(
                revision=_REVISION,
                prompt_protocol="custom",
            )
        with self.assertRaisesRegex(TypeError, "local_files_only"):
            QwenCustomVoicePromptEstimator(
                revision=_REVISION,
                local_files_only=cast(Any, 1),
            )

        estimator = QwenCustomVoicePromptEstimator(
            model="fake-qwen",
            revision=_REVISION,
        )
        with _fake_transformers(), self.assertRaisesRegex(ValueError, "must not be empty"):
            estimator.estimate("  ")

    def test_rejects_bad_processor_outputs(self):
        cases = (
            (object(), TypeError, "mapping"),
            ({"input_ids": [1, 2]}, TypeError, "Tensor"),
            ({"input_ids": torch.ones(1, 2, 3)}, ValueError, "rank 1 or rank 2"),
            ({"input_ids": torch.ones(2, 3)}, ValueError, "exactly one text"),
            (
                {"input_ids": torch.empty((1, 0), dtype=torch.long)},
                ValueError,
                "must not be empty",
            ),
        )
        for output, error, message in cases:
            with self.subTest(message=message):
                FakeAutoProcessor.output = output
                estimator = QwenCustomVoicePromptEstimator(
                    model="fake-qwen",
                    revision=_REVISION,
                )
                with _fake_transformers(), self.assertRaisesRegex(error, message):
                    estimator.estimate("hello")


class FakeAutoProcessor:
    loaded: ClassVar[Any] = None
    output: ClassVar[object | None] = None

    def __init__(self, load_args):
        self.load_args = load_args
        self.calls = []
        self.unpickleable = lambda: None
        type(self).loaded = self

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return cls((model, kwargs))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.output is not None:
            return self.output
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])}


class FakeQwen3TTSModel:
    loaded: ClassVar[Any] = None

    def __init__(self, load_args):
        self.load_args = load_args
        self.generate_calls = []
        type(self).loaded = self

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return cls((model, kwargs))

    def get_supported_speakers(self):
        return ("Vivian", "Ryan")

    def generate_custom_voice(self, **kwargs):
        self.generate_calls.append(kwargs)
        text = kwargs["text"]
        if isinstance(text, list):
            return (
                [
                    torch.tensor([float(index * 2), float(index * 2 + 1)])
                    for index in range(len(text))
                ],
                16000,
            )
        return torch.tensor([1.0, 2.0]), 16000


class _fake_qwen_tts:
    def __init__(self) -> None:
        self.previous = {}

    def __enter__(self):
        module: Any = types.ModuleType("qwen_tts")
        module.Qwen3TTSModel = FakeQwen3TTSModel
        self.previous = {"qwen_tts": sys.modules.get("qwen_tts")}
        sys.modules["qwen_tts"] = module
        return self

    def __exit__(self, exc_type, exc, tb):
        previous = self.previous["qwen_tts"]
        if previous is None:
            sys.modules.pop("qwen_tts", None)
        else:
            sys.modules["qwen_tts"] = previous


class _fake_transformers:
    def __init__(self) -> None:
        self.previous = {}

    def __enter__(self):
        module: Any = types.ModuleType("transformers")
        module.AutoProcessor = FakeAutoProcessor
        self.previous = {"transformers": sys.modules.get("transformers")}
        sys.modules["transformers"] = module
        return self

    def __exit__(self, exc_type, exc, tb):
        previous = self.previous["transformers"]
        if previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = previous


if __name__ == "__main__":
    unittest.main()
