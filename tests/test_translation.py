import pickle
import sys
import types
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import torch

from anytrain.translation import validate_translation_batch
from anytrain.translation.qwen import (
    QwenTranslationLoadOptions,
    QwenTranslationSpec,
    QwenTranslator,
)

_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class TranslationProtocolTest(unittest.TestCase):
    def test_batch_requires_position_aligned_nonempty_strings(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_translation_batch(["hello"], ["English"], ["Chinese", "French"])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_translation_batch([], [], [])
        with self.assertRaisesRegex(ValueError, r"texts\[0\]"):
            validate_translation_batch(["  "], ["English"], ["Chinese"])


class QwenTranslationSpecTest(unittest.TestCase):
    def test_semantic_identity_tracks_only_output_semantics(self):
        first = _spec()
        repeated = _spec()
        changed_dtype = _spec(dtype="float32")
        changed_generation = _spec(generation_options=(("repetition_penalty", 1.1),))

        self.assertEqual(first.semantic_id(), repeated.semantic_id())
        self.assertEqual(first.config_hash(), repeated.config_hash())
        self.assertNotEqual(first.semantic_id(), changed_dtype.semantic_id())
        self.assertNotEqual(first.semantic_id(), changed_generation.semantic_id())
        self.assertEqual(
            first.factory(QwenTranslationLoadOptions(cache_dir="first")).spec.semantic_id(),
            first.factory(
                QwenTranslationLoadOptions(cache_dir="second", local_files_only=True)
            ).spec.semantic_id(),
        )

    def test_spec_rejects_physical_model_path_and_reserved_generation_options(self):
        with self.assertRaisesRegex(ValueError, "40-character commit"):
            QwenTranslationSpec(revision="main")
        with self.assertRaisesRegex(ValueError, "absolute path"):
            QwenTranslationSpec(revision=_REVISION, model="/models/qwen")
        with self.assertRaisesRegex(ValueError, "reserved"):
            _spec(generation_options=(("do_sample", True),))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _spec(generation_options=(("temperature", 0.0), ("temperature", 0.1)))

    def test_factory_is_pickleable_and_lazy(self):
        factory = _spec().factory(QwenTranslationLoadOptions(cache_dir=Path("cache")))

        restored = pickle.loads(pickle.dumps(factory))

        self.assertEqual(restored, factory)
        self.assertEqual(restored.spec.semantic_id(), factory.spec.semantic_id())


class QwenTranslatorTest(unittest.TestCase):
    def setUp(self):
        FakeModel.loaded = None
        FakeTokenizer.loaded = None

    def test_from_pretrained_uses_spec_and_separate_load_options(self):
        with _fake_transformers():
            translator = QwenTranslator.from_pretrained(
                _spec(dtype="bfloat16"),
                device="cpu",
                load_options=QwenTranslationLoadOptions(cache_dir="storage/hf"),
            )

        self.assertEqual(
            FakeTokenizer.loaded.load_args,
            (
                "fake-qwen",
                {"cache_dir": "storage/hf", "revision": _REVISION},
            ),
        )
        self.assertEqual(
            FakeModel.loaded.load_args,
            (
                "fake-qwen",
                {
                    "cache_dir": "storage/hf",
                    "device_map": "cpu",
                    "dtype": torch.bfloat16,
                    "revision": _REVISION,
                },
            ),
        )
        self.assertEqual(translator.semantic_id(), translator.spec.semantic_id())

    def test_from_pretrained_resolves_cached_repo_for_local_only_loads(self):
        with (
            _fake_transformers(),
            patch(
                "huggingface_hub.snapshot_download",
                return_value="/cache/qwen-snapshot",
            ) as snapshot_download,
        ):
            QwenTranslator.from_pretrained(
                _spec(),
                device="cpu",
                load_options=QwenTranslationLoadOptions(
                    cache_dir="storage/hf",
                    local_files_only=True,
                ),
            )

        snapshot_download.assert_called_once_with(
            repo_id="fake-qwen",
            cache_dir="storage/hf",
            local_files_only=True,
            revision=_REVISION,
        )
        self.assertEqual(FakeTokenizer.loaded.load_args[0], "/cache/qwen-snapshot")
        self.assertEqual(FakeModel.loaded.load_args[0], "/cache/qwen-snapshot")

    def test_lazy_factory_loads_only_when_called(self):
        factory = _spec().factory()

        self.assertIsNone(FakeModel.loaded)
        with _fake_transformers():
            translator = factory("cpu")

        self.assertIsInstance(translator, QwenTranslator)
        self.assertIsNotNone(FakeModel.loaded)

    def test_from_pretrained_translates_plain_strings(self):
        with _fake_transformers():
            translator = QwenTranslator.from_pretrained(
                _spec(max_new_tokens=32),
                device="cpu",
            )

        outputs = translator.translate(
            ["hello", "world"],
            ["English", "English"],
            ["Chinese", "French"],
        )

        self.assertEqual(outputs, ["translation-0", "translation-1"])
        self.assertEqual(FakeTokenizer.loaded.padding_side, "left")
        self.assertEqual(FakeModel.loaded.generate_calls[0]["max_new_tokens"], 32)
        self.assertFalse(FakeModel.loaded.generate_calls[0]["do_sample"])
        self.assertIn("Translate this English text into Chinese", FakeTokenizer.loaded.prompts[0])

    def test_token_estimator_loads_no_model_and_uses_backend_prompt(self):
        estimator = _spec(max_new_tokens=32).token_estimator()

        self.assertFalse(estimator.tokenizer_loaded)
        with _fake_transformers():
            prompt_tokens = estimator.prompt_tokens(
                "hello world",
                source_language="English",
                target_language="Chinese",
            )
            planned_tokens = estimator.planned_tokens(
                "hello world",
                source_language="English",
                target_language="Chinese",
            )
            decoded_tokens = estimator.decoded_target_tokens("translated text")

        self.assertTrue(estimator.tokenizer_loaded)
        self.assertIsNone(FakeModel.loaded)
        self.assertEqual(planned_tokens, prompt_tokens + 32)
        self.assertEqual(decoded_tokens, 2)
        self.assertIn("Translate this English text into Chinese", FakeTokenizer.loaded.texts[0])


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    loaded: ClassVar[Any] = None

    def __init__(self, load_args=None):
        self.load_args = load_args
        self.padding_side = "right"
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.prompts = []
        self.texts = []
        type(self).loaded = self

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return cls((model, kwargs))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        self.assert_template_options = (tokenize, add_generation_prompt, enable_thinking)
        return messages[-1]["content"]

    def __call__(self, values, **_kwargs):
        if isinstance(values, str):
            self.texts.append(values)
            return {"input_ids": list(range(max(1, len(values.split()))))}
        self.prompts = list(values)
        return FakeBatch(input_ids=torch.tensor([[1, 2], [3, 4]]))

    def batch_decode(self, values, **_kwargs):
        return [f"</think> translation-{index}" for index in range(values.shape[0])]


class FakeModel:
    loaded: ClassVar[Any] = None

    def __init__(self, load_args=None):
        self.load_args = load_args
        self.device = torch.device("cpu")
        self.generate_calls = []
        type(self).loaded = self

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return cls((model, kwargs))

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        batch = kwargs["input_ids"].shape[0]
        prefix = kwargs["input_ids"]
        suffix = torch.arange(batch, dtype=torch.int64).reshape(batch, 1)
        return torch.cat((prefix, suffix), dim=1)


class _fake_transformers:
    def __enter__(self):
        module: Any = types.ModuleType("transformers")
        module.AutoModelForCausalLM = FakeModel
        module.AutoTokenizer = FakeTokenizer
        self.previous = sys.modules.get("transformers")
        sys.modules["transformers"] = module
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = self.previous


def _spec(**options) -> QwenTranslationSpec:
    return QwenTranslationSpec(
        revision=_REVISION,
        model="fake-qwen",
        **options,
    )


if __name__ == "__main__":
    unittest.main()
