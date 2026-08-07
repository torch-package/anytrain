import sys
import types
import unittest

import torch

from anytrain.translation import validate_translation_batch
from anytrain.translation.qwen import QwenTranslator


class TranslationProtocolTest(unittest.TestCase):
    def test_batch_requires_position_aligned_nonempty_strings(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_translation_batch(["hello"], ["English"], ["Chinese", "French"])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_translation_batch([], [], [])
        with self.assertRaisesRegex(ValueError, r"texts\[0\]"):
            validate_translation_batch(["  "], ["English"], ["Chinese"])


class QwenTranslatorTest(unittest.TestCase):
    def test_from_pretrained_omits_unset_revision(self):
        with _fake_transformers():
            QwenTranslator.from_pretrained("fake-qwen", device="cpu")

        self.assertEqual(FakeTokenizer.loaded.load_args, ("fake-qwen", {}))
        self.assertEqual(
            FakeModel.loaded.load_args,
            ("fake-qwen", {"device_map": "cpu"}),
        )

    def test_from_pretrained_forwards_tokenizer_hub_options(self):
        with _fake_transformers():
            QwenTranslator.from_pretrained(
                "fake-qwen",
                revision="main",
                cache_dir="storage/hf",
                local_files_only=True,
                token="hf-token",
                trust_remote_code=True,
                device="cpu",
                dtype=torch.bfloat16,
            )

        self.assertEqual(
            FakeTokenizer.loaded.load_args,
            (
                "fake-qwen",
                {
                    "cache_dir": "storage/hf",
                    "local_files_only": True,
                    "revision": "main",
                    "token": "hf-token",
                    "trust_remote_code": True,
                },
            ),
        )

    def test_from_pretrained_translates_plain_strings(self):
        with _fake_transformers():
            translator = QwenTranslator.from_pretrained(
                "fake-qwen",
                revision="main",
                device="cpu",
                max_new_tokens=32,
                dtype=torch.bfloat16,
            )

        outputs = translator.translate(
            ["hello", "world"],
            ["English", "English"],
            ["Chinese", "French"],
        )

        self.assertEqual(outputs, ["translation-0", "translation-1"])
        self.assertEqual(
            FakeModel.loaded.load_args,
            (
                "fake-qwen",
                {
                    "dtype": torch.bfloat16,
                    "revision": "main",
                    "device_map": "cpu",
                },
            ),
        )
        self.assertEqual(FakeTokenizer.loaded.load_args, ("fake-qwen", {"revision": "main"}))
        self.assertEqual(FakeTokenizer.loaded.padding_side, "left")
        self.assertEqual(FakeModel.loaded.generate_calls[0]["max_new_tokens"], 32)
        self.assertFalse(FakeModel.loaded.generate_calls[0]["do_sample"])
        self.assertIn("Translate this English text into Chinese", FakeTokenizer.loaded.prompts[0])

    def test_config_hash_tracks_generation_semantics(self):
        first = QwenTranslator(FakeModel(), FakeTokenizer())
        second = QwenTranslator(FakeModel(), FakeTokenizer())

        self.assertEqual(first.config_hash(), second.config_hash())


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    loaded = None

    def __init__(self, load_args=None):
        self.load_args = load_args
        self.padding_side = "right"
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.prompts = []
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

    def __call__(self, prompts, **_kwargs):
        self.prompts = list(prompts)
        return FakeBatch(input_ids=torch.tensor([[1, 2], [3, 4]]))

    def batch_decode(self, values, **_kwargs):
        return [f"</think> translation-{index}" for index in range(values.shape[0])]


class FakeModel:
    loaded = None

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
        module = types.ModuleType("transformers")
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


if __name__ == "__main__":
    unittest.main()
