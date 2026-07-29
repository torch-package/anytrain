import sys
import types
import unittest

import torch

from anytrain.tts import TTSOptions
from anytrain.tts.qwen import QwenCustomVoiceTTS, QwenCustomVoiceTTSConfig


class QwenTTSAdapterTest(unittest.TestCase):
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


class FakeQwen3TTSModel:
    loaded = None

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
                [torch.tensor([float(index * 2), float(index * 2 + 1)]) for index in range(len(text))],
                16000,
            )
        return torch.tensor([1.0, 2.0]), 16000


class _fake_qwen_tts:
    def __init__(self) -> None:
        self.previous = {}

    def __enter__(self):
        module = types.ModuleType("qwen_tts")
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


if __name__ == "__main__":
    unittest.main()
