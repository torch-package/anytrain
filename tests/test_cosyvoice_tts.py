import os
import sys
import threading
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar
from unittest.mock import patch

import torch

import anytrain.tts.cosyvoice.tts as cosyvoice_module
from anytrain.tts import TTSOptions, TTSVoiceMode, WaveformReference
from anytrain.tts.cosyvoice import (
    DEFAULT_MODEL_V2,
    DEFAULT_MODEL_V3,
    CosyVoiceTTS,
    CosyVoiceTTSConfig,
    CosyVoiceVersion,
)


class CosyVoiceTTSAdapterTest(unittest.TestCase):
    def test_from_pretrained_selects_version_default_and_load_options(self):
        FakeCosyVoiceModel.loaded = None
        with (
            patch("anytrain.tts.cosyvoice.tts.COSYVOICE_VLLM_ENVIRONMENT"),
            _fake_cosyvoice(),
        ):
            backend = CosyVoiceTTS.from_pretrained(
                version="v2",
                device="auto",
                load_vllm=True,
                max_concurrency=2,
                runtime_kwargs={"speed": 1.1},
            )

        self.assertEqual(backend.config.version, CosyVoiceVersion.V2)
        self.assertEqual(backend.config.model, DEFAULT_MODEL_V2)
        self.assertEqual(
            FakeCosyVoiceModel.loaded.load_kwargs,
            {"model_dir": DEFAULT_MODEL_V2, "load_vllm": True},
        )
        self.assertEqual(backend.config.runtime_kwargs, {"speed": 1.1})
        self.assertEqual(backend.config.max_concurrency, 2)
        self.assertEqual(
            FakeModelRegistry.registered,
            [
                (
                    "CosyVoice2ForCausalLM",
                    "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
                )
            ],
        )

    def test_vllm_load_uses_process_local_cuda_visibility(self):
        FakeCosyVoiceModel.loaded = None
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device", return_value=nullcontext()),
                patch("anytrain.tts.cosyvoice.tts.COSYVOICE_VLLM_ENVIRONMENT"),
                _fake_cosyvoice(),
            ):
                CosyVoiceTTS.from_pretrained(device="cuda:0", load_vllm=True)

            self.assertEqual(FakeCosyVoiceModel.loaded.cuda_visible_devices, "7")
            self.assertEqual(FakeModelRegistry.registration_visibility, ["7"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_vllm_load_validates_environment_before_import(self):
        with (
            patch("anytrain.tts.cosyvoice.tts.COSYVOICE_VLLM_ENVIRONMENT") as environment,
            self.assertRaisesRegex(RuntimeError, "separate environment"),
        ):
            environment.validate.side_effect = RuntimeError("use a separate environment")
            CosyVoiceTTS.from_pretrained(load_vllm=True)

    def test_capabilities_expose_reference_and_available_speakers(self):
        backend = CosyVoiceTTS(FakeCosyVoiceModel())

        self.assertTrue(backend.capabilities.supports(TTSVoiceMode.REFERENCE_AUDIO))
        self.assertTrue(backend.capabilities.supports(TTSVoiceMode.SPEAKER_ID))
        self.assertEqual(backend.capabilities.speaker_ids, ("中文女", "英文男"))

    def test_zero_shot_aggregates_generator_chunks(self):
        model = FakeCosyVoiceModel()
        backend = CosyVoiceTTS(model)
        reference = WaveformReference(torch.zeros(1, 160), 16_000)

        output = backend.synthesize(
            "你好",
            TTSOptions(prompt_text="参考文本", extra={"text_frontend": False}),
            reference_audio=reference,
        )

        self.assertEqual(tuple(output.waveform.shape), (1, 4))
        self.assertTrue(
            torch.equal(output.waveform, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        )
        self.assertEqual(output.sample_rate, 24_000)
        self.assertEqual(output.meta["mode"], "zero_shot")
        self.assertEqual(
            model.calls,
            [
                (
                    "zero_shot",
                    "你好",
                    "You are a helpful assistant.<|endofprompt|>参考文本",
                    (".wav", True),
                    "reference.wav",
                    "",
                    {"stream": False, "text_frontend": False},
                )
            ],
        )

    def test_batch_maps_prompt_text_and_references_per_sample(self):
        model = FakeCosyVoiceModel()
        backend = CosyVoiceTTS(model)
        references = [
            WaveformReference(torch.full((1, 16), 1.0), 16_000),
            WaveformReference(torch.full((1, 32), 2.0), 16_000),
        ]

        outputs = backend.synthesize(
            ["hello", "world"],
            TTSOptions(prompt_text=["one", "two"]),
            reference_audios=references,
        )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(
            [call[2] for call in model.calls],
            [
                "You are a helpful assistant.<|endofprompt|>one",
                "You are a helpful assistant.<|endofprompt|>two",
            ],
        )
        self.assertEqual([call[3] for call in model.calls], [(".wav", True)] * 2)

    def test_v2_prompt_text_is_unchanged(self):
        model = FakeCosyVoiceModel()
        backend = CosyVoiceTTS(
            model,
            CosyVoiceTTSConfig(
                version=CosyVoiceVersion.V2,
                model=DEFAULT_MODEL_V2,
            ),
        )
        backend.synthesize(
            "hello",
            TTSOptions(prompt_text="prompt"),
            reference_audio=WaveformReference(torch.ones(1, 16), 16_000),
        )

        self.assertEqual(model.calls[0][2], "prompt")

    def test_path_reference_is_passed_through(self):
        model = FakeCosyVoiceModel()
        backend = CosyVoiceTTS(model)
        with TemporaryDirectory() as directory:
            reference = Path(directory, "prompt.wav")
            reference.touch()
            backend.synthesize(
                "hello",
                TTSOptions(prompt_text="prompt"),
                reference_audio=reference,
            )

        self.assertEqual(model.calls[0][3], (".wav", True))
        self.assertEqual(model.calls[0][4], "prompt.wav")

    def test_vllm_batch_submits_concurrently_and_preserves_order(self):
        model = ConcurrentCosyVoiceModel()
        backend = CosyVoiceTTS(
            model,
            CosyVoiceTTSConfig(
                load_kwargs={"load_vllm": True},
                max_concurrency=2,
            ),
        )
        references = [
            WaveformReference(torch.ones(1, 16), 16_000),
            WaveformReference(torch.ones(1, 16), 16_000),
        ]

        outputs = backend.synthesize(
            ["first", "second"],
            TTSOptions(prompt_text=["one", "two"]),
            reference_audios=references,
        )

        self.assertEqual(
            [output.waveform[0, 0].item() for output in outputs],
            [1.0, 2.0],
        )
        self.assertEqual(len(set(model.thread_ids)), 2)
        self.assertEqual(model.grad_enabled, [False, False])

    def test_instruction_and_speaker_modes_use_one_public_entry(self):
        model = FakeCosyVoiceModel()
        backend = CosyVoiceTTS(model)
        reference = WaveformReference(torch.zeros(1, 16), 16_000)

        instructed = backend.synthesize(
            "你好",
            TTSOptions(instruction="用四川话说", speaker="saved"),
            reference_audio=reference,
        )
        sft = backend.synthesize("hello", TTSOptions(speaker="英文男"))

        self.assertEqual(instructed.meta["mode"], "instruct2")
        self.assertEqual(sft.meta["mode"], "sft")
        self.assertEqual([call[0] for call in model.calls], ["instruct2", "sft"])

    def test_rejects_ambiguous_or_unsupported_options(self):
        backend = CosyVoiceTTS(FakeCosyVoiceModel())
        reference = WaveformReference(torch.zeros(1, 16), 16_000)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            backend.synthesize(
                "hello",
                TTSOptions(prompt_text="prompt", instruction="style"),
                reference_audio=reference,
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            backend.synthesize(
                "hello",
                TTSOptions(prompt_text="prompt", temperature=0.8),
                reference_audio=reference,
            )


class CosyVoiceConfigTest(unittest.TestCase):
    def test_defaults_and_hash_are_stable(self):
        self.assertEqual(CosyVoiceTTSConfig().model, DEFAULT_MODEL_V3)
        first = CosyVoiceTTSConfig(
            load_kwargs={"load_vllm": True, "fp16": False},
            runtime_kwargs={"speed": 1.0, "text_frontend": True},
        )
        second = CosyVoiceTTSConfig(
            load_kwargs={"fp16": False, "load_vllm": True},
            runtime_kwargs={"text_frontend": True, "speed": 1.0},
        )
        self.assertEqual(
            CosyVoiceTTS(FakeCosyVoiceModel(), first).config_hash(),
            CosyVoiceTTS(FakeCosyVoiceModel(), second).config_hash(),
        )
        self.assertEqual(
            CosyVoiceTTS(FakeCosyVoiceModel(), first).config_hash(),
            CosyVoiceTTS(
                FakeCosyVoiceModel(),
                CosyVoiceTTSConfig(
                    load_kwargs={"load_vllm": True, "fp16": False},
                    runtime_kwargs={"speed": 1.0, "text_frontend": True},
                    max_concurrency=2,
                ),
            ).config_hash(),
        )

    def test_concurrent_inference_requires_vllm(self):
        with self.assertRaisesRegex(ValueError, "requires load_vllm=True"):
            CosyVoiceTTSConfig(max_concurrency=2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            CosyVoiceTTSConfig(max_concurrency=0)

    def test_tts_options_merge_prompt_fields(self):
        merged = TTSOptions(prompt_text="base", instruction=None).merged(
            TTSOptions(prompt_text=["left", "right"])
        )

        self.assertEqual(merged.prompt_text, ["left", "right"])
        self.assertIsNone(merged.instruction)


class FakeCosyVoiceModel:
    loaded: ClassVar[Any] = None

    def __init__(self, load_kwargs: dict[str, object] | None = None) -> None:
        self.load_kwargs = load_kwargs
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.sample_rate = 24_000
        self.calls: list[tuple[Any, ...]] = []
        type(self).loaded = self

    def list_avaliable_spks(self):
        return ["中文女", "英文男"]

    def inference_zero_shot(
        self,
        text,
        prompt_text,
        prompt_speech,
        zero_shot_spk_id="",
        **kwargs,
    ):
        self.calls.append(
            (
                "zero_shot",
                text,
                prompt_text,
                (Path(prompt_speech).suffix, Path(prompt_speech).is_file()),
                Path(prompt_speech).name,
                zero_shot_spk_id,
                kwargs,
            )
        )
        yield {"tts_speech": torch.tensor([[0.0, 1.0]])}
        yield {"tts_speech": torch.tensor([[2.0, 3.0]])}

    def inference_cross_lingual(
        self,
        text,
        prompt_speech,
        zero_shot_spk_id="",
        **kwargs,
    ):
        self.calls.append(
            (
                "cross_lingual",
                text,
                (Path(prompt_speech).suffix, Path(prompt_speech).is_file()),
                Path(prompt_speech).name,
                zero_shot_spk_id,
                kwargs,
            )
        )
        yield {"tts_speech": torch.ones(1, 2)}

    def inference_instruct2(
        self,
        text,
        instruction,
        prompt_speech,
        zero_shot_spk_id="",
        **kwargs,
    ):
        self.calls.append(
            (
                "instruct2",
                text,
                instruction,
                (Path(prompt_speech).suffix, Path(prompt_speech).is_file()),
                Path(prompt_speech).name,
                zero_shot_spk_id,
                kwargs,
            )
        )
        yield {"tts_speech": torch.ones(1, 2)}

    def inference_sft(self, text, speaker, **kwargs):
        self.calls.append(("sft", text, speaker, kwargs))
        yield {"tts_speech": torch.ones(1, 2)}


class ConcurrentCosyVoiceModel:
    def __init__(self) -> None:
        self.sample_rate = 24_000
        self.barrier = threading.Barrier(2)
        self.thread_ids: list[int] = []
        self.grad_enabled: list[bool] = []

    def list_avaliable_spks(self):
        return []

    def inference_zero_shot(
        self,
        text,
        prompt_text,
        prompt_speech,
        zero_shot_spk_id="",
        **kwargs,
    ):
        del prompt_text, prompt_speech, zero_shot_spk_id, kwargs
        self.thread_ids.append(threading.get_ident())
        self.grad_enabled.append(torch.is_grad_enabled())
        self.barrier.wait(timeout=2)
        marker = 1.0 if text == "first" else 2.0
        yield {"tts_speech": torch.full((1, 2), marker)}


class FakeModelRegistry:
    registered: ClassVar[list[tuple[str, object]]] = []
    registration_visibility: ClassVar[list[str | None]] = []

    @classmethod
    def register_model(cls, name: str, model: object) -> None:
        cls.registered.append((name, model))
        cls.registration_visibility.append(os.environ.get("CUDA_VISIBLE_DEVICES"))


class _fake_cosyvoice:
    def __init__(self) -> None:
        self.previous: dict[str, Any] = {}

    def __enter__(self):
        cosyvoice_module._register_vllm_model.cache_clear()
        FakeModelRegistry.registered = []
        FakeModelRegistry.registration_visibility = []
        root = types.ModuleType("cosyvoice")
        cli = types.ModuleType("cosyvoice.cli")
        module = types.ModuleType("cosyvoice.cli.cosyvoice")
        vllm = types.ModuleType("vllm")
        vllm.__dict__["ModelRegistry"] = FakeModelRegistry

        def auto_model(**kwargs: object) -> FakeCosyVoiceModel:
            return FakeCosyVoiceModel(kwargs)

        module.__dict__["AutoModel"] = auto_model
        self.previous = {
            name: sys.modules.get(name)
            for name in (
                "cosyvoice",
                "cosyvoice.cli",
                "cosyvoice.cli.cosyvoice",
                "vllm",
            )
        }
        sys.modules["cosyvoice"] = root
        sys.modules["cosyvoice.cli"] = cli
        sys.modules["cosyvoice.cli.cosyvoice"] = module
        sys.modules["vllm"] = vllm
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, previous in self.previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


if __name__ == "__main__":
    unittest.main()
