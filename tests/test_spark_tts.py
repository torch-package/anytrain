from __future__ import annotations

import sys
import types
from contextlib import AbstractContextManager
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from anytrain.tts import EncodedAudioReference, TTSOptions, TTSVoiceMode
from anytrain.tts.spark import SparkTTS, SparkTTSConfig


class FakeSparkRuntime:
    loaded: ClassVar[FakeSparkRuntime | None] = None

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.model_dir = model_dir
        self.device = device
        self.sample_rate = 16_000
        self.calls: list[dict[str, object]] = []
        type(self).loaded = self

    def inference(
        self,
        text: str,
        *,
        prompt_speech_path: Path,
        prompt_text: str,
        **runtime: object,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "text": text,
                "prompt_text": prompt_text,
                "reference_exists": prompt_speech_path.is_file(),
                "reference_suffix": prompt_speech_path.suffix,
                **runtime,
            }
        )
        return torch.tensor([0.0, 0.5, -0.5])


class _FakeSparkModules(AbstractContextManager[None]):
    def __init__(self) -> None:
        self.previous: dict[str, types.ModuleType | None] = {}

    def __enter__(self) -> None:
        cli = types.ModuleType("cli")
        cli.__path__ = []  # type: ignore[attr-defined]
        spark = types.ModuleType("cli.SparkTTS")
        spark.SparkTTS = FakeSparkRuntime  # type: ignore[attr-defined]
        for name, module in (("cli", cli), ("cli.SparkTTS", spark)):
            self.previous[name] = sys.modules.get(name)
            sys.modules[name] = module

    def __exit__(self, exc_type, exc, traceback) -> None:
        for name, previous in self.previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_spark_capabilities_expose_reference_audio_only(tmp_path: Path) -> None:
    with _FakeSparkModules():
        tts = SparkTTS.from_pretrained(tmp_path, device="cpu")

    assert tts.capabilities.supports(TTSVoiceMode.REFERENCE_AUDIO)
    assert not tts.capabilities.supports(TTSVoiceMode.SPEAKER_ID)
    assert tts.capabilities.speaker_ids is None


def test_spark_synthesize_materializes_reference_and_prompt_text(tmp_path: Path) -> None:
    with _FakeSparkModules():
        tts = SparkTTS.from_pretrained(
            tmp_path,
            device="cpu",
            runtime_kwargs={"temperature": 0.7, "top_p": 0.9},
        )

    output = tts.synthesize(
        "target",
        TTSOptions(prompt_text="reference", extra={"top_k": 20}),
        reference_audio=EncodedAudioReference(b"encoded-audio", ".flac"),
    )

    assert torch.equal(output.waveform, torch.tensor([[0.0, 0.5, -0.5]]))
    assert output.sample_rate == 16_000
    runtime = FakeSparkRuntime.loaded
    assert runtime is not None
    assert runtime.calls == [
        {
            "text": "target",
            "prompt_text": "reference",
            "reference_exists": True,
            "reference_suffix": ".flac",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
        }
    ]


def test_spark_batch_preserves_aligned_references_and_prompts(tmp_path: Path) -> None:
    with _FakeSparkModules():
        tts = SparkTTS.from_pretrained(tmp_path, device="cpu")

    outputs = tts.synthesize(
        ["one", "two"],
        TTSOptions(prompt_text=["ref one", "ref two"]),
        reference_audios=[
            EncodedAudioReference(b"one", ".wav"),
            EncodedAudioReference(b"two", ".opus"),
        ],
    )

    assert len(outputs) == 2
    runtime = FakeSparkRuntime.loaded
    assert runtime is not None
    assert [
        (call["text"], call["prompt_text"], call["reference_suffix"]) for call in runtime.calls
    ] == [
        ("one", "ref one", ".wav"),
        ("two", "ref two", ".opus"),
    ]


def test_spark_requires_reference_prompt_and_rejects_speaker(tmp_path: Path) -> None:
    with _FakeSparkModules():
        tts = SparkTTS.from_pretrained(tmp_path, device="cpu")

    with pytest.raises(ValueError, match="requires reference_audio"):
        tts.synthesize("target", TTSOptions(prompt_text="reference"))
    with pytest.raises(TypeError, match="requires prompt_text"):
        tts.synthesize(
            "target",
            reference_audio=EncodedAudioReference(b"audio", ".wav"),
        )
    with pytest.raises(ValueError, match="does not support speaker ids"):
        tts.synthesize(
            "target",
            TTSOptions(
                speaker="fake",
                prompt_text="reference",
            ),
            reference_audio=EncodedAudioReference(b"audio", ".wav"),
        )


def test_spark_config_hash_keeps_semantic_options() -> None:
    first = SparkTTS(
        model=object(),
        config=SparkTTSConfig(runtime_kwargs={"top_k": 20}),
    )
    repeated = SparkTTS(
        model=object(),
        config=SparkTTSConfig(runtime_kwargs={"top_k": 20}),
    )
    changed = SparkTTS(
        model=object(),
        config=SparkTTSConfig(runtime_kwargs={"top_k": 30}),
    )

    assert first.config_hash() == repeated.config_hash()
    assert first.config_hash() != changed.config_hash()


def test_spark_config_rejects_non_numeric_sampling_defaults() -> None:
    config = SparkTTSConfig(runtime_kwargs={"temperature": "hot"})

    with pytest.raises(TypeError, match="temperature must be numeric"):
        config.options()


def test_spark_top_k_separates_type_and_range_errors(tmp_path: Path) -> None:
    with _FakeSparkModules():
        tts = SparkTTS.from_pretrained(tmp_path, device="cpu")

    with pytest.raises(TypeError, match="top_k"):
        tts.synthesize(
            "hello",
            TTSOptions(prompt_text="reference", extra={"top_k": 1.5}),
            reference_audio=EncodedAudioReference(b"audio", ".wav"),
        )
    with pytest.raises(ValueError, match="top_k"):
        tts.synthesize(
            "hello",
            TTSOptions(prompt_text="reference", extra={"top_k": 0}),
            reference_audio=EncodedAudioReference(b"audio", ".wav"),
        )
