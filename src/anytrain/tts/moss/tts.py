"""MOSS-TTS v1.5 inference adapter.

This module exposes the stable text-to-waveform facade. Hugging Face loading
details and v1.5 generation mechanics live in the sibling private modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

import torch
from torch import nn
from typing_extensions import Unpack

from anytrain.tts import AudioReference, TTSOptions, TTSOutput, validate_text
from anytrain.tts._common import config_hash

from ._config import (
    DEFAULT_CODEC_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    MossLoadKwargs,
    MossRuntimeKwargs,
    MossTTSConfig,
)
from ._loading import load_pretrained_pair
from ._output import processor_outputs
from ._runtime import (
    audio_reference,
    audio_references,
    generate_for_texts,
    move_processor,
    processor_sample_rate,
    resolve_device,
    validate_text_batch,
    validate_v15_options,
)
from ._types import Processor


@dataclass
class MossTTS:
    model: Any
    processor: Processor
    config: MossTTSConfig = field(default_factory=MossTTSConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_loaded_from: str = "custom"

    name: str = "moss-tts-v1.5"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if isinstance(self.model, nn.Module) and "device_map" not in self.config.load_kwargs:
            self.model = self.model.to(self.device)
        move_processor(self.processor, self.device)

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_dir: str | Path | None = None,
        codec_model: str | Path = DEFAULT_CODEC_MODEL,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_kwargs: Unpack[MossLoadKwargs],
    ) -> MossTTS:
        runtime = dict(runtime_kwargs or {})
        config = MossTTSConfig(
            model=str(model),
            codec_model=str(codec_model),
            sample_rate=sample_rate,
            language=language,
            revision=revision,
            trust_remote_code=trust_remote_code,
            load_kwargs=load_kwargs,
            runtime_kwargs=runtime,
        )
        loaded, processor = load_pretrained_pair(
            model,
            cache_dir=cache_dir,
            codec_model=codec_model,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            load_kwargs=load_kwargs,
        )
        return cls(
            model=loaded,
            processor=processor,
            config=config,
            device=resolve_device(device),
            model_loaded_from="transformers",
        )

    @property
    def sample_rate(self) -> int:
        return processor_sample_rate(self.processor, self.config.sample_rate)

    def config_hash(self) -> str:
        return config_hash(self.config.hash_dict())

    @overload
    def synthesize(
        self,
        text: str,
        options: TTSOptions | None = None,
        reference_audio_path: AudioReference | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> TTSOutput: ...

    @overload
    def synthesize(
        self,
        text: Sequence[str],
        options: TTSOptions | None = None,
        reference_audio_paths: Sequence[AudioReference] | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> list[TTSOutput]: ...

    @torch.no_grad()
    def synthesize(
        self,
        text: str | Sequence[str],
        options: TTSOptions | None = None,
        reference_audio_path: AudioReference | None = None,
        reference_audio_paths: Sequence[AudioReference] | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> TTSOutput | list[TTSOutput]:
        resolved = self._options(options, runtime_kwargs)
        validate_v15_options(resolved)
        if isinstance(text, str):
            if reference_audio_paths is not None:
                raise TypeError("single text requires reference_audio_path.")
            return self._synthesize_one(text, resolved, reference_audio_path)
        if reference_audio_path is not None:
            raise TypeError("text batches require reference_audio_paths.")
        return self._synthesize_batch(text, resolved, reference_audio_paths)

    def _synthesize_one(
        self,
        text: str,
        options: TTSOptions,
        reference_audio_path: AudioReference | None,
    ) -> TTSOutput:
        decoded = self._generate_for_texts(
            [validate_text(text)],
            options,
            [None if reference_audio_path is None else audio_reference(reference_audio_path)],
        )
        return processor_outputs(
            decoded,
            expected_count=1,
            sample_rate=processor_sample_rate(
                self.processor,
                options.sample_rate or self.sample_rate,
            ),
            backend=DEFAULT_MODEL,
        )[0]

    def _synthesize_batch(
        self,
        text: Sequence[str],
        options: TTSOptions,
        reference_audio_paths: Sequence[AudioReference] | None,
    ) -> list[TTSOutput]:
        texts = validate_text_batch(text)
        decoded = self._generate_for_texts(
            texts,
            options,
            audio_references(reference_audio_paths, expected_count=len(texts)),
        )
        return processor_outputs(
            decoded,
            expected_count=len(texts),
            sample_rate=processor_sample_rate(
                self.processor,
                options.sample_rate or self.sample_rate,
            ),
            backend=DEFAULT_MODEL,
        )

    def _generate_for_texts(
        self,
        texts: Sequence[str],
        options: TTSOptions,
        references: Sequence[str | None],
    ) -> object:
        return generate_for_texts(
            model=self.model,
            processor=self.processor,
            device=self.device,
            texts=texts,
            options=options,
            references=references,
        )

    def _options(
        self,
        options: TTSOptions | None,
        runtime_kwargs: Mapping[str, object] | None = None,
    ) -> TTSOptions:
        resolved = self.config.options().merged(options)
        if runtime_kwargs:
            resolved = resolved.merged(TTSOptions(extra={**resolved.extra, **runtime_kwargs}))
        return resolved
