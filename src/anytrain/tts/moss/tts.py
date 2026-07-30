"""MOSS-TTS v1.5 inference adapter.

This module loads the Hugging Face remote-code model and processor, then exposes
the single stable text-to-waveform path used by MOSS-TTS v1.5.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast, overload

import torch
from torch import Tensor, nn
from typing_extensions import Unpack

from anytrain._compat import strict_zip
from anytrain.tts import (
    AudioReference,
    TTSOptions,
    TTSOutput,
    validate_text,
)
from anytrain.tts._common import config_hash, hashable_mapping, positive_int, validate_mapping

from ._deps import load_transformers_auto_model_class, load_transformers_auto_processor_class
from ._output import processor_outputs

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEFAULT_CODEC_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DEFAULT_SAMPLE_RATE = 24000


class MossLoadKwargs(TypedDict, total=False):
    attn_implementation: object
    device_map: object
    dtype: object
    low_cpu_mem_usage: bool
    torch_dtype: object
    use_safetensors: bool


class MossRuntimeKwargs(TypedDict, total=False):
    do_sample: bool
    max_new_tokens: int
    temperature: float
    tokens: object
    top_p: float


class _AttentionModel(Protocol):
    def _set_attention_implementation(self, value: str) -> None: ...


class _Processor(Protocol):
    model_config: Any

    def __call__(self, value: object, **kwargs: object) -> object: ...

    def build_user_message(self, **kwargs: object) -> object: ...

    def decode(self, value: object) -> object: ...


@dataclass(frozen=True)
class MossTTSConfig:
    model: str = DEFAULT_MODEL
    codec_model: str = DEFAULT_CODEC_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    language: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty.")
        if not self.codec_model:
            raise ValueError("codec_model must be non-empty.")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        validate_mapping(self.load_kwargs, "load_kwargs")
        validate_mapping(self.runtime_kwargs, "runtime_kwargs")

    def options(self) -> TTSOptions:
        return TTSOptions(
            language=self.language,
            sample_rate=self.sample_rate,
            extra=self.runtime_kwargs,
        )

    def hash_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "codec_model": self.codec_model,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "load_kwargs": hashable_mapping(self.load_kwargs),
            "runtime_kwargs": hashable_mapping(self.runtime_kwargs),
        }


@dataclass
class MossTTS:
    model: Any
    processor: _Processor
    config: MossTTSConfig = field(default_factory=MossTTSConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_loaded_from: str = "custom"

    name: str = "moss-tts-v1.5"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if isinstance(self.model, nn.Module) and "device_map" not in self.config.load_kwargs:
            self.model = self.model.to(self.device)
        _move_processor(self.processor, self.device)

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
        pretrained_kwargs = _pretrained_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            load_kwargs=load_kwargs,
        )
        try:
            auto_model_cls = load_transformers_auto_model_class()
            auto_processor_cls = load_transformers_auto_processor_class()
        except ImportError as exc:
            raise ImportError(
                "`anytrain.tts.moss` requires `transformers` to load remote-code "
                "MOSS-TTS v1.5 checkpoints and processors. Install Moss TTS "
                'dependencies with `python -m pip install "torchaudio>=2.0" transformers`.'
            ) from exc
        try:
            loaded = auto_model_cls.from_pretrained(str(model), **pretrained_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Hugging Face MOSS-TTS v1.5 checkpoint {model!s}"
            ) from exc
        try:
            processor = auto_processor_cls.from_pretrained(
                _pretrained_source(
                    model,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                ),
                **_processor_pretrained_kwargs(
                    cache_dir=cache_dir,
                    codec_model=codec_model,
                    local_files_only=local_files_only,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Hugging Face MOSS-TTS v1.5 processor {model!s}"
            ) from exc
        _set_attention_implementation(loaded, load_kwargs.get("attn_implementation", "sdpa"))
        resolved_device = _device(device)
        return cls(
            model=loaded,
            processor=cast(_Processor, processor),
            config=config,
            device=resolved_device,
            model_loaded_from="transformers",
        )

    @property
    def sample_rate(self) -> int:
        return _processor_sample_rate(self.processor, self.config.sample_rate)

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
        _validate_v15_options(resolved)
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
        text = validate_text(text)
        decoded = self._generate_for_texts(
            [text],
            options,
            [None if reference_audio_path is None else _audio_reference(reference_audio_path)],
        )
        return processor_outputs(
            decoded,
            expected_count=1,
            sample_rate=_processor_sample_rate(
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
        texts = _validate_text_batch(text)
        decoded = self._generate_for_texts(
            texts,
            options,
            _audio_references(reference_audio_paths, expected_count=len(texts)),
        )
        return processor_outputs(
            decoded,
            expected_count=len(texts),
            sample_rate=_processor_sample_rate(
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
        conversations = [
            [
                self.processor.build_user_message(
                    **_processor_message_kwargs(text, options, reference)
                )
            ]
            for text, reference in strict_zip(texts, references)
        ]
        batch = self.processor(conversations, mode="generation")
        inputs = _processor_batch_inputs(batch, self.device)
        with _seeded_rng(options.seed, self.device):
            raw = self.model.generate(**inputs, **_processor_generation_kwargs(options))
        return self.processor.decode(raw)

    def _options(
        self,
        options: TTSOptions | None,
        runtime_kwargs: Mapping[str, object] | None = None,
    ) -> TTSOptions:
        resolved = self.config.options().merged(options)
        if runtime_kwargs:
            resolved = resolved.merged(TTSOptions(extra={**resolved.extra, **runtime_kwargs}))
        return resolved


def _pretrained_kwargs(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
    load_kwargs: Mapping[str, object],
) -> dict[str, object]:
    kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    kwargs.update(load_kwargs)
    return kwargs


def _processor_pretrained_kwargs(
    *,
    cache_dir: str | Path | None,
    codec_model: object,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    kwargs["codec_path"] = _pretrained_source(
        str(codec_model),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=None,
    )
    return kwargs


def _pretrained_source(
    source: str | Path,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
) -> str:
    raw = str(source)
    if Path(raw).expanduser().exists() or not local_files_only:
        return raw
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Resolving a cached MOSS-TTS repository requires `huggingface-hub`. "
            'Install it with `python -m pip install "huggingface-hub>=0.23"`.'
        ) from exc
    return snapshot_download(
        repo_id=raw,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=True,
        revision=revision,
    )


def _processor_message_kwargs(
    text: str,
    options: TTSOptions,
    reference_audio_path: str | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"text": text}
    if options.language is not None:
        kwargs["language"] = options.language
    if reference_audio_path is not None:
        kwargs["reference"] = [reference_audio_path]
    tokens = options.extra.get("tokens")
    if tokens is not None:
        kwargs["tokens"] = tokens
    return kwargs


def _audio_reference(value: AudioReference) -> str:
    if not isinstance(value, (str, PathLike)):
        raise TypeError("reference audio path must be a string or path-like object.")
    return str(value)


def _audio_references(
    values: Sequence[AudioReference] | None,
    *,
    expected_count: int,
) -> list[str | None]:
    if values is None:
        return [None] * expected_count
    if isinstance(values, (str, bytes, PathLike)):
        raise TypeError("reference_audio_paths must be a sequence of paths.")
    if len(values) != expected_count:
        raise ValueError(
            "reference_audio_paths length must match text batch length: "
            f"got {len(values)}, expected {expected_count}."
        )
    return [_audio_reference(value) for value in values]


def _validate_text_batch(text: Sequence[str]) -> list[str]:
    if not isinstance(text, Sequence):
        raise TypeError("text must be a string or a sequence of strings.")
    if len(text) == 0:
        raise ValueError("text batch must not be empty.")
    values: list[str] = []
    for index, item in enumerate(text):
        if not isinstance(item, str):
            raise TypeError(f"text[{index}] must be a string.")
        try:
            values.append(validate_text(item))
        except ValueError as exc:
            raise ValueError(f"text[{index}] must not be empty.") from exc
    return values


def _processor_generation_kwargs(options: TTSOptions) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    do_sample = options.extra.get("do_sample")
    if do_sample is not None:
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean when set.")
        kwargs["do_sample"] = do_sample

    max_new_tokens = _option_or_extra(options.max_new_tokens, options.extra, "max_new_tokens")
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = positive_int(max_new_tokens, "max_new_tokens")

    temperature = _option_or_extra(options.temperature, options.extra, "temperature")
    if temperature is not None:
        kwargs["temperature"] = _positive_float(temperature, "temperature")
        kwargs.setdefault("do_sample", True)

    top_p = _option_or_extra(options.top_p, options.extra, "top_p")
    if top_p is not None:
        kwargs["top_p"] = _top_p(top_p)
        kwargs.setdefault("do_sample", True)
    return kwargs


def _option_or_extra(
    option_value: object | None,
    extra: Mapping[str, object],
    name: str,
) -> object | None:
    return option_value if option_value is not None else extra.get(name)


def _processor_batch_inputs(batch: object, device: torch.device) -> dict[str, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("processor output must be a mapping.")
    inputs: dict[str, Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            inputs[str(key)] = value.to(device)
    if "input_ids" not in inputs:
        raise KeyError("processor output must include `input_ids`.")
    return inputs


def _processor_sample_rate(processor: _Processor, fallback: int) -> int:
    model_config = getattr(processor, "model_config", None)
    value = getattr(model_config, "sampling_rate", None)
    return fallback if value is None else int(value)


def _set_attention_implementation(model: object, value: object) -> None:
    attention = _attention_model(model)
    if attention is not None:
        attention._set_attention_implementation(str(value))


@contextmanager
def _seeded_rng(seed: int | None, device: torch.device) -> Iterator[None]:
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = _cuda_rng_states()
    try:
        torch.manual_seed(seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        _restore_cuda_rng_states(cuda_states)


def _cuda_rng_states() -> list[Tensor]:
    if not torch.cuda.is_available():
        return []
    return torch.cuda.get_rng_state_all()


def _restore_cuda_rng_states(states: Sequence[Tensor]) -> None:
    if states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(states))


def _attention_model(model: object) -> _AttentionModel | None:
    setter = getattr(model, "_set_attention_implementation", None)
    return cast(_AttentionModel, model) if callable(setter) else None


def _move_processor(processor: _Processor, device: torch.device) -> None:
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    to = getattr(audio_tokenizer, "to", None)
    if callable(to):
        processor.audio_tokenizer = to(device)


def _validate_v15_options(options: TTSOptions) -> None:
    if options.speaker is not None:
        raise ValueError("MOSS-TTS v1.5 does not support speaker ids; use prompt audio.")
    _validate_runtime_kwargs(options.extra, "runtime options")


def _validate_runtime_kwargs(value: Mapping[str, object], name: str) -> None:
    validate_mapping(value, name)
    unsupported = set(value) - set(MossRuntimeKwargs.__annotations__)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"unsupported MOSS-TTS v1.5 runtime options: {names}.")


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    resolved = float(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _top_p(value: object) -> float:
    resolved = _positive_float(value, "top_p")
    if resolved > 1:
        raise ValueError("top_p must be in (0, 1].")
    return resolved


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
