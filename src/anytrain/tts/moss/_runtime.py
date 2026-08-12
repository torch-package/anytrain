from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from torch import Tensor

from anytrain._compat import strict_zip
from anytrain.tts import (
    AudioReference,
    EncodedAudioReference,
    TTSOptions,
    WaveformReference,
    validate_text,
)
from anytrain.tts._common import positive_float, positive_int, probability, validate_mapping

from ._config import MossRuntimeKwargs
from ._types import AudioTokenizerProcessor, DeviceMovable, Processor


def generate_for_texts(
    *,
    model: Any,
    processor: Processor,
    device: torch.device,
    texts: Sequence[str],
    options: TTSOptions,
    references: Sequence[AudioReference | None],
) -> object:
    with materialized_audio_references(references) as paths:
        conversations = [
            [processor.build_user_message(**processor_message_kwargs(text, options, reference))]
            for text, reference in strict_zip(texts, paths)
        ]
        batch = processor(conversations, mode="generation")
        inputs = processor_batch_inputs(batch, device)
        with seeded_rng(options.seed, device):
            raw = model.generate(**inputs, **processor_generation_kwargs(options))
        return processor.decode(raw)


def processor_message_kwargs(
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


def audio_references(
    values: Sequence[AudioReference] | None,
    *,
    expected_count: int,
) -> list[AudioReference | None]:
    if values is None:
        return [None] * expected_count
    if isinstance(values, (str, bytes, PathLike)):
        raise TypeError("reference_audios must be a sequence of audio references.")
    if len(values) != expected_count:
        raise ValueError(
            "reference_audios length must match text batch length: "
            f"got {len(values)}, expected {expected_count}."
        )
    return list(values)


@contextmanager
def materialized_audio_references(
    values: Sequence[AudioReference | None],
) -> Iterator[list[str | None]]:
    """Expose references as paths for the complete MOSS generation lifecycle."""

    temporary: TemporaryDirectory[str] | None = None
    paths: list[str | None] = []
    try:
        for index, value in enumerate(values):
            if value is None:
                paths.append(None)
                continue
            if isinstance(value, (str, PathLike)):
                paths.append(str(value))
                continue
            if temporary is None:
                temporary = TemporaryDirectory(prefix="anytrain-moss-reference-")
            root = Path(temporary.name)
            if isinstance(value, WaveformReference):
                path = root / f"reference-{index}.wav"
                _save_waveform_reference(value, path)
            elif isinstance(value, EncodedAudioReference):
                path = root / f"reference-{index}{value.suffix}"
                path.write_bytes(value.data)
            else:
                raise TypeError(
                    "reference audio must be a path, WaveformReference, "
                    "or EncodedAudioReference."
                )
            paths.append(str(path))
        yield paths
    finally:
        if temporary is not None:
            temporary.cleanup()


def _save_waveform_reference(value: WaveformReference, path: Path) -> None:
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError(
            "WaveformReference requires torchaudio; install it with "
            "`python -m pip install \"torchaudio>=2.0\"`."
        ) from error
    torchaudio.save(
        str(path),
        value.waveform.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0),
        value.sample_rate,
    )


def validate_text_batch(text: Sequence[str]) -> list[str]:
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


def processor_generation_kwargs(options: TTSOptions) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    do_sample = options.extra.get("do_sample")
    if do_sample is not None:
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean when set.")
        kwargs["do_sample"] = do_sample

    max_new_tokens = option_or_extra(options.max_new_tokens, options.extra, "max_new_tokens")
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = positive_int(max_new_tokens, "max_new_tokens")

    temperature = option_or_extra(options.temperature, options.extra, "temperature")
    if temperature is not None:
        kwargs["temperature"] = positive_float(temperature, "temperature")
        kwargs.setdefault("do_sample", True)

    top_p = option_or_extra(options.top_p, options.extra, "top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p_value(top_p)
        kwargs.setdefault("do_sample", True)
    return kwargs


def option_or_extra(
    option_value: object | None,
    extra: Mapping[str, object],
    name: str,
) -> object | None:
    return option_value if option_value is not None else extra.get(name)


def processor_batch_inputs(batch: object, device: torch.device) -> dict[str, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("processor output must be a mapping.")
    inputs: dict[str, Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            inputs[str(key)] = value.to(device)
    if "input_ids" not in inputs:
        raise KeyError("processor output must include `input_ids`.")
    return inputs


def processor_sample_rate(processor: Processor, fallback: int) -> int:
    positive_int(fallback, "fallback sample rate")
    model_config = getattr(processor, "model_config", None)
    value = getattr(model_config, "sampling_rate", None)
    return fallback if value is None else positive_int(value, "processor sample rate")


@contextmanager
def seeded_rng(seed: int | None, device: torch.device) -> Iterator[None]:
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = cuda_rng_states()
    try:
        torch.manual_seed(seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        restore_cuda_rng_states(cuda_states)


def cuda_rng_states() -> list[Tensor]:
    if not torch.cuda.is_available():
        return []
    return torch.cuda.get_rng_state_all()


def restore_cuda_rng_states(states: Sequence[Tensor]) -> None:
    if states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(states))


def move_processor(processor: Processor, device: torch.device) -> None:
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    to = getattr(audio_tokenizer, "to", None)
    if callable(to):
        cast(AudioTokenizerProcessor, processor).audio_tokenizer = cast(DeviceMovable, to(device))


def validate_v15_options(options: TTSOptions) -> None:
    if options.speaker is not None:
        raise ValueError("MOSS-TTS v1.5 does not support speaker ids; use prompt audio.")
    validate_runtime_kwargs(options.extra, "runtime options")


def validate_runtime_kwargs(value: Mapping[str, object], name: str) -> None:
    validate_mapping(value, name)
    unsupported = set(value) - set(MossRuntimeKwargs.__annotations__)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"unsupported MOSS-TTS v1.5 runtime options: {names}.")


def top_p_value(value: object) -> float:
    return probability(value, "top_p")


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
