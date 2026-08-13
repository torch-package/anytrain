"""Lazy vLLM-Omni construction and audio-output normalization.

Backend modules own prompt construction because Qwen3-TTS and MOSS-TTS use
different processor contracts. This module only owns the common optional
runtime boundary and the stable ``TTSOutput`` conversion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Protocol, cast

import torch
from torch import Tensor

from ._common import positive_float, positive_int, probability
from .environments import VLLM_OMNI_ENVIRONMENT
from .protocol import TTSOptions, TTSOutput, waveform_duration


class OmniEngine(Protocol):
    @property
    def default_sampling_params_list(self) -> Sequence[object]: ...

    def generate(
        self,
        prompts: object,
        sampling_params_list: object = None,
        *,
        use_tqdm: bool = False,
    ) -> Iterable[object]: ...


class StageSamplingParams(Protocol):
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    seed: int


def load_omni(model: str, kwargs: Mapping[str, object]) -> OmniEngine:
    """Construct the optional offline vLLM-Omni engine."""

    VLLM_OMNI_ENVIRONMENT.validate()
    try:
        from vllm_omni import Omni  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "vLLM-Omni TTS requires `vllm-omni`; install a version that "
            "supports the selected TTS checkpoint."
        ) from exc
    try:
        return cast(OmniEngine, Omni(model=model, **kwargs))
    except Exception as exc:
        raise RuntimeError(f"failed to load vLLM-Omni TTS checkpoint {model!r}") from exc


def generate_audio(
    engine: OmniEngine,
    prompts: Sequence[Mapping[str, object]],
    *,
    options: TTSOptions,
    sample_rate: int,
    backend: str,
    meta: Sequence[Mapping[str, object]] | None = None,
    max_concurrency: int | None = None,
    prompt_extra_keys: frozenset[str] = frozenset(),
) -> list[TTSOutput]:
    """Generate ordered audio outputs from vLLM-Omni prompts."""

    if not prompts:
        raise ValueError("vLLM-Omni prompts must not be empty.")
    positive_int(sample_rate, "sample_rate")
    if max_concurrency is not None:
        positive_int(max_concurrency, "max_concurrency")
    if meta is not None and len(meta) != len(prompts):
        raise ValueError("meta length must match prompt count.")
    batch_size = len(prompts) if max_concurrency is None else max_concurrency
    outputs: list[TTSOutput] = []
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        outputs.extend(
            _generate_audio_batch(
                engine,
                prompts[start:end],
                options=options,
                sample_rate=sample_rate,
                backend=backend,
                meta=None if meta is None else meta[start:end],
                prompt_extra_keys=prompt_extra_keys,
            )
        )
    return outputs


def _generate_audio_batch(
    engine: OmniEngine,
    prompts: Sequence[Mapping[str, object]],
    *,
    options: TTSOptions,
    sample_rate: int,
    backend: str,
    meta: Sequence[Mapping[str, object]] | None,
    prompt_extra_keys: frozenset[str],
) -> list[TTSOutput]:
    sampling = _sampling_params(engine, options, prompt_extra_keys)
    raw_outputs = engine.generate(
        list(prompts),
        sampling,
        use_tqdm=False,
    )
    chunks: dict[int, list[Tensor]] = {}
    sample_rates: dict[int, int] = {}
    fallback_index = 0
    for raw in raw_outputs:
        payload = _multimodal_output(raw)
        if payload is None:
            continue
        audio = payload.get("audio")
        if audio is None:
            audio = payload.get("model_outputs")
        values = _audio_chunks(audio)
        if not values:
            continue
        index = _request_index(raw)
        if index is None:
            index = fallback_index
            fallback_index += 1
        if index < 0 or index >= len(prompts):
            raise RuntimeError(f"vLLM-Omni returned invalid request index {index}.")
        chunks.setdefault(index, []).extend(values)
        output_sample_rate = _sample_rate(payload, sample_rate)
        previous = sample_rates.setdefault(index, output_sample_rate)
        if previous != output_sample_rate:
            raise RuntimeError("vLLM-Omni changed sample rate within one request.")

    missing = [index for index in range(len(prompts)) if index not in chunks]
    if missing:
        raise RuntimeError(
            "vLLM-Omni returned no audio for request indices: "
            + ", ".join(str(index) for index in missing)
            + "."
        )
    outputs: list[TTSOutput] = []
    for index in range(len(prompts)):
        waveform = _waveform(chunks[index])
        output_sample_rate = sample_rates[index]
        output_meta: dict[str, object] = {
            "backend": backend,
            "transformer_engine": "vllm_omni",
        }
        if meta is not None:
            output_meta.update(meta[index])
        outputs.append(
            TTSOutput(
                waveform=waveform,
                sample_rate=output_sample_rate,
                duration=waveform_duration(waveform, output_sample_rate),
                meta=output_meta,
            )
        )
    return outputs


def _sampling_params(
    engine: OmniEngine,
    options: TTSOptions,
    prompt_extra_keys: frozenset[str],
) -> object:
    sampling_keys = frozenset(
        (
            "do_sample",
            "max_new_tokens",
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
        )
    )
    unknown = set(options.extra) - sampling_keys - prompt_extra_keys
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unsupported vLLM-Omni TTS runtime options: {names}.")
    defaults = getattr(engine, "default_sampling_params_list", None)
    if defaults is None:
        if any(
            value is not None
            for value in (
                options.max_new_tokens,
                options.temperature,
                options.top_p,
                options.seed,
            )
        ) or any(name in options.extra for name in sampling_keys):
            raise RuntimeError(
                "vLLM-Omni engine does not expose default_sampling_params_list; "
                "request sampling controls cannot be applied."
            )
        return None
    params = deepcopy(list(defaults))
    if not params:
        raise RuntimeError("vLLM-Omni returned no stage sampling parameters.")
    talker = cast(StageSamplingParams, params[0])
    max_new_tokens = options.max_new_tokens or options.extra.get("max_new_tokens")
    if max_new_tokens is not None:
        talker.max_tokens = positive_int(max_new_tokens, "max_new_tokens")
    temperature = (
        options.temperature
        if options.temperature is not None
        else options.extra.get("temperature")
    )
    if temperature is not None:
        talker.temperature = positive_float(temperature, "temperature")
    top_p = options.top_p if options.top_p is not None else options.extra.get("top_p")
    if top_p is not None:
        talker.top_p = probability(top_p, "top_p")
    if options.seed is not None:
        talker.seed = options.seed
    top_k = options.extra.get("top_k")
    if top_k is not None:
        talker.top_k = positive_int(top_k, "top_k")
    repetition_penalty = options.extra.get("repetition_penalty")
    if repetition_penalty is not None:
        talker.repetition_penalty = positive_float(
            repetition_penalty,
            "repetition_penalty",
        )
    do_sample = options.extra.get("do_sample")
    if do_sample is not None:
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean when set.")
        if not do_sample:
            talker.temperature = 0.0
    return params


def _multimodal_output(value: object) -> Mapping[str, object] | None:
    payload = getattr(value, "multimodal_output", None)
    if isinstance(payload, Mapping):
        return cast(Mapping[str, object], payload)
    request_output = getattr(value, "request_output", None)
    outputs = getattr(request_output, "outputs", None)
    if isinstance(outputs, Sequence) and outputs:
        payload = getattr(outputs[0], "multimodal_output", None)
        if isinstance(payload, Mapping):
            return cast(Mapping[str, object], payload)
    return None


def _audio_chunks(value: object) -> list[Tensor]:
    if isinstance(value, Tensor):
        tensor = cast(Tensor, value)
        return [tensor] if tensor.numel() > 0 else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        chunks: list[Tensor] = []
        for item in value:
            if not isinstance(item, Tensor):
                raise TypeError("vLLM-Omni audio chunks must be torch.Tensor values.")
            if item.numel() > 0:
                chunks.append(item)
        return chunks
    if value is None:
        return []
    raise TypeError("vLLM-Omni audio output must be a Tensor or sequence of Tensors.")


def _request_index(value: object) -> int | None:
    request_id = getattr(value, "request_id", None)
    if not isinstance(request_id, str):
        request_output = getattr(value, "request_output", None)
        request_id = getattr(request_output, "request_id", None)
    if not isinstance(request_id, str):
        return None
    prefix, separator, _ = request_id.partition("_")
    if not separator or not prefix.isdigit():
        return None
    return int(prefix)


def _sample_rate(payload: Mapping[str, object], fallback: int) -> int:
    value: object = fallback
    for key in ("sr", "sample_rate", "audio_sample_rate"):
        if key in payload:
            value = payload[key]
            break
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("vLLM-Omni returned an empty sample-rate sequence.")
        value = value[-1]
    if isinstance(value, Tensor):
        tensor = cast(Tensor, value)
        if tensor.numel() != 1:
            raise ValueError("vLLM-Omni sample rate tensor must be scalar.")
        value = tensor.item()
    return positive_int(value, "vLLM-Omni sample rate")


def _waveform(chunks: Sequence[Tensor]) -> Tensor:
    normalized: list[Tensor] = []
    channels: int | None = None
    for chunk in chunks:
        value = chunk.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError("vLLM-Omni waveform chunks must have shape [time] or [channels, time].")
        if channels is None:
            channels = value.shape[0]
        elif value.shape[0] != channels:
            raise ValueError("vLLM-Omni waveform chunks changed channel count.")
        normalized.append(value)
    waveform = torch.cat(normalized, dim=-1).contiguous()
    if not torch.isfinite(waveform).all():
        raise RuntimeError("vLLM-Omni returned non-finite audio.")
    return waveform
