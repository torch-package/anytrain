from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from anytrain.tts import TTSOutput, waveform_duration


def processor_outputs(
    value: object,
    *,
    expected_count: int,
    sample_rate: int,
    backend: str,
) -> list[TTSOutput]:
    messages = processor_messages(value, expected_count=expected_count)
    return [
        processor_message(message, sample_rate=sample_rate, backend=backend)
        for message in messages
    ]


def processor_messages(value: object, *, expected_count: int) -> list[object]:
    if isinstance(value, TTSOutput):
        if expected_count == 1:
            return [value]
        raise ValueError("processor decode returned one output for a text batch.")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        if expected_count == 1:
            return [value]
        raise TypeError("processor decode output for a text batch must be a sequence.")
    if len(value) != expected_count:
        raise ValueError(
            "processor decode output length must match text batch length: "
            f"got {len(value)}, expected {expected_count}."
        )
    return list(value)


def processor_message(message: object, *, sample_rate: int, backend: str) -> TTSOutput:
    if isinstance(message, TTSOutput):
        return message
    audio = first_processor_audio(message)
    return output(
        {
            "waveform": audio,
            "sample_rate": sample_rate,
            "meta": {"backend": backend},
        },
        sample_rate=sample_rate,
    )


def first_processor_audio(message: object) -> Tensor:
    audio_codes_list = getattr(message, "audio_codes_list", None)
    if audio_codes_list is not None:
        if len(audio_codes_list) == 0:
            raise ValueError("processor message returned no audio.")
        value = audio_codes_list[0]
        if isinstance(value, Tensor):
            return value
    if isinstance(message, Mapping):
        for key in ("audio", "waveform"):
            value = message.get(key)
            if isinstance(value, Tensor):
                return value
        audio_codes_list = message.get("audio_codes_list")
        if isinstance(audio_codes_list, Sequence) and len(audio_codes_list) > 0:
            value = audio_codes_list[0]
            if isinstance(value, Tensor):
                return value
    raise TypeError("processor decode output must include audio.")


def output(value: object, *, sample_rate: int) -> TTSOutput:
    if isinstance(value, TTSOutput):
        return value
    if isinstance(value, Tensor):
        wave = waveform(value)
        return TTSOutput(
            waveform=wave,
            sample_rate=sample_rate,
            duration=waveform_duration(wave, sample_rate),
        )
    if isinstance(value, Mapping):
        wave = waveform(waveform_value(value))
        output_sample_rate = int(value.get("sample_rate", sample_rate))
        duration = float(value.get("duration", waveform_duration(wave, output_sample_rate)))
        meta = value.get("meta", {})
        if not isinstance(meta, Mapping):
            raise TypeError("output meta must be a mapping when set.")
        return TTSOutput(
            waveform=wave,
            sample_rate=output_sample_rate,
            duration=duration,
            meta=meta,
        )
    raise TypeError("processor decode output must be TTSOutput, a Tensor, or a mapping.")


def waveform(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("waveform must be a torch.Tensor.")
    wave = value.detach()
    wave = wave.float() if not torch.is_floating_point(wave) else wave.to(dtype=torch.float32)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.ndim != 2:
        raise ValueError("waveform must have shape [time] or [channels, time].")
    return wave.contiguous()


def waveform_value(value: Mapping[object, object]) -> Tensor:
    for key in ("waveform", "audio"):
        item = value.get(key)
        if isinstance(item, Tensor):
            return item
    raise KeyError("processor decode mapping must include `waveform` or `audio`.")
