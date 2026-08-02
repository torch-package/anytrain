from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def validate_sample_rate(sample_rate: int, *, name: str = "sample_rate") -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError(f"{name} must be an integer.")
    if sample_rate <= 0:
        raise ValueError(f"{name} must be positive.")
    return sample_rate


def float32_waveform(audio: Any, *, type_error: str) -> Tensor:
    if isinstance(audio, Tensor):
        wave = audio.detach()
    else:
        try:
            wave = torch.as_tensor(audio)
        except Exception as exc:
            raise TypeError(type_error) from exc

    if torch.is_floating_point(wave):
        return wave.to(dtype=torch.float32)
    return wave.float()


def channel_time_waveform(
    waveform: Any,
    *,
    shape_error: str,
    empty_channel_error: str | None = None,
    empty_time_error: str | None = None,
    cpu: bool = False,
) -> Tensor:
    wave = float32_waveform(waveform, type_error=shape_error)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.ndim != 2:
        raise ValueError(shape_error)
    if empty_channel_error is not None and wave.shape[0] == 0:
        raise ValueError(empty_channel_error)
    if empty_time_error is not None and wave.shape[-1] == 0:
        raise ValueError(empty_time_error)
    if cpu:
        wave = wave.cpu()
    return wave.contiguous()


def mono_wave_batch(audio: Any, *, type_error: str) -> Tensor:
    wave = float32_waveform(audio, type_error=type_error)

    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    elif wave.ndim == 2:
        pass
    elif wave.ndim == 3:
        if wave.shape[1] == 0:
            raise ValueError("audio channel dimension must not be empty.")
        wave = wave.mean(dim=1)
    else:
        raise ValueError("audio must have shape [time], [batch, time], or [batch, channel, time].")

    if wave.shape[0] == 0 or wave.shape[-1] == 0:
        raise ValueError("audio waveform must not be empty.")
    return wave.contiguous()


def resample_waveform(
    audio: Tensor,
    sample_rate: int,
    target_sample_rate: int,
    *,
    context: str,
) -> Tensor:
    sample_rate = validate_sample_rate(sample_rate)
    target_sample_rate = validate_sample_rate(
        target_sample_rate,
        name="target_sample_rate",
    )
    if sample_rate == target_sample_rate:
        return audio

    try:
        from torchaudio.functional import resample
    except ImportError as exc:
        raise ImportError(
            f'{context} requires torchaudio. Install it with `python -m pip install "torchaudio>=2.0"`.'
        ) from exc
    return resample(audio, sample_rate, target_sample_rate)
