from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from anytrain._audio import (
    channel_time_waveform,
    mono_wave_batch,
    resample_waveform,
    validate_sample_rate,
)


def load_wave_batch(audio: Any, sample_rate: int) -> tuple[Tensor, int]:
    if isinstance(audio, (str, Path)):
        return _load_audio_file(Path(audio))

    sample_rate = validate_sample_rate(sample_rate)
    return coerce_wave_batch(audio), sample_rate


def coerce_wave_batch(audio: Any) -> Tensor:
    return mono_wave_batch(
        audio,
        type_error="audio must be a Tensor-like waveform or an audio file path.",
    )


def resample_wave(wave: Tensor, sample_rate: int, target_sample_rate: int) -> Tensor:
    return resample_waveform(
        wave,
        sample_rate,
        target_sample_rate,
        context="Resampling speech evaluator audio",
    )


def _load_audio_file(path: Path) -> tuple[Tensor, int]:
    torchaudio_error: Exception | None = None
    try:
        return _load_audio_file_with_torchaudio(path)
    except Exception as exc:
        torchaudio_error = exc

    try:
        return _load_audio_file_with_soundfile(path)
    except Exception as soundfile_error:
        raise RuntimeError(
            "Failed to read speech evaluator audio file with torchaudio or soundfile. "
            'Install audio file dependencies with `python -m pip install "torchaudio>=2.0" "soundfile>=0.12"`. '
            f"torchaudio error: {torchaudio_error}; soundfile error: {soundfile_error}"
        ) from soundfile_error


def _load_audio_file_with_torchaudio(path: Path) -> tuple[Tensor, int]:
    import torchaudio

    wave, sample_rate = torchaudio.load(str(path))
    sample_rate = validate_sample_rate(int(sample_rate))
    return (
        _mono_channel_time_wave(
            wave,
            shape_error="torchaudio.load(...) must return waveform shape [channel, time].",
        ),
        sample_rate,
    )


def _load_audio_file_with_soundfile(path: Path) -> tuple[Tensor, int]:
    import soundfile  # pyright: ignore[reportMissingImports]

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    sample_rate = validate_sample_rate(int(sample_rate))
    wave = torch.from_numpy(data).transpose(0, 1)
    return (
        _mono_channel_time_wave(
            wave,
            shape_error="soundfile.read(...) must return waveform shape [channel, time].",
        ),
        sample_rate,
    )


def _mono_channel_time_wave(wave: Tensor, *, shape_error: str) -> Tensor:
    wave = channel_time_waveform(
        wave,
        shape_error=shape_error,
        empty_channel_error="audio channel dimension must not be empty.",
        empty_time_error="audio waveform must not be empty.",
    )
    return wave.mean(dim=0, keepdim=True).contiguous()
