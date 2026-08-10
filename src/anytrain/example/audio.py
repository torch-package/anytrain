from __future__ import annotations

import warnings
from pathlib import Path

from torch import Tensor

from anytrain._audio import channel_time_waveform

from .resources import color_your_night_path, vctk_path


class _AudioRangeError(ValueError):
    pass


def load_example_audio(
    path: Path | str,
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    path = Path(path)
    _validate_time_range(start_seconds=start_seconds, duration=duration)

    torchcodec_error: Exception | None = None
    try:
        return _load_audio_with_torchcodec(
            path,
            start_seconds=start_seconds,
            duration=duration,
        )
    except _AudioRangeError:
        raise
    except Exception as exc:
        torchcodec_error = exc
        warnings.warn(
            "torchcodec failed to read example audio; falling back to torchaudio. "
            f"torchcodec error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        return _load_audio_with_torchaudio(
            path,
            start_seconds=start_seconds,
            duration=duration,
        )
    except _AudioRangeError:
        raise
    except Exception as torchaudio_error:
        raise RuntimeError(
            "Failed to read example audio with torchcodec and torchaudio. "
            'Install audio dependencies with `python -m pip install "torchcodec>=0.2" "torchaudio>=2.0"`. '
            f"torchcodec error: {torchcodec_error}; torchaudio error: {torchaudio_error}"
        ) from torchaudio_error


def vctk(
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    return load_example_audio(vctk_path(), start_seconds=start_seconds, duration=duration)


def color_your_night(
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    return load_example_audio(
        color_your_night_path(),
        start_seconds=start_seconds,
        duration=duration,
    )


def _load_audio_with_torchcodec(
    path: Path,
    *,
    start_seconds: float,
    duration: float | None,
) -> tuple[Tensor, int]:
    from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

    decoder = AudioDecoder(str(path))
    metadata = decoder.metadata
    sample_rate = metadata.sample_rate
    if sample_rate is None:
        raise RuntimeError("torchcodec audio metadata does not expose a sample rate.")
    total_duration = metadata.duration_seconds
    if total_duration is None:
        raise RuntimeError("torchcodec audio metadata does not expose a duration.")
    stop_seconds = _stop_seconds(
        start_seconds=start_seconds,
        duration=duration,
        total_duration=total_duration,
    )
    waveform = decoder.get_samples_played_in_range(start_seconds, stop_seconds).data
    return _normalize_waveform(waveform), sample_rate


def _load_audio_with_torchaudio(
    path: Path,
    *,
    start_seconds: float,
    duration: float | None,
) -> tuple[Tensor, int]:
    import torchaudio

    waveform, loaded_sample_rate = torchaudio.load(str(path))
    sample_rate = int(loaded_sample_rate)
    total_duration = waveform.shape[-1] / sample_rate
    _stop_seconds(
        start_seconds=start_seconds,
        duration=duration,
        total_duration=total_duration,
    )
    start_frame = int(round(start_seconds * sample_rate))
    if duration is None:
        sliced = waveform[..., start_frame:]
    else:
        num_frames = int(round(duration * sample_rate))
        stop_frame = start_frame + num_frames
        sliced = waveform[..., start_frame:stop_frame]
    return _normalize_waveform(sliced), sample_rate


def _validate_time_range(*, start_seconds: float, duration: float | None) -> None:
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative.")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive when provided.")


def _stop_seconds(
    *,
    start_seconds: float,
    duration: float | None,
    total_duration: float,
) -> float | None:
    if start_seconds >= total_duration:
        raise _AudioRangeError(
            f"start_seconds ({start_seconds}s) must be smaller than audio duration "
            f"({total_duration}s)."
        )
    if duration is None:
        return None
    stop_seconds = start_seconds + duration
    if stop_seconds > total_duration:
        raise _AudioRangeError(
            f"start_seconds + duration ({stop_seconds}s) cannot exceed audio duration "
            f"({total_duration}s)."
        )
    return stop_seconds


def _normalize_waveform(waveform: Tensor) -> Tensor:
    return channel_time_waveform(
        waveform,
        shape_error="example audio waveform must have shape (channels, time).",
        cpu=True,
    )
