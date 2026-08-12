from __future__ import annotations

import math
import warnings
from numbers import Real
from pathlib import Path

from torch import Tensor

from anytrain._audio import channel_time_waveform, validate_sample_rate

from .resources import color_your_night_path, vctk_path


class _AudioRangeError(ValueError):
    pass


def load_example_audio(
    path: Path | str,
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("path must not be empty.")
    path = Path(path)
    start_seconds, duration = _validate_time_range(
        start_seconds=start_seconds,
        duration=duration,
    )

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
    raw_sample_rate = metadata.sample_rate
    if raw_sample_rate is None:
        raise RuntimeError("torchcodec audio metadata does not expose a sample rate.")
    sample_rate = validate_sample_rate(raw_sample_rate)
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
    sample_rate = validate_sample_rate(loaded_sample_rate)
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


def _validate_time_range(
    *,
    start_seconds: float,
    duration: float | None,
) -> tuple[float, float | None]:
    start_seconds = _finite_real(start_seconds, "start_seconds")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative.")
    if duration is not None:
        duration = _finite_real(duration, "duration")
        if duration <= 0:
            raise ValueError("duration must be positive when provided.")
    return start_seconds, duration


def _stop_seconds(
    *,
    start_seconds: float,
    duration: float | None,
    total_duration: float,
) -> float | None:
    total_duration = _finite_real(total_duration, "audio duration")
    if total_duration <= 0:
        raise ValueError("audio duration must be positive.")
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
        empty_channel_error="example audio waveform must contain at least one channel.",
        empty_time_error="example audio waveform must not be empty.",
        cpu=True,
    )


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a host real number, not bool or Tensor.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved
