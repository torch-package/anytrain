from __future__ import annotations

from numbers import Real

from torch import Tensor

from anytrain._audio import resample_waveform


def resample(audio: Tensor, sample_rate: int, target_sample_rate: int) -> Tensor:
    return resample_waveform(
        audio,
        sample_rate,
        target_sample_rate,
        context="Codec input resampling",
    )


def model_frame_rate(model: object, sample_rate: int) -> float:
    """Read a backend's primary frame rate from common upstream metadata."""
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "encoder", None),
        getattr(model, "feature_extractor", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        frame_rate = _positive_real(getattr(candidate, "frame_rate", None))
        if frame_rate is not None:
            return frame_rate
        hop_length = _positive_real(getattr(candidate, "hop_length", None))
        if hop_length is not None:
            return float(sample_rate / hop_length)
        downsampling_ratio = _positive_real(getattr(candidate, "downsampling_ratio", None))
        if downsampling_ratio is not None:
            return float(sample_rate / downsampling_ratio)
    raise TypeError(
        "codec backend must expose a positive frame_rate, hop_length, or downsampling_ratio."
    )


def _positive_real(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    resolved = float(value)
    return resolved if resolved > 0 else None
