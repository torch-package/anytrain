from __future__ import annotations

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
        value = getattr(candidate, "frame_rate", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        hop_length = getattr(candidate, "hop_length", None)
        if isinstance(hop_length, (int, float)) and not isinstance(hop_length, bool) and hop_length > 0:
            return float(sample_rate / hop_length)
        downsampling_ratio = getattr(candidate, "downsampling_ratio", None)
        if (
            isinstance(downsampling_ratio, (int, float))
            and not isinstance(downsampling_ratio, bool)
            and downsampling_ratio > 0
        ):
            return float(sample_rate / downsampling_ratio)
    raise TypeError(
        "codec backend must expose a positive frame_rate, hop_length, or downsampling_ratio."
    )
