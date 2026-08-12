from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .planning import AudioPlanningEstimator
from .spec import AudioBackendIdentity

PLANNING_IDENTITY_VERSION = "audio-token-planning-v1"


@dataclass(frozen=True)
class _ConvolutionPlanningEstimator:
    backend_identity: AudioBackendIdentity
    version: str
    sample_rate: int
    kernels: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.version, name="planning estimator version")
        _positive_int(self.sample_rate, name="planning estimator sample_rate")
        _positive_int_tuple(self.kernels, name="planning estimator kernels")
        _positive_int_tuple(self.strides, name="planning estimator strides")
        if len(self.kernels) != len(self.strides):
            raise ValueError("planning estimator kernels and strides must align.")

    @property
    def identity(self) -> str:
        return _identity(
            backend=self.backend_identity,
            version=self.version,
            algorithm="unpadded-convolution",
            parameters={
                "sample_rate": self.sample_rate,
                "kernels": self.kernels,
                "strides": self.strides,
            },
        )

    def estimate(self, source_frames: int, source_rate: int) -> int:
        units = _resampled_frames(
            source_frames,
            source_rate,
            target_rate=self.sample_rate,
        )
        for kernel, stride in zip(self.kernels, self.strides, strict=True):
            units = (units - kernel) // stride + 1
        if units <= 0:
            raise ValueError(
                "source audio is shorter than the planning estimator receptive field."
            )
        return units


@dataclass(frozen=True)
class _ChunkedPooledPlanningEstimator:
    backend_identity: AudioBackendIdentity
    version: str
    sample_rate: int
    feature_hop_length: int
    convolution_stride: int
    pooling: int
    maximum_chunk_frames: int

    def __post_init__(self) -> None:
        _non_empty_string(self.version, name="planning estimator version")
        _positive_int(self.sample_rate, name="planning estimator sample_rate")
        _positive_int(
            self.feature_hop_length,
            name="planning estimator feature_hop_length",
        )
        _positive_int(
            self.convolution_stride,
            name="planning estimator convolution_stride",
        )
        _positive_int(self.pooling, name="planning estimator pooling")
        _positive_int(
            self.maximum_chunk_frames,
            name="planning estimator maximum_chunk_frames",
        )

    @property
    def identity(self) -> str:
        return _identity(
            backend=self.backend_identity,
            version=self.version,
            algorithm="chunked-feature-pooling",
            parameters={
                "sample_rate": self.sample_rate,
                "feature_hop_length": self.feature_hop_length,
                "convolution_stride": self.convolution_stride,
                "pooling": self.pooling,
                "maximum_chunk_frames": self.maximum_chunk_frames,
            },
        )

    def estimate(self, source_frames: int, source_rate: int) -> int:
        frames = _resampled_frames(
            source_frames,
            source_rate,
            target_rate=self.sample_rate,
        )
        full_chunks, remainder = divmod(frames, self.maximum_chunk_frames)
        units = full_chunks * self._chunk_units(self.maximum_chunk_frames)
        if remainder:
            units += self._chunk_units(remainder)
        return units

    def _chunk_units(self, frames: int) -> int:
        features = _ceil_div(frames, self.feature_hop_length)
        convolved = _ceil_div(features, self.convolution_stride)
        return _ceil_div(convolved, self.pooling)


def convolution_planning_estimator(
    *,
    backend_identity: AudioBackendIdentity,
    version: str,
    sample_rate: int,
    kernels: tuple[int, ...],
    strides: tuple[int, ...],
) -> AudioPlanningEstimator:
    return _ConvolutionPlanningEstimator(
        backend_identity=backend_identity,
        version=version,
        sample_rate=sample_rate,
        kernels=kernels,
        strides=strides,
    )


def chunked_pooled_planning_estimator(
    *,
    backend_identity: AudioBackendIdentity,
    version: str,
    sample_rate: int,
    feature_hop_length: int,
    convolution_stride: int,
    pooling: int,
    maximum_chunk_frames: int,
) -> AudioPlanningEstimator:
    return _ChunkedPooledPlanningEstimator(
        backend_identity=backend_identity,
        version=version,
        sample_rate=sample_rate,
        feature_hop_length=feature_hop_length,
        convolution_stride=convolution_stride,
        pooling=pooling,
        maximum_chunk_frames=maximum_chunk_frames,
    )


def _resampled_frames(source_frames: int, source_rate: int, *, target_rate: int) -> int:
    _positive_int(source_frames, name="source_frames")
    _positive_int(source_rate, name="source_rate")
    return _ceil_div(source_frames * target_rate, source_rate)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _identity(
    *,
    backend: AudioBackendIdentity,
    version: str,
    algorithm: str,
    parameters: dict[str, object],
) -> str:
    payload = {
        "algorithm": algorithm,
        "backend": {
            "artifact": backend.artifact,
            "preset": backend.preset,
            "revision": backend.revision,
        },
        "parameters": parameters,
        "version": version,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{PLANNING_IDENTITY_VERSION}:{digest}"


def _positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _positive_int_tuple(value: object, *, name: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{name} must be a non-empty tuple of integers.")
    for item in value:
        _positive_int(item, name=f"{name} item")


def _non_empty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


__all__ = ["chunked_pooled_planning_estimator", "convolution_planning_estimator"]
