from __future__ import annotations

import math
from math import prod
from numbers import Real
from typing import Protocol, cast, runtime_checkable

from torch import Tensor


class SemanticCodebookCodec(Protocol):
    @property
    def semantic_codebook(self) -> Tensor: ...


@runtime_checkable
class _CodebookCapability(Protocol):
    @property
    def semantic_codebook(self) -> Tensor: ...


@runtime_checkable
class _SemanticFeatureCapability(Protocol):
    @property
    def semantic_feature_dim(self) -> int: ...


@runtime_checkable
class _CodebookSizesCapability(Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...


@runtime_checkable
class _SampleRateCapability(Protocol):
    @property
    def sample_rate(self) -> int: ...


@runtime_checkable
class _FrameRateCapability(Protocol):
    @property
    def frame_rate(self) -> float: ...


@runtime_checkable
class _FsqLevelsCapability(Protocol):
    @property
    def fsq_levels(self) -> tuple[tuple[int, ...], ...]: ...


@runtime_checkable
class _FsqLevelValuesCapability(Protocol):
    @property
    def fsq_level_values(
        self,
    ) -> tuple[tuple[tuple[float, ...], ...], ...] | None: ...


@runtime_checkable
class _FsqRadixCapability(Protocol):
    @property
    def fsq_radix_order(self) -> str: ...


def codebook_codec(codec: object) -> SemanticCodebookCodec:
    if not isinstance(codec, _CodebookCapability):
        raise TypeError("audio embeddings require a semantic codebook capability.")
    _semantic_codebook(codec.semantic_codebook)
    return cast(SemanticCodebookCodec, codec)


def semantic_feature_dim(codec: object) -> int:
    if isinstance(codec, _CodebookCapability):
        return int(_semantic_codebook(codec.semantic_codebook).size(-1))
    if not isinstance(codec, _SemanticFeatureCapability):
        raise TypeError("audio backend requires a semantic codebook or feature dimension.")
    return _positive_int(codec.semantic_feature_dim, "semantic feature dimension")


def frame_codebook_sizes(codec: object) -> tuple[int, ...]:
    if not isinstance(codec, _CodebookSizesCapability):
        raise TypeError("audio backend codebook metadata is required.")
    return _positive_sizes(codec.codebook_sizes, "audio backend codebooks")


def fsq_levels(codec: object) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(codec, _SemanticFeatureCapability):
        return None
    if semantic_feature_dim(codec) != 1 or not isinstance(codec, _FsqLevelsCapability):
        return None
    raw = codec.fsq_levels
    if not isinstance(raw, tuple) or not raw:
        raise TypeError("fsq_levels must be a non-empty tuple of stages.")
    levels: list[tuple[int, ...]] = []
    for stage_index, stage in enumerate(raw):
        if not isinstance(stage, tuple) or not stage:
            raise TypeError(f"fsq_levels[{stage_index}] must be a non-empty tuple.")
        resolved_stage: list[int] = []
        for level_index, level in enumerate(stage):
            if isinstance(level, bool) or not isinstance(level, int):
                raise TypeError(
                    f"fsq_levels[{stage_index}][{level_index}] must be an integer."
                )
            if level < 2:
                raise ValueError("FSQ levels must be at least 2.")
            resolved_stage.append(level)
        levels.append(tuple(resolved_stage))
    resolved_levels = tuple(levels)
    if isinstance(codec, _CodebookSizesCapability):
        sizes = frame_codebook_sizes(codec)
        if len(sizes) != len(resolved_levels):
            raise ValueError("fsq_levels must align with codebook sizes.")
        for size, stage in zip(sizes, resolved_levels, strict=True):
            if prod(stage) != size:
                raise ValueError(
                    f"FSQ levels {stage} must multiply to codebook size {size}."
                )
    return resolved_levels


def fsq_level_values(
    codec: object,
) -> tuple[tuple[tuple[float, ...], ...], ...] | None:
    levels = fsq_levels(codec)
    if levels is None or not isinstance(codec, _FsqLevelValuesCapability):
        return None
    raw = codec.fsq_level_values
    if raw is None:
        return None
    if not isinstance(raw, tuple):
        raise TypeError("fsq_level_values must be a tuple of stages.")
    if len(raw) != len(levels):
        raise ValueError("fsq_level_values must align with FSQ stages.")
    result: list[tuple[tuple[float, ...], ...]] = []
    for stage_levels, stage_values in zip(levels, raw, strict=True):
        if not isinstance(stage_values, tuple):
            raise TypeError("each fsq_level_values stage must be a tuple.")
        if len(stage_values) != len(stage_levels):
            raise ValueError("fsq_level_values must align with FSQ digits.")
        digits: list[tuple[float, ...]] = []
        for level, values in zip(stage_levels, stage_values, strict=True):
            if not isinstance(values, tuple):
                raise TypeError("each FSQ digit value set must be a tuple.")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise TypeError("FSQ digit values must be real numbers.")
            digit = tuple(float(value) for value in values)
            if len(digit) != level:
                raise ValueError("FSQ digit values must match their level count.")
            if any(not math.isfinite(value) for value in digit):
                raise ValueError("FSQ digit values must be finite.")
            if any(
                left >= right
                for left, right in zip(digit, digit[1:], strict=False)
            ):
                raise ValueError("FSQ digit values must be strictly increasing.")
            digits.append(digit)
        result.append(tuple(digits))
    return tuple(result)


def fsq_radix_order(codec: object) -> str | None:
    if fsq_levels(codec) is None or not isinstance(codec, _FsqRadixCapability):
        return None
    value = codec.fsq_radix_order
    if not isinstance(value, str) or not value:
        raise TypeError("FSQ radix order must be a non-empty string.")
    return value


def codec_sample_rate(codec: object) -> int:
    if not isinstance(codec, _SampleRateCapability):
        raise TypeError("audio backend sample rate metadata is required.")
    return _positive_int(codec.sample_rate, "audio backend sample rate")


def codec_frame_rate(codec: object) -> float:
    if not isinstance(codec, _FrameRateCapability):
        raise TypeError("audio backend frame rate metadata is required.")
    value = codec.frame_rate
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("audio backend frame rate must be a number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("audio backend frame rate must be finite and positive.")
    return result


def _positive_sizes(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{name} must be a non-empty tuple of integers.")
    return tuple(_positive_int(size, f"{name} size") for size in value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _semantic_codebook(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("semantic codebook must be a Tensor.")
    if value.dim() not in {2, 3} or any(size <= 0 for size in value.shape):
        raise ValueError(
            "semantic codebook must have non-empty [vocab, dim] or "
            "[codebooks, vocab, dim] shape."
        )
    return value


__all__ = [
    "SemanticCodebookCodec",
    "codebook_codec",
    "codec_frame_rate",
    "codec_sample_rate",
    "frame_codebook_sizes",
    "fsq_level_values",
    "fsq_levels",
    "fsq_radix_order",
    "semantic_feature_dim",
]
