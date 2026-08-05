from __future__ import annotations

import math
from dataclasses import dataclass
from enum import auto

from torch import Tensor

from anytrain._compat import StrEnum

from .protocol import AcousticLayout, SemanticAcousticCodec, SemanticGlobalCodec


class AudioCodeSchema(StrEnum):
    """Portable layouts used by audio tokenizer and detokenizer capability views."""

    FRAME = auto()
    SEMANTIC_ACOUSTIC = auto()
    SEMANTIC_GLOBAL = auto()


@dataclass(frozen=True)
class AudioBackendIdentity:
    """Resolved backend resource identity, independent from its token schema."""

    preset: str
    artifact: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.preset, name="preset")
        if self.artifact is not None:
            _non_empty_string(self.artifact, name="artifact")
        if self.revision is not None:
            _non_empty_string(self.revision, name="revision")


@dataclass(frozen=True)
class AudioCodeSpec:
    """Static token layout metadata for one registered audio backend preset."""

    view: str
    schema: AudioCodeSchema
    sample_rate: int | None
    frame_rate: float
    frame_codebook_sizes: tuple[int, ...] = ()
    semantic_codebook_sizes: tuple[int, ...] = ()
    acoustic_codebook_sizes: tuple[int, ...] = ()
    global_codebook_sizes: tuple[int, ...] = ()
    acoustic_layout: AcousticLayout | None = None
    acoustic_unit_length: int | None = None
    global_unit_length: int | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.view, name="view")
        if not isinstance(self.schema, AudioCodeSchema):
            raise TypeError("schema must be an AudioCodeSchema.")
        if self.sample_rate is not None:
            _positive_int(self.sample_rate, name="sample_rate")
        _positive_float(self.frame_rate, name="frame_rate")
        _optional_codebook_sizes(
            self.frame_codebook_sizes,
            name="frame_codebook_sizes",
        )
        _optional_codebook_sizes(
            self.semantic_codebook_sizes,
            name="semantic_codebook_sizes",
        )
        _optional_codebook_sizes(
            self.acoustic_codebook_sizes,
            name="acoustic_codebook_sizes",
        )
        _optional_codebook_sizes(
            self.global_codebook_sizes,
            name="global_codebook_sizes",
        )
        self._validate_schema()

    @property
    def primary_codebook_sizes(self) -> tuple[int, ...]:
        """Codebooks occupying the preset's primary temporal token axis."""

        if self.schema is AudioCodeSchema.FRAME:
            return self.frame_codebook_sizes
        return self.semantic_codebook_sizes

    def _validate_schema(self) -> None:
        if self.schema is AudioCodeSchema.FRAME:
            if not self.frame_codebook_sizes:
                raise ValueError("frame schema requires frame_codebook_sizes.")
            if (
                self.semantic_codebook_sizes
                or self.acoustic_codebook_sizes
                or self.global_codebook_sizes
            ):
                raise ValueError("frame schema must not declare structured codebooks.")
            if (
                self.acoustic_layout is not None
                or self.acoustic_unit_length is not None
                or self.global_unit_length is not None
            ):
                raise ValueError("frame schema must not declare structured unit metadata.")
            return

        if not self.semantic_codebook_sizes:
            raise ValueError("structured schema requires semantic_codebook_sizes.")
        if self.schema is AudioCodeSchema.SEMANTIC_ACOUSTIC:
            if not self.acoustic_codebook_sizes:
                raise ValueError("semantic-acoustic schema requires acoustic_codebook_sizes.")
            if self.global_codebook_sizes or self.global_unit_length is not None:
                raise ValueError("semantic-acoustic schema must not declare global unit metadata.")
            if not isinstance(self.acoustic_layout, AcousticLayout):
                raise TypeError(
                    "semantic-acoustic schema requires acoustic_layout to be an AcousticLayout."
                )
            if self.acoustic_layout is AcousticLayout.FIXED_LENGTH:
                if self.acoustic_unit_length is None:
                    raise ValueError(
                        "fixed-length semantic-acoustic schema requires acoustic_unit_length."
                    )
                _positive_int(self.acoustic_unit_length, name="acoustic_unit_length")
            elif self.acoustic_unit_length is not None:
                raise ValueError(
                    "frame-aligned semantic-acoustic schema must not set acoustic_unit_length."
                )
            return

        if not self.global_codebook_sizes:
            raise ValueError("semantic-global schema requires global_codebook_sizes.")
        if self.global_unit_length is None:
            raise ValueError("semantic-global schema requires global_unit_length.")
        _positive_int(self.global_unit_length, name="global_unit_length")
        if (
            self.acoustic_codebook_sizes
            or self.acoustic_layout is not None
            or self.acoustic_unit_length is not None
        ):
            raise ValueError("semantic-global schema must not declare acoustic unit metadata.")


@dataclass(frozen=True)
class SemanticAcousticCodecSpec:
    """Portable structure exposed by a semantic-acoustic codec backend."""

    sample_rate: int
    frame_rate: float
    semantic_frame_rate: float
    semantic_codebook_sizes: tuple[int, ...]
    semantic_embedding_dim: int
    acoustic_codebook_sizes: tuple[int, ...]
    acoustic_feature_dim: int
    acoustic_layout: AcousticLayout
    acoustic_unit_length: int | None

    def __post_init__(self) -> None:
        _positive_int(self.sample_rate, name="sample_rate")
        _positive_float(self.frame_rate, name="frame_rate")
        _positive_float(self.semantic_frame_rate, name="semantic_frame_rate")
        _codebook_sizes(self.semantic_codebook_sizes, name="semantic_codebook_sizes")
        _positive_int(self.semantic_embedding_dim, name="semantic_embedding_dim")
        _codebook_sizes(self.acoustic_codebook_sizes, name="acoustic_codebook_sizes")
        _positive_int(self.acoustic_feature_dim, name="acoustic_feature_dim")
        if not isinstance(self.acoustic_layout, AcousticLayout):
            raise TypeError("acoustic_layout must be an AcousticLayout.")
        if self.acoustic_layout is AcousticLayout.FIXED_LENGTH:
            if self.acoustic_unit_length is None:
                raise ValueError("fixed-length acoustic layout requires acoustic_unit_length.")
            _positive_int(self.acoustic_unit_length, name="acoustic_unit_length")
        elif self.acoustic_unit_length is not None:
            raise ValueError("frame-aligned acoustic layout must not set acoustic_unit_length.")


@dataclass(frozen=True)
class SemanticGlobalCodecSpec:
    """Portable structure exposed by a semantic-global codec backend."""

    sample_rate: int
    frame_rate: float
    semantic_frame_rate: float
    semantic_codebook_sizes: tuple[int, ...]
    semantic_embedding_dim: int
    global_codebook_sizes: tuple[int, ...]
    global_feature_dim: int
    global_unit_length: int

    def __post_init__(self) -> None:
        _positive_int(self.sample_rate, name="sample_rate")
        _positive_float(self.frame_rate, name="frame_rate")
        _positive_float(self.semantic_frame_rate, name="semantic_frame_rate")
        _codebook_sizes(self.semantic_codebook_sizes, name="semantic_codebook_sizes")
        _positive_int(self.semantic_embedding_dim, name="semantic_embedding_dim")
        _codebook_sizes(self.global_codebook_sizes, name="global_codebook_sizes")
        _positive_int(self.global_feature_dim, name="global_feature_dim")
        _positive_int(self.global_unit_length, name="global_unit_length")


def semantic_acoustic_spec(codec: SemanticAcousticCodec) -> SemanticAcousticCodecSpec:
    """Inspect and validate the portable structure of an instantiated codec."""

    semantic_codebook_sizes = codec.semantic_codebook_sizes
    semantic_codebook = codec.semantic_codebook
    _semantic_codebook(semantic_codebook, semantic_codebook_sizes)
    return SemanticAcousticCodecSpec(
        sample_rate=codec.sample_rate,
        frame_rate=codec.frame_rate,
        semantic_frame_rate=codec.semantic_frame_rate,
        semantic_codebook_sizes=semantic_codebook_sizes,
        semantic_embedding_dim=int(semantic_codebook.size(1)),
        acoustic_codebook_sizes=codec.acoustic_codebook_sizes,
        acoustic_feature_dim=codec.acoustic_feature_dim,
        acoustic_layout=codec.acoustic_layout,
        acoustic_unit_length=codec.acoustic_unit_length,
    )


def semantic_global_spec(codec: SemanticGlobalCodec) -> SemanticGlobalCodecSpec:
    """Inspect and validate the portable structure of a semantic-global codec."""

    semantic_codebook_sizes = codec.semantic_codebook_sizes
    semantic_codebook = codec.semantic_codebook
    _semantic_codebook(semantic_codebook, semantic_codebook_sizes)
    return SemanticGlobalCodecSpec(
        sample_rate=codec.sample_rate,
        frame_rate=codec.frame_rate,
        semantic_frame_rate=codec.semantic_frame_rate,
        semantic_codebook_sizes=semantic_codebook_sizes,
        semantic_embedding_dim=int(semantic_codebook.size(1)),
        global_codebook_sizes=codec.global_codebook_sizes,
        global_feature_dim=codec.global_feature_dim,
        global_unit_length=codec.global_unit_length,
    )


def _semantic_codebook(value: object, sizes: tuple[int, ...]) -> None:
    _codebook_sizes(sizes, name="semantic_codebook_sizes")
    if not isinstance(value, Tensor):
        raise TypeError("semantic_codebook must be a Tensor.")
    if value.dim() != 2:
        raise ValueError("semantic_codebook must have shape [vocab, embedding].")
    if value.size(0) != sizes[0]:
        raise ValueError(
            "semantic_codebook vocab axis must match semantic_codebook_sizes[0]: "
            f"{value.size(0)} != {sizes[0]}."
        )
    if value.size(1) <= 0:
        raise ValueError("semantic_codebook embedding dimension must be positive.")
    if not value.is_floating_point() or value.is_complex():
        raise TypeError("semantic_codebook must be floating point.")


def _codebook_sizes(value: object, *, name: str) -> None:
    if not _optional_codebook_sizes(value, name=name):
        raise ValueError(f"{name} must not be empty.")


def _optional_codebook_sizes(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of integers.")
    resolved: list[int] = []
    for index, size in enumerate(value):
        _positive_int(size, name=f"{name}[{index}]")
        resolved.append(int(size))
    return tuple(resolved)


def _non_empty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _positive_float(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


__all__ = [
    "AudioBackendIdentity",
    "AudioCodeSchema",
    "AudioCodeSpec",
    "SemanticAcousticCodecSpec",
    "SemanticGlobalCodecSpec",
    "semantic_acoustic_spec",
    "semantic_global_spec",
]
