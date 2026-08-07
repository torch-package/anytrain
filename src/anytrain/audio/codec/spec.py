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
    """Static code inventory required by an audio detokenizer.

    ``names[i]``, ``vocab_sizes[i]`` and ``frame_rates[i]`` describe the same
    code stream.  A ``None`` frame rate marks a fixed-length stream (for
    example BiCodec's global speaker codes); every other stream is aligned to
    the declared rate.
    """

    names: tuple[str, ...]
    vocab_sizes: tuple[int, ...]
    sample_rate: int
    frame_rates: tuple[float | None, ...]

    def __post_init__(self) -> None:
        _names(self.names)
        _codebook_sizes(self.vocab_sizes, name="vocab_sizes", allow_empty=True)
        _positive_int(self.sample_rate, name="sample_rate")
        _frame_rates(self.frame_rates)
        if len(self.names) != len(self.vocab_sizes):
            raise ValueError("names and vocab_sizes must have the same length.")
        if len(self.names) != len(self.frame_rates):
            raise ValueError("names and frame_rates must have the same length.")

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        """Compatibility alias for code-oriented backend callers."""

        return self.vocab_sizes

    @property
    def primary_codebook_sizes(self) -> tuple[int, ...]:
        """The first code stream's vocabulary, retained for codec callers."""

        return self.vocab_sizes[:1]

    def select(self, names: tuple[str, ...]) -> AudioCodeSpec:
        """Return the sub-spec for the requested streams, in requested order."""

        requested = _names(names, name="selected names")
        indices = self._indices(requested)
        return AudioCodeSpec(
            names=requested,
            vocab_sizes=tuple(self.vocab_sizes[index] for index in indices),
            sample_rate=self.sample_rate,
            frame_rates=tuple(self.frame_rates[index] for index in indices),
        )

    def exclude(self, names: tuple[str, ...]) -> AudioCodeSpec:
        """Return the sub-spec after removing the requested streams."""

        excluded_names = _names(names, name="excluded names")
        self._indices(excluded_names)
        excluded = set(excluded_names)
        selected = tuple(name for name in self.names if name not in excluded)
        return self.select(selected)

    def _indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        positions = {name: index for index, name in enumerate(self.names)}
        missing = tuple(name for name in names if name not in positions)
        if missing:
            raise ValueError(
                f"unknown code names {missing!r}; available names are {self.names!r}."
            )
        return tuple(positions[name] for name in names)


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


def _codebook_sizes(value: object, *, name: str, allow_empty: bool = False) -> None:
    if not allow_empty and not _optional_codebook_sizes(value, name=name):
        raise ValueError(f"{name} must not be empty.")
    _optional_codebook_sizes(value, name=name)


def _optional_codebook_sizes(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of integers.")
    resolved: list[int] = []
    for index, size in enumerate(value):
        _positive_int(size, name=f"{name}[{index}]")
        resolved.append(int(size))
    return tuple(resolved)


def _names(value: object, *, name: str = "names") -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of strings.")
    for index, item in enumerate(value):
        _non_empty_string(item, name=f"{name}[{index}]")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique values.")
    return value


def _frame_rates(value: object) -> tuple[float | None, ...]:
    if not isinstance(value, tuple):
        raise TypeError("frame_rates must be a tuple of positive numbers or None.")
    for index, rate in enumerate(value):
        if rate is not None:
            _positive_float(rate, name=f"frame_rates[{index}]")
    return value


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
