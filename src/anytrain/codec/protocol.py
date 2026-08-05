from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from typing import TYPE_CHECKING, TypeVar, Union

from torch import Tensor
from typing_extensions import Protocol

from anytrain._compat import StrEnum

if TYPE_CHECKING:
    from .spec import AudioBackendIdentity, AudioCodeSpec


class AcousticLayout(StrEnum):
    """Relationship between semantic and acoustic unit axes."""

    FRAME_ALIGNED = auto()
    FIXED_LENGTH = auto()


class CodecCapability(StrEnum):
    """Stable codec contracts that a backend loader can advertise."""

    FRAME_CODES = auto()
    SEMANTIC_ACOUSTIC = auto()
    SEMANTIC_GLOBAL = auto()


class AudioCapability(StrEnum):
    """Waveform conversion directions exposed by an audio backend preset."""

    TOKENIZER = auto()
    DETOKENIZER = auto()


@dataclass(frozen=True)
class SemanticAcousticCodes:
    """Structured codec units with backend-defined acoustic content.

    ``semantic`` has shape ``[batch, time, semantic_codebook]`` and
    ``acoustic`` has shape ``[batch, acoustic_unit, acoustic_codebook]``.
    ``AcousticLayout`` defines how the two middle axes relate.
    """

    semantic: Tensor
    acoustic: Tensor


@dataclass(frozen=True)
class SemanticGlobalCodes:
    """Semantic time units plus a separate fixed global unit set.

    ``semantic`` has shape ``[batch, time, semantic_codebook]`` and
    ``global_codes`` has shape ``[batch, global_unit, global_codebook]``.
    The global axis is independent of the semantic time axis.
    """

    semantic: Tensor
    global_codes: Tensor


AudioCodes = Union[Tensor, SemanticAcousticCodes, SemanticGlobalCodes]
CodesT_co = TypeVar("CodesT_co", covariant=True)
CodesT_contra = TypeVar("CodesT_contra", contravariant=True)
CodesT = TypeVar("CodesT")


class CodecBackend(Protocol):
    """Identity and output-audio contract shared by codec backends."""

    @property
    def name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float:
        """Rate of the backend's primary temporal unit."""
        ...


class AudioTokenizer(Protocol[CodesT_co]):
    """Capability view that converts waveform batches into one audio code schema."""

    @property
    def backend(self) -> CodecBackend: ...

    @property
    def identity(self) -> AudioBackendIdentity: ...

    @property
    def spec(self) -> AudioCodeSpec: ...

    def tokenize(self, audio: Tensor, sample_rate: int) -> CodesT_co: ...


class AudioDetokenizer(Protocol[CodesT_contra]):
    """Capability view that converts one audio code schema back into waveform batches."""

    @property
    def backend(self) -> CodecBackend: ...

    @property
    def identity(self) -> AudioBackendIdentity: ...

    @property
    def spec(self) -> AudioCodeSpec: ...

    def detokenize(self, codes: CodesT_contra) -> Tensor: ...


class AudioCodec(AudioTokenizer[CodesT], AudioDetokenizer[CodesT], Protocol[CodesT]):
    """Backend view providing both waveform tokenization directions."""


class FrameTokenizerBackend(CodecBackend, Protocol):
    """Directional backend that converts waveform batches to frame codes."""

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Encode [batch, channels, time] audio as [batch, frame, codebook] ids."""
        ...


class FrameDetokenizerBackend(CodecBackend, Protocol):
    """Directional backend that converts complete frame codes to waveforms."""

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def decode(self, codes: Tensor) -> Tensor:
        """Decode [batch, frame, codebook] ids at ``self.sample_rate``."""
        ...


class FrameCodec(FrameTokenizerBackend, FrameDetokenizerBackend, Protocol):
    """Codec providing both directions for units on one shared frame axis."""


class SemanticAcousticCodec(CodecBackend, Protocol):
    """Codec exposing structured units used by semantic-acoustic training.

    This is the second backend capability beside :class:`FrameCodec`.  The
    semantic-only waveform decoder is intentionally owned by the downstream
    ``semantic-acoustic-codec`` runtime, not by this backend protocol.
    """

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_layout(self) -> AcousticLayout: ...

    @property
    def semantic_codebook(self) -> Tensor: ...

    @property
    def semantic_frame_rate(self) -> float: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_unit_length(self) -> int | None: ...

    def tokenize(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> SemanticAcousticCodes:
        """Tokenize [batch, channels, time] audio into structured units."""
        ...

    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        """Decode structured units at ``self.sample_rate``."""
        ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        """Convert ``[batch, acoustic_unit, codebook]`` ids to continuous features."""
        ...

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        """Decode semantic ids and ``[batch, acoustic_unit, feature]`` features."""
        ...


class SemanticGlobalCodec(CodecBackend, Protocol):
    """Codec exposing semantic time units and independent global units."""

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def global_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def semantic_codebook(self) -> Tensor: ...

    @property
    def semantic_frame_rate(self) -> float: ...

    @property
    def global_feature_dim(self) -> int: ...

    @property
    def global_unit_length(self) -> int: ...

    def tokenize(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> SemanticGlobalCodes:
        """Tokenize audio into semantic time units and global units."""
        ...

    def detokenize(self, codes: SemanticGlobalCodes) -> Tensor:
        """Decode semantic and global units at ``self.sample_rate``."""
        ...

    def global_codes_to_features(self, global_codes: Tensor) -> Tensor:
        """Convert ``[batch, global_unit, codebook]`` ids to global features."""
        ...

    def decode_features(self, semantic_codes: Tensor, global_features: Tensor) -> Tensor:
        """Decode semantic ids and ``[batch, global_unit, feature]`` features."""
        ...


__all__ = [
    "AcousticLayout",
    "AudioCapability",
    "AudioCodec",
    "AudioCodes",
    "AudioDetokenizer",
    "AudioTokenizer",
    "CodecBackend",
    "CodecCapability",
    "FrameCodec",
    "FrameDetokenizerBackend",
    "FrameTokenizerBackend",
    "SemanticAcousticCodec",
    "SemanticAcousticCodes",
    "SemanticGlobalCodec",
    "SemanticGlobalCodes",
]
