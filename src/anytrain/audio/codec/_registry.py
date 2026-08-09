from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .protocol import AudioCapability
from .spec import AudioBackendIdentity, AudioCodeSpec

CodecLoader = Callable[..., Any]
if TYPE_CHECKING:
    from anytrain.audio._detokenizer import AudioDetokenizer
    from anytrain.audio._tokenizer import AudioTokenizer

    from .protocol import CodecBackend

TokenizerFactory = Callable[
    ["CodecBackend", AudioCodeSpec, AudioBackendIdentity],
    "AudioTokenizer",
]
DetokenizerFactory = Callable[
    ["CodecBackend", AudioCodeSpec, AudioBackendIdentity],
    "AudioDetokenizer",
]


@dataclass(frozen=True)
class CodecEntry:
    """Static preset descriptor backed by at most one shared model loader."""

    name: str
    loader: CodecLoader | None = None
    audio_capabilities: frozenset[AudioCapability] = frozenset()
    audio_spec: AudioCodeSpec | None = None
    identity: AudioBackendIdentity | None = None
    tokenizer: TokenizerFactory | None = None
    detokenizer: DetokenizerFactory | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("codec registry entry name must not be empty.")
        if not self.audio_capabilities:
            raise ValueError(f"codec registry entry {self.name!r} must declare capabilities.")
        if self.audio_capabilities and self.audio_spec is None:
            raise ValueError(
                f"codec registry entry {self.name!r} with audio capabilities must "
                "provide static audio_spec metadata."
            )
        declares_tokenizer = AudioCapability.TOKENIZER in self.audio_capabilities
        if declares_tokenizer != (self.tokenizer is not None):
            raise ValueError(
                f"codec registry entry {self.name!r} must provide a tokenizer factory "
                "exactly when it declares the tokenizer capability."
            )
        declares_detokenizer = AudioCapability.DETOKENIZER in self.audio_capabilities
        if declares_detokenizer != (self.detokenizer is not None):
            raise ValueError(
                f"codec registry entry {self.name!r} must provide a detokenizer factory "
                "exactly when it declares the detokenizer capability."
            )

    @property
    def resolved_identity(self) -> AudioBackendIdentity:
        if self.identity is not None:
            return self.identity
        return AudioBackendIdentity(preset=self.name)


__all__ = [
    "CodecEntry",
    "CodecLoader",
    "DetokenizerFactory",
    "TokenizerFactory",
]
