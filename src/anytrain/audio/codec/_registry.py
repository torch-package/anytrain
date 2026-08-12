from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .planning import AudioPlanningEstimator
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
    planning_estimator: AudioPlanningEstimator | None = None
    tokenizer: TokenizerFactory | None = None
    detokenizer: DetokenizerFactory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("codec registry entry name must be a string.")
        if not self.name:
            raise ValueError("codec registry entry name must not be empty.")
        if self.loader is not None and not callable(self.loader):
            raise TypeError(f"codec registry entry {self.name!r} loader must be callable.")
        if not isinstance(self.audio_capabilities, frozenset) or any(
            not isinstance(capability, AudioCapability)
            for capability in self.audio_capabilities
        ):
            raise TypeError(
                f"codec registry entry {self.name!r} capabilities must be a frozenset "
                "of AudioCapability values."
            )
        if not self.audio_capabilities:
            raise ValueError(f"codec registry entry {self.name!r} must declare capabilities.")
        if self.audio_spec is None:
            raise ValueError(
                f"codec registry entry {self.name!r} with audio capabilities must "
                "provide static audio_spec metadata."
            )
        if not isinstance(self.audio_spec, AudioCodeSpec):
            raise TypeError(
                f"codec registry entry {self.name!r} audio_spec must be an AudioCodeSpec."
            )
        if self.identity is not None:
            if not isinstance(self.identity, AudioBackendIdentity):
                raise TypeError(
                    f"codec registry entry {self.name!r} identity must be an "
                    "AudioBackendIdentity."
                )
            if self.identity.preset != self.name:
                raise ValueError(
                    f"codec registry entry {self.name!r} identity preset must match its name."
                )
        if self.tokenizer is not None and not callable(self.tokenizer):
            raise TypeError(
                f"codec registry entry {self.name!r} tokenizer factory must be callable."
            )
        if self.detokenizer is not None and not callable(self.detokenizer):
            raise TypeError(
                f"codec registry entry {self.name!r} detokenizer factory must be callable."
            )
        declares_tokenizer = AudioCapability.TOKENIZER in self.audio_capabilities
        if declares_tokenizer != (self.tokenizer is not None):
            raise ValueError(
                f"codec registry entry {self.name!r} must provide a tokenizer factory "
                "exactly when it declares the tokenizer capability."
            )
        if self.planning_estimator is not None:
            if not declares_tokenizer:
                raise ValueError(
                    f"codec registry entry {self.name!r} cannot provide a planning "
                    "estimator without the tokenizer capability."
                )
            if self.planning_estimator.backend_identity != self.resolved_identity:
                raise ValueError(
                    f"codec registry entry {self.name!r} planning estimator must use "
                    "the registered backend identity."
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
