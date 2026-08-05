from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .protocol import AudioCapability, CodecCapability
from .spec import AudioBackendIdentity, AudioCodeSpec

CodecLoader = Callable[..., Any]


@dataclass(frozen=True)
class CodecEntry:
    """Static preset descriptor backed by at most one shared model loader."""

    name: str
    capabilities: frozenset[CodecCapability]
    loader: CodecLoader | None = None
    audio_capabilities: frozenset[AudioCapability] = frozenset()
    audio_spec: AudioCodeSpec | None = None
    identity: AudioBackendIdentity | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("codec registry entry name must not be empty.")
        if not self.capabilities and not self.audio_capabilities:
            raise ValueError(f"codec registry entry {self.name!r} must declare capabilities.")
        if self.capabilities and self.loader is None:
            raise ValueError(
                f"codec registry entry {self.name!r} with legacy codec capabilities "
                "must provide one backend loader."
            )
        if self.audio_capabilities and self.audio_spec is None:
            raise ValueError(
                f"codec registry entry {self.name!r} with audio capabilities must "
                "provide static audio_spec metadata."
            )

    @property
    def resolved_identity(self) -> AudioBackendIdentity:
        if self.identity is not None:
            return self.identity
        return AudioBackendIdentity(preset=self.name)


__all__ = ["CodecEntry", "CodecLoader"]
