from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .protocol import CodecCapability

CodecLoader = Callable[..., Any]


@dataclass(frozen=True)
class CodecEntry:
    """Static registry descriptor declared by each codec wrapper."""

    name: str
    capabilities: frozenset[CodecCapability]
    frame_loader: CodecLoader | None = None
    semantic_acoustic_loader: CodecLoader | None = None
    semantic_global_loader: CodecLoader | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("codec registry entry name must not be empty.")
        if not self.capabilities:
            raise ValueError(f"codec registry entry {self.name!r} must declare capabilities.")
        if (
            CodecCapability.FRAME_CODES in self.capabilities
            and self.frame_loader is None
        ):
            raise ValueError(f"codec registry entry {self.name!r} is missing frame_loader.")
        if (
            CodecCapability.SEMANTIC_ACOUSTIC in self.capabilities
            and self.semantic_acoustic_loader is None
        ):
            raise ValueError(
                f"codec registry entry {self.name!r} is missing semantic_acoustic_loader."
            )
        if (
            CodecCapability.SEMANTIC_GLOBAL in self.capabilities
            and self.semantic_global_loader is None
        ):
            raise ValueError(
                f"codec registry entry {self.name!r} is missing semantic_global_loader."
            )


__all__ = ["CodecEntry", "CodecLoader"]
