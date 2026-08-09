from __future__ import annotations

from enum import auto
from typing import Protocol

from anytrain._compat import StrEnum


class AudioCapability(StrEnum):
    TOKENIZER = auto()
    DETOKENIZER = auto()


class CodecBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...


__all__ = ["AudioCapability", "CodecBackend"]
