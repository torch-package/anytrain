from __future__ import annotations

from typing import Protocol

from .spec import AudioBackendIdentity


class AudioPlanningEstimator(Protocol):
    """Model-free estimate of tokenizer units from source audio frames."""

    @property
    def backend_identity(self) -> AudioBackendIdentity: ...

    @property
    def version(self) -> str: ...

    @property
    def identity(self) -> str: ...

    def estimate(self, source_frames: int, source_rate: int) -> int: ...


__all__ = ["AudioPlanningEstimator"]
