"""Public routing-grid descriptors for MoE layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
Grid1dContext = Literal["global", "local"]


@dataclass(frozen=True)
class NativeGrid:
    """No tiling: route and execute at the caller-visible native resolution."""


@dataclass(frozen=True)
class Grid1d:
    """Tile a 1D sequence into routing segments.

    segment_size is the logical owned-region stride used by routing and
    restore. context and fill are optional user overrides; when left as None
    they are resolved from the bound expert.
    """

    segment_size: int
    context: Grid1dContext | None = None
    fill: PaddingMode | None = None

    def __post_init__(self) -> None:
        if self.segment_size <= 0:
            raise ValueError(
                f"Grid1d.segment_size must be positive, got {self.segment_size}."
            )
        if self.context is not None and self.context not in {"global", "local"}:
            raise ValueError(f"unsupported Grid1d.context {self.context!r}.")
        if self.fill is not None and self.fill not in {
            "zeros",
            "reflect",
            "replicate",
            "circular",
        }:
            raise ValueError(f"unsupported Grid1d.fill {self.fill!r}.")


MoEGrid = NativeGrid | Grid1d

__all__ = [
    "Grid1dContext",
    "Grid1d",
    "NativeGrid",
    "MoEGrid",
]
