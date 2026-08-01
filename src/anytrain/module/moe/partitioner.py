"""MoE partition/restore helpers for routing-grid layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


class PartitionLayout(Protocol):
    mask: Tensor | None

    def restore(self, cell_outputs: Tensor) -> Tensor:
        """Fold expert cell outputs back to the caller-visible layout."""
        ...


class MoEPartitioner(Protocol):
    def partition(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, PartitionLayout]:
        """Return expert cells, optional router features, and restore layout."""
        ...


@dataclass(eq=False)
class IdentityLayout:
    mask: Tensor | None = None

    def restore(self, cell_outputs: Tensor) -> Tensor:
        return cell_outputs


class IdentityPartitioner:
    """Feature-last identity partitioner: route and execute on the same cells."""

    def partition(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, PartitionLayout]:
        return x, None, IdentityLayout(mask=mask)


__all__ = [
    "IdentityLayout",
    "IdentityPartitioner",
    "MoEPartitioner",
    "PartitionLayout",
]
