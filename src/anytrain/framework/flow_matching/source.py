from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._validation import finite_real, non_negative_int, positive_int


@dataclass(frozen=True)
class GaussianSource:
    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.randn_like(x_1)


@dataclass(frozen=True)
class UniformSource:
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        low = finite_real(self.low, "low")
        high = finite_real(self.high, "high")
        if high <= low:
            raise ValueError(f"high must be greater than low, got low={self.low}, high={self.high}.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.empty_like(x_1).uniform_(self.low, self.high)


@dataclass(frozen=True)
class UniformTokenSource:
    vocab_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocab_size", positive_int(self.vocab_size, "vocab_size"))

    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=x_1.shape,
            device=x_1.device,
            dtype=torch.long,
        )


@dataclass(frozen=True)
class MaskTokenSource:
    mask_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_id", non_negative_int(self.mask_id, "mask_id"))

    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.full_like(x_1, fill_value=self.mask_id, dtype=torch.long)


__all__ = [
    "GaussianSource",
    "MaskTokenSource",
    "UniformSource",
    "UniformTokenSource",
]
