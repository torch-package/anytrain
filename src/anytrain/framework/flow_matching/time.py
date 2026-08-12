from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._validation import finite_real, positive_int, positive_real

DEFAULT_TIME_EPS = 1e-3


@dataclass(frozen=True)
class LogitNormalTimeSampler:
    mean: float = 0.0
    std: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        mean = finite_real(self.mean, "mean")
        std = positive_real(self.std, "std")
        t_min = finite_real(self.t_min, "t_min")
        t_max = finite_real(self.t_max, "t_max")
        if t_min < 0:
            raise ValueError(f"t_min must be non-negative, got {t_min}.")
        if t_max > 1:
            raise ValueError(f"t_max must be at most 1, got {t_max}.")
        if t_max <= t_min:
            raise ValueError(
                f"t_max must be greater than t_min, got t_min={t_min}, t_max={t_max}."
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "t_min", t_min)
        object.__setattr__(self, "t_max", t_max)

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        batch_size = positive_int(batch_size, "batch_size")

        z = torch.randn(batch_size, device=device) * self.std + self.mean
        t = torch.sigmoid(z)
        return t * (self.t_max - self.t_min) + self.t_min


@dataclass(frozen=True)
class UniformTimeSampler:
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        t_min = finite_real(self.t_min, "t_min")
        t_max = finite_real(self.t_max, "t_max")
        if t_min < 0:
            raise ValueError(f"t_min must be non-negative, got {t_min}.")
        if t_max > 1:
            raise ValueError(f"t_max must be at most 1, got {t_max}.")
        if t_max <= t_min:
            raise ValueError(
                f"t_max must be greater than t_min, got t_min={t_min}, t_max={t_max}."
            )
        object.__setattr__(self, "t_min", t_min)
        object.__setattr__(self, "t_max", t_max)

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        batch_size = positive_int(batch_size, "batch_size")

        t = torch.rand(batch_size, device=device)
        return t * (self.t_max - self.t_min) + self.t_min


__all__ = ["DEFAULT_TIME_EPS", "LogitNormalTimeSampler", "UniformTimeSampler"]
