from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor

Name = TypeVar("Name")


@dataclass(frozen=True)
class Metric:
    values: Tensor
    weights: Tensor

    def count(self) -> int:
        self._validate_structure()
        weights = self.weights.detach()
        integral = (
            weights.eq(weights.round()).all()
            if torch.is_floating_point(weights)
            else weights.new_tensor(True, dtype=torch.bool)
        )
        valid = torch.isfinite(weights).all() & weights.ge(0).all() & integral
        if not bool(valid):
            raise ValueError(
                "weighted metric weights must contain finite non-negative integer counts."
            )
        count = int(weights.sum().item())
        if count <= 0:
            raise ValueError("weighted metric weights must contain a positive total count.")
        return count

    def _mean(self, count: int) -> Tensor:
        dtype = torch.promote_types(self.values.dtype, torch.float32)
        weights = self.weights.to(device=self.values.device, dtype=dtype)
        values = self.values.to(dtype=dtype)
        values = values.masked_fill(weights.eq(0), 0)
        return (values * weights).sum() / count

    def reduced(self) -> tuple[Tensor, int]:
        count = self.count()
        return self._mean(count), count

    def _validate_structure(self) -> None:
        if self.values.shape != self.weights.shape:
            raise ValueError("weighted metric values and weights must have the same shape.")
        if not torch.is_floating_point(self.values):
            raise TypeError("weighted metric values must use a floating-point dtype.")
        if self.weights.is_complex():
            raise TypeError("weighted metric weights must be real-valued counts.")


def reduce(metrics: Mapping[Name, Metric]) -> dict[Name, tuple[Tensor, int]]:
    if not isinstance(metrics, Mapping):
        raise TypeError("weighted metrics must be a mapping.")
    if not metrics:
        raise ValueError("weighted metrics must contain at least one value.")

    counts: dict[int, tuple[Tensor, int]] = {}
    result: dict[Name, tuple[Tensor, int]] = {}
    for name, metric in metrics.items():
        if not isinstance(metric, Metric):
            raise TypeError("weighted metric values must be Metric instances.")
        key = id(metric.weights)
        cached = counts.get(key)
        if cached is None or cached[0] is not metric.weights:
            count = metric.count()
            counts[key] = (metric.weights, count)
        else:
            metric._validate_structure()
            count = cached[1]
        result[name] = (metric._mean(count), count)
    return result


__all__ = ["Metric", "reduce"]
