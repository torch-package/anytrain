from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class TimeBucketedMean:
    edges: Tensor
    total: Tensor
    count: Tensor

    @property
    def mean(self) -> Tensor:
        return self.total / self.count.clamp_min(1)

    @property
    def populated(self) -> Tensor:
        return self.count > 0


def time_bucketed_mean(
    values: Tensor,
    time: Tensor,
    *,
    bucket_count: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> TimeBucketedMean:
    bucket_count, t_min, t_max = _validate_bucket_args(bucket_count, t_min, t_max)
    _validate_tensors(values, time)

    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    flat_values = values.reshape(-1).to(dtype=dtype)
    flat_time = time.reshape(-1).to(dtype=dtype)
    if not bool(torch.isfinite(flat_time).all().item()):
        raise ValueError("time must contain only finite values.")
    if bool(((flat_time < t_min) | (flat_time > t_max)).any().item()):
        raise ValueError(
            f"time values must be within [t_min, t_max], got t_min={t_min}, t_max={t_max}."
        )

    width = t_max - t_min
    bucket = torch.floor((flat_time - t_min) / width * bucket_count).long()
    bucket = bucket.clamp(min=0, max=bucket_count - 1)

    total = torch.zeros(bucket_count, device=values.device, dtype=dtype)
    count = torch.zeros(bucket_count, device=values.device, dtype=dtype)
    total.index_add_(0, bucket, flat_values)
    count.index_add_(0, bucket, torch.ones_like(flat_values, dtype=dtype))
    edges = torch.linspace(
        t_min,
        t_max,
        bucket_count + 1,
        device=values.device,
        dtype=dtype,
    )
    return TimeBucketedMean(edges=edges, total=total, count=count)


def _validate_bucket_args(
    bucket_count: int,
    t_min: float,
    t_max: float,
) -> tuple[int, float, float]:
    if isinstance(bucket_count, bool) or not isinstance(bucket_count, int):
        raise TypeError("bucket_count must be an integer.")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    t_min = _finite_real(t_min, "t_min")
    t_max = _finite_real(t_max, "t_max")
    if t_min < 0:
        raise ValueError("t_min must be non-negative.")
    if t_max <= t_min:
        raise ValueError(
            f"t_max must be greater than t_min, got t_min={t_min}, t_max={t_max}."
        )
    return bucket_count, t_min, t_max


def _validate_tensors(values: Tensor, time: Tensor) -> None:
    if not isinstance(values, Tensor) or not isinstance(time, Tensor):
        raise TypeError("values and time must be tensors.")
    if values.shape != time.shape:
        raise ValueError(
            f"values and time must have the same shape, got {values.shape} and {time.shape}."
        )
    if values.ndim == 0:
        raise ValueError("values and time must include a batch dimension.")
    if values.device != time.device:
        raise ValueError("values and time must be on the same device.")
    if not values.is_floating_point():
        raise TypeError(f"values must be a floating tensor, got {values.dtype}.")
    if not time.is_floating_point():
        raise TypeError(f"time must be a floating tensor, got {time.dtype}.")


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a host real number, not bool or Tensor.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


__all__ = ["TimeBucketedMean", "time_bucketed_mean"]
