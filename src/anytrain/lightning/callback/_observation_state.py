from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, cast

import lightning.pytorch as pl
import torch
from torch import Tensor

from anytrain.observation import (
    Bins,
    Curve,
    Distribution,
    Heatmap,
    Profile,
    Reduction,
)

from .. import _capability
from ._naming import validate_tag_part


@dataclass(frozen=True)
class _HistogramSchema:
    kind: Literal["bins", "distribution"]
    sizes: tuple[int, ...]
    group_names: tuple[str, ...]
    bucket_limits: tuple[tuple[float, ...], ...]
    ranges: tuple[tuple[float, float], ...]

    @property
    def groups(self) -> int:
        return len(self.sizes)

    @property
    def max_bins(self) -> int:
        return max(self.sizes)

    @property
    def dense_bins(self) -> int:
        return self.groups * self.max_bins


@dataclass
class _HistogramDelta:
    schema: _HistogramSchema
    mass: Tensor
    observations: Tensor
    total: Tensor
    sums: Tensor
    sum_squares: Tensor
    invalid: Tensor


@dataclass(frozen=True)
class _HistogramSnapshot:
    schema: _HistogramSchema
    mass: Tensor
    observations: Tensor
    total: Tensor
    sums: Tensor
    sum_squares: Tensor
    invalid: Tensor


@dataclass(frozen=True)
class _ProfileSchema:
    edges: tuple[float, ...]

    @property
    def bins(self) -> int:
        return len(self.edges) - 1


@dataclass
class _ProfileDelta:
    schema: _ProfileSchema
    counts: Tensor
    sums: Tensor
    sum_squares: Tensor
    total: Tensor
    invalid: Tensor


@dataclass(frozen=True)
class _ProfileSnapshot:
    schema: _ProfileSchema
    counts: Tensor
    sums: Tensor
    sum_squares: Tensor
    total: Tensor
    invalid: Tensor


class _CurveState:
    def __init__(self, value: Curve, *, reduction: Reduction) -> None:
        scalar = value.value.detach().float()
        self.reduction = reduction
        self.pending_sum = torch.zeros_like(scalar)
        self.pending_count = torch.zeros_like(scalar)
        self.last_sum = torch.zeros_like(scalar)
        self.last_count = torch.zeros_like(scalar)
        self.update(value, reduction=reduction)

    def update(self, value: Curve, *, reduction: Reduction) -> None:
        if reduction != self.reduction:
            raise ValueError("observation Curve reduction changed during training.")
        scalar = value.value.detach().float()
        self._to(scalar.device)
        if reduction == Reduction.Last:
            self.pending_sum.copy_(scalar)
            self.pending_count.fill_(1)
            return
        self.pending_sum.add_(scalar)
        self.pending_count.add_(1)

    def finalize(self, trainer: pl.Trainer, *, sync_distributed: bool) -> None:
        self.last_sum, self.last_count = _sum_many(
            trainer,
            (self.pending_sum, self.pending_count),
            sync_distributed,
        )
        self.pending_sum.zero_()
        self.pending_count.zero_()

    def value(self) -> Tensor | None:
        if not bool((self.last_count > 0).item()):
            return None
        return self.last_sum / self.last_count

    def discard_pending(self) -> None:
        self.pending_sum.zero_()
        self.pending_count.zero_()

    def _to(self, device: torch.device) -> None:
        if self.pending_sum.device == device:
            return
        self.pending_sum = self.pending_sum.to(device)
        self.pending_count = self.pending_count.to(device)
        self.last_sum = self.last_sum.to(device)
        self.last_count = self.last_count.to(device)


@dataclass(frozen=True)
class _HeatmapSchema:
    shape: tuple[int, int]
    dtype: str
    value_range: tuple[float, float] | None


class _HeatmapState:
    def __init__(
        self,
        value: Heatmap,
        *,
        keep_image: bool,
        max_elements: int,
    ) -> None:
        self.schema = _heatmap_schema(value)
        self.pending = False
        self.pending_schema: _HeatmapSchema | None = None
        self.pending_image: Tensor | None = None
        self.pending_error: tuple[Literal["value", "runtime"], str] | None = None
        self.update(
            value,
            keep_image=keep_image,
            max_elements=max_elements,
        )

    def update(
        self,
        value: Heatmap,
        *,
        keep_image: bool,
        max_elements: int,
    ) -> None:
        # A Heatmap is a snapshot. Repeated forward emissions replace the
        # pending snapshot, while backward checkpoint replay is disabled by the
        # callback lifecycle.
        schema = _heatmap_schema(value)
        self.pending = True
        self.pending_schema = schema
        self.pending_image = None
        self.pending_error = None
        if schema != self.schema:
            self.pending_error = (
                "value",
                "observation heatmap schema changed during training.",
            )
            return
        if value.values.numel() > max_elements:
            self.pending_error = (
                "value",
                "observation heatmap contains "
                f"{value.values.numel()} elements, exceeding "
                f"max_heatmap_elements={max_elements}; select a smaller semantic view "
                "inside the module.",
            )
            return
        if not keep_image:
            return
        try:
            self.pending_image = _heatmap_image(value)
        except Exception as error:
            self.pending_error = (
                "value" if isinstance(error, ValueError) else "runtime",
                f"heatmap image materialization failed: {type(error).__name__}: {error}",
            )

    def value(self) -> Tensor | None:
        return self.pending_image

    def discard_pending(self) -> None:
        self.pending = False
        self.pending_schema = None
        self.pending_image = None
        self.pending_error = None


class _HistogramState:
    def __init__(self, delta: _HistogramDelta, *, half_life_steps: float) -> None:
        self.schema = delta.schema
        self.half_life_steps = half_life_steps
        self.pending_mass = torch.zeros_like(delta.mass)
        self.pending_observations = torch.zeros_like(delta.observations)
        self.pending_total = torch.zeros_like(delta.total)
        self.pending_sums = torch.zeros_like(delta.sums)
        self.pending_sum_squares = torch.zeros_like(delta.sum_squares)
        self.pending_invalid = torch.zeros_like(delta.invalid)
        self.last_mass = torch.zeros_like(delta.mass)
        self.last_observations = torch.zeros_like(delta.observations)
        self.last_total = torch.zeros_like(delta.total)
        self.last_sums = torch.zeros_like(delta.sums)
        self.last_sum_squares = torch.zeros_like(delta.sum_squares)
        self.last_invalid = torch.zeros_like(delta.invalid)
        self.ema_mass = torch.zeros_like(delta.mass)
        self.ema_observations = torch.zeros_like(delta.observations)
        self.ema_total = torch.zeros_like(delta.total)
        self.ema_sums = torch.zeros_like(delta.sums)
        self.ema_sum_squares = torch.zeros_like(delta.sum_squares)
        self.invalid_seen = torch.zeros_like(delta.invalid)
        self.ema_step: int | None = None
        self.update(delta, half_life_steps=half_life_steps)

    def update(self, delta: _HistogramDelta, *, half_life_steps: float) -> None:
        if delta.schema != self.schema:
            raise ValueError("observation histogram schema changed during training.")
        if half_life_steps != self.half_life_steps:
            raise ValueError("observation EMA half-life changed during training.")
        self._to(delta.mass.device)
        self.pending_mass.add_(delta.mass)
        self.pending_observations.add_(delta.observations)
        self.pending_total.add_(delta.total)
        self.pending_sums.add_(delta.sums)
        self.pending_sum_squares.add_(delta.sum_squares)
        self.pending_invalid.add_(delta.invalid)

    def finalize(
        self,
        trainer: pl.Trainer,
        *,
        step: int,
        sync_distributed: bool,
    ) -> None:
        values = (
            self.pending_mass,
            self.pending_observations,
            self.pending_total,
            self.pending_sums,
            self.pending_sum_squares,
            self.pending_invalid,
        )
        reduced = _sum_many(trainer, values, sync_distributed)
        (
            self.last_mass,
            self.last_observations,
            self.last_total,
            self.last_sums,
            self.last_sum_squares,
            self.last_invalid,
        ) = reduced
        self.invalid_seen = self.invalid_seen + self.last_invalid

        if self.ema_step is None:
            self.ema_mass.copy_(self.last_mass)
            self.ema_observations.copy_(self.last_observations)
            self.ema_total.copy_(self.last_total)
            self.ema_sums.copy_(self.last_sums)
            self.ema_sum_squares.copy_(self.last_sum_squares)
        else:
            delta_steps = step - self.ema_step
            if delta_steps <= 0:
                raise RuntimeError("observation EMA step must move forwards.")
            decay = 2.0 ** (-delta_steps / self.half_life_steps)
            one_minus_decay = 1.0 - decay
            self.ema_mass.mul_(decay).add_(self.last_mass, alpha=one_minus_decay)
            self.ema_observations.mul_(decay).add_(
                self.last_observations,
                alpha=one_minus_decay,
            )
            self.ema_total.mul_(decay).add_(self.last_total, alpha=one_minus_decay)
            self.ema_sums.mul_(decay).add_(self.last_sums, alpha=one_minus_decay)
            self.ema_sum_squares.mul_(decay).add_(
                self.last_sum_squares,
                alpha=one_minus_decay,
            )
        self.ema_step = step
        self.discard_pending()

    def current(self) -> _HistogramSnapshot:
        return _HistogramSnapshot(
            self.schema,
            self.last_mass,
            self.last_observations,
            self.last_total,
            self.last_sums,
            self.last_sum_squares,
            self.last_invalid,
        )

    def ema(self) -> _HistogramSnapshot:
        return _HistogramSnapshot(
            self.schema,
            self.ema_mass,
            self.ema_observations,
            self.ema_total,
            self.ema_sums,
            self.ema_sum_squares,
            self.invalid_seen,
        )

    def discard_pending(self) -> None:
        self.pending_mass.zero_()
        self.pending_observations.zero_()
        self.pending_total.zero_()
        self.pending_sums.zero_()
        self.pending_sum_squares.zero_()
        self.pending_invalid.zero_()

    def state_dict(self) -> dict[str, Any]:
        if self.ema_step is None:
            raise RuntimeError("cannot save observation EMA before a completed step.")
        return {
            "schema": _schema_state(self.schema),
            "half_life_steps": self.half_life_steps,
            "ema_step": self.ema_step,
            "ema_mass": self.ema_mass.detach(),
            "ema_observations": self.ema_observations.detach(),
            "ema_total": self.ema_total.detach(),
            "ema_sums": self.ema_sums.detach(),
            "ema_sum_squares": self.ema_sum_squares.detach(),
            "invalid_seen": self.invalid_seen.detach(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> _HistogramState:
        schema = _schema_from_state(state.get("schema"))
        half_life_steps = _positive_real(
            state.get("half_life_steps"),
            "observation EMA half_life_steps",
        )
        ema_step = _non_negative_int(state.get("ema_step"), "observation EMA step")
        shape = (schema.groups, schema.max_bins)
        ema_mass = _state_tensor(state, "ema_mass", shape, non_negative=True)
        vector_shape = (schema.groups,)
        ema_observations = _state_tensor(
            state,
            "ema_observations",
            vector_shape,
            non_negative=True,
        )
        ema_total = _state_tensor(
            state,
            "ema_total",
            vector_shape,
            non_negative=True,
        )
        ema_sums = _state_tensor(state, "ema_sums", vector_shape)
        ema_sum_squares = _state_tensor(
            state,
            "ema_sum_squares",
            vector_shape,
            non_negative=True,
        )
        invalid_seen = _state_tensor(
            state,
            "invalid_seen",
            (),
            non_negative=True,
        )

        value = cls.__new__(cls)
        value.schema = schema
        value.half_life_steps = half_life_steps
        value.pending_mass = torch.zeros_like(ema_mass)
        value.pending_observations = torch.zeros_like(ema_observations)
        value.pending_total = torch.zeros_like(ema_total)
        value.pending_sums = torch.zeros_like(ema_sums)
        value.pending_sum_squares = torch.zeros_like(ema_sum_squares)
        value.pending_invalid = torch.zeros_like(invalid_seen)
        value.last_mass = torch.zeros_like(ema_mass)
        value.last_observations = torch.zeros_like(ema_observations)
        value.last_total = torch.zeros_like(ema_total)
        value.last_sums = torch.zeros_like(ema_sums)
        value.last_sum_squares = torch.zeros_like(ema_sum_squares)
        value.last_invalid = torch.zeros_like(invalid_seen)
        value.ema_mass = ema_mass
        value.ema_observations = ema_observations
        value.ema_total = ema_total
        value.ema_sums = ema_sums
        value.ema_sum_squares = ema_sum_squares
        value.invalid_seen = invalid_seen
        value.ema_step = ema_step
        return value

    def _to(self, device: torch.device) -> None:
        if self.pending_mass.device == device:
            return
        self.pending_mass = self.pending_mass.to(device)
        self.pending_observations = self.pending_observations.to(device)
        self.pending_total = self.pending_total.to(device)
        self.pending_sums = self.pending_sums.to(device)
        self.pending_sum_squares = self.pending_sum_squares.to(device)
        self.pending_invalid = self.pending_invalid.to(device)
        self.last_mass = self.last_mass.to(device)
        self.last_observations = self.last_observations.to(device)
        self.last_total = self.last_total.to(device)
        self.last_sums = self.last_sums.to(device)
        self.last_sum_squares = self.last_sum_squares.to(device)
        self.last_invalid = self.last_invalid.to(device)
        self.ema_mass = self.ema_mass.to(device)
        self.ema_observations = self.ema_observations.to(device)
        self.ema_total = self.ema_total.to(device)
        self.ema_sums = self.ema_sums.to(device)
        self.ema_sum_squares = self.ema_sum_squares.to(device)
        self.invalid_seen = self.invalid_seen.to(device)


class _ProfileState:
    def __init__(self, delta: _ProfileDelta, *, half_life_steps: float) -> None:
        self.schema = delta.schema
        self.half_life_steps = half_life_steps
        self.pending_counts = torch.zeros_like(delta.counts)
        self.pending_sums = torch.zeros_like(delta.sums)
        self.pending_sum_squares = torch.zeros_like(delta.sum_squares)
        self.pending_total = torch.zeros_like(delta.total)
        self.pending_invalid = torch.zeros_like(delta.invalid)
        self.last_counts = torch.zeros_like(delta.counts)
        self.last_sums = torch.zeros_like(delta.sums)
        self.last_sum_squares = torch.zeros_like(delta.sum_squares)
        self.last_total = torch.zeros_like(delta.total)
        self.last_invalid = torch.zeros_like(delta.invalid)
        self.ema_counts = torch.zeros_like(delta.counts)
        self.ema_sums = torch.zeros_like(delta.sums)
        self.ema_sum_squares = torch.zeros_like(delta.sum_squares)
        self.ema_total = torch.zeros_like(delta.total)
        self.invalid_seen = torch.zeros_like(delta.invalid)
        self.ema_step: int | None = None
        self.update(delta, half_life_steps=half_life_steps)

    def update(self, delta: _ProfileDelta, *, half_life_steps: float) -> None:
        if delta.schema != self.schema:
            raise ValueError("observation profile schema changed during training.")
        if half_life_steps != self.half_life_steps:
            raise ValueError("observation EMA half-life changed during training.")
        self._to(delta.counts.device)
        self.pending_counts.add_(delta.counts)
        self.pending_sums.add_(delta.sums)
        self.pending_sum_squares.add_(delta.sum_squares)
        self.pending_total.add_(delta.total)
        self.pending_invalid.add_(delta.invalid)

    def finalize(
        self,
        trainer: pl.Trainer,
        *,
        step: int,
        sync_distributed: bool,
    ) -> None:
        values = (
            self.pending_counts,
            self.pending_sums,
            self.pending_sum_squares,
            self.pending_total,
            self.pending_invalid,
        )
        reduced = _sum_many(trainer, values, sync_distributed)
        (
            self.last_counts,
            self.last_sums,
            self.last_sum_squares,
            self.last_total,
            self.last_invalid,
        ) = reduced
        self.invalid_seen = self.invalid_seen + self.last_invalid

        if self.ema_step is None:
            self.ema_counts.copy_(self.last_counts)
            self.ema_sums.copy_(self.last_sums)
            self.ema_sum_squares.copy_(self.last_sum_squares)
            self.ema_total.copy_(self.last_total)
        else:
            delta_steps = step - self.ema_step
            if delta_steps <= 0:
                raise RuntimeError("observation EMA step must move forwards.")
            decay = 2.0 ** (-delta_steps / self.half_life_steps)
            one_minus_decay = 1.0 - decay
            self.ema_counts.mul_(decay).add_(
                self.last_counts,
                alpha=one_minus_decay,
            )
            self.ema_sums.mul_(decay).add_(
                self.last_sums,
                alpha=one_minus_decay,
            )
            self.ema_sum_squares.mul_(decay).add_(
                self.last_sum_squares,
                alpha=one_minus_decay,
            )
            self.ema_total.mul_(decay).add_(
                self.last_total,
                alpha=one_minus_decay,
            )
        self.ema_step = step
        self.discard_pending()

    def current(self) -> _ProfileSnapshot:
        return _ProfileSnapshot(
            self.schema,
            self.last_counts,
            self.last_sums,
            self.last_sum_squares,
            self.last_total,
            self.last_invalid,
        )

    def ema(self) -> _ProfileSnapshot:
        return _ProfileSnapshot(
            self.schema,
            self.ema_counts,
            self.ema_sums,
            self.ema_sum_squares,
            self.ema_total,
            self.invalid_seen,
        )

    def discard_pending(self) -> None:
        self.pending_counts.zero_()
        self.pending_sums.zero_()
        self.pending_sum_squares.zero_()
        self.pending_total.zero_()
        self.pending_invalid.zero_()

    def state_dict(self) -> dict[str, Any]:
        if self.ema_step is None:
            raise RuntimeError("cannot save observation EMA before a completed step.")
        return {
            "schema": {"edges": self.schema.edges},
            "half_life_steps": self.half_life_steps,
            "ema_step": self.ema_step,
            "ema_counts": self.ema_counts.detach(),
            "ema_sums": self.ema_sums.detach(),
            "ema_sum_squares": self.ema_sum_squares.detach(),
            "ema_total": self.ema_total.detach(),
            "invalid_seen": self.invalid_seen.detach(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> _ProfileState:
        schema = _profile_schema_from_state(state.get("schema"))
        half_life_steps = _positive_real(
            state.get("half_life_steps"),
            "observation EMA half_life_steps",
        )
        ema_step = _non_negative_int(state.get("ema_step"), "observation EMA step")
        shape = (schema.bins,)
        ema_counts = _state_tensor(state, "ema_counts", shape, non_negative=True)
        ema_sums = _state_tensor(state, "ema_sums", shape)
        ema_sum_squares = _state_tensor(
            state,
            "ema_sum_squares",
            shape,
            non_negative=True,
        )
        ema_total = _state_tensor(state, "ema_total", (), non_negative=True)
        invalid_seen = _state_tensor(state, "invalid_seen", (), non_negative=True)

        value = cls.__new__(cls)
        value.schema = schema
        value.half_life_steps = half_life_steps
        value.pending_counts = torch.zeros_like(ema_counts)
        value.pending_sums = torch.zeros_like(ema_sums)
        value.pending_sum_squares = torch.zeros_like(ema_sum_squares)
        value.pending_total = torch.zeros_like(ema_total)
        value.pending_invalid = torch.zeros_like(invalid_seen)
        value.last_counts = torch.zeros_like(ema_counts)
        value.last_sums = torch.zeros_like(ema_sums)
        value.last_sum_squares = torch.zeros_like(ema_sum_squares)
        value.last_total = torch.zeros_like(ema_total)
        value.last_invalid = torch.zeros_like(invalid_seen)
        value.ema_counts = ema_counts
        value.ema_sums = ema_sums
        value.ema_sum_squares = ema_sum_squares
        value.ema_total = ema_total
        value.invalid_seen = invalid_seen
        value.ema_step = ema_step
        return value

    def _to(self, device: torch.device) -> None:
        if self.pending_counts.device == device:
            return
        self.pending_counts = self.pending_counts.to(device)
        self.pending_sums = self.pending_sums.to(device)
        self.pending_sum_squares = self.pending_sum_squares.to(device)
        self.pending_total = self.pending_total.to(device)
        self.pending_invalid = self.pending_invalid.to(device)
        self.last_counts = self.last_counts.to(device)
        self.last_sums = self.last_sums.to(device)
        self.last_sum_squares = self.last_sum_squares.to(device)
        self.last_total = self.last_total.to(device)
        self.last_invalid = self.last_invalid.to(device)
        self.ema_counts = self.ema_counts.to(device)
        self.ema_sums = self.ema_sums.to(device)
        self.ema_sum_squares = self.ema_sum_squares.to(device)
        self.ema_total = self.ema_total.to(device)
        self.invalid_seen = self.invalid_seen.to(device)


@dataclass(frozen=True)
class _StateOwner:
    stage: Literal["train", "validation"]
    context: str
    module_path: str
    spec_name: str
    item: str


@dataclass
class _StateEntry:
    state: _CurveState | _HeatmapState | _HistogramState | _ProfileState
    tag: str
    owner: _StateOwner


def _distribution_delta(value: Distribution) -> _HistogramDelta:
    samples = value.values.detach().float()
    edges = torch.tensor(value.edges, dtype=samples.dtype, device=samples.device)
    bins = len(value.edges) - 1
    internal_edges = edges[1:-1]
    indices = torch.bucketize(samples, internal_edges)
    valid = torch.isfinite(samples) & (samples >= edges[0]) & (samples <= edges[-1])
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
    counts = torch.zeros(bins, dtype=torch.float32, device=samples.device)
    counts.scatter_add_(0, safe_indices, valid.to(dtype=torch.float32))
    valid_values = torch.where(valid, samples, torch.zeros_like(samples))
    observations = valid.sum().to(dtype=torch.float32).reshape(1)
    total = samples.new_tensor(samples.numel(), dtype=torch.float32).reshape(1)
    schema = _HistogramSchema(
        kind="distribution",
        sizes=(bins,),
        group_names=("",),
        bucket_limits=(tuple(float(item) for item in value.edges[1:]),),
        ranges=((float(value.edges[0]), float(value.edges[-1])),),
    )
    return _HistogramDelta(
        schema=schema,
        mass=counts.reshape(1, bins),
        observations=observations,
        total=total,
        sums=valid_values.sum().reshape(1),
        sum_squares=valid_values.square().sum().reshape(1),
        invalid=(~valid).sum().to(dtype=torch.float32),
    )


def _profile_delta(value: Profile) -> _ProfileDelta:
    samples = value.values.detach().float()
    coordinates = value.coordinates.detach().float()
    edges = torch.tensor(value.edges, dtype=coordinates.dtype, device=coordinates.device)
    bins = len(value.edges) - 1
    indices = torch.bucketize(coordinates, edges[1:-1])
    valid = (
        torch.isfinite(samples)
        & torch.isfinite(coordinates)
        & (coordinates >= edges[0])
        & (coordinates <= edges[-1])
    )
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
    valid_values = torch.where(valid, samples, torch.zeros_like(samples))
    counts = torch.zeros(bins, dtype=torch.float32, device=samples.device)
    sums = torch.zeros_like(counts)
    sum_squares = torch.zeros_like(counts)
    weights = valid.to(dtype=torch.float32)
    counts.scatter_add_(0, safe_indices, weights)
    sums.scatter_add_(0, safe_indices, valid_values)
    sum_squares.scatter_add_(0, safe_indices, valid_values.square())
    return _ProfileDelta(
        schema=_ProfileSchema(tuple(float(item) for item in value.edges)),
        counts=counts,
        sums=sums,
        sum_squares=sum_squares,
        total=samples.new_tensor(samples.numel(), dtype=torch.float32),
        invalid=(~valid).sum().to(dtype=torch.float32),
    )


def _bins_delta(value: Bins) -> _HistogramDelta:
    raw = value.values.detach().long()
    sizes = tuple(value.sizes)
    groups = len(sizes)
    if value.group_dim is None:
        flat = raw.reshape(-1, 1)
        active = (
            flat != -1
            if value.active_mask is None
            else value.active_mask.detach().bool().reshape(-1, 1)
        )
        observed_groups = 1
    else:
        dim = value.group_dim % raw.ndim
        flat = raw.movedim(dim, -1).reshape(-1, raw.shape[dim])
        active = (
            flat != -1
            if value.active_mask is None
            else value.active_mask.detach().bool().movedim(dim, -1).reshape_as(flat)
        )
        observed_groups = flat.shape[1]

    max_bins = max(sizes)
    counts = torch.zeros(groups, max_bins, dtype=torch.float32, device=raw.device)
    observations = torch.zeros(groups, dtype=torch.float32, device=raw.device)
    total = torch.zeros(groups, dtype=torch.float32, device=raw.device)
    sums = torch.zeros(groups, dtype=torch.float32, device=raw.device)
    sum_squares = torch.zeros(groups, dtype=torch.float32, device=raw.device)
    invalid = torch.zeros((), dtype=torch.float32, device=raw.device)
    for group in range(observed_groups):
        group_values = flat[:, group]
        group_active = active[:, group]
        in_range = (group_values >= 0) & (group_values < sizes[group])
        valid = group_active & in_range
        invalid = invalid + (group_active & ~in_range).sum().to(dtype=torch.float32)
        safe = torch.where(valid, group_values, torch.zeros_like(group_values))
        counts[group].scatter_add_(0, safe, valid.to(dtype=torch.float32))
        valid_values = torch.where(valid, group_values, torch.zeros_like(group_values)).float()
        observations[group] = valid.sum()
        total[group] = group_values.numel()
        sums[group] = valid_values.sum()
        sum_squares[group] = valid_values.square().sum()

    group_names = value.group_names or tuple(
        "" if groups == 1 else f"group_{index}" for index in range(groups)
    )
    schema = _HistogramSchema(
        kind="bins",
        sizes=sizes,
        group_names=group_names,
        bucket_limits=tuple(
            tuple(index + 0.5 for index in range(size)) for size in sizes
        ),
        ranges=tuple((0.0, float(max(size - 1, 0))) for size in sizes),
    )
    return _HistogramDelta(
        schema=schema,
        mass=counts,
        observations=observations,
        total=total,
        sums=sums,
        sum_squares=sum_squares,
        invalid=invalid,
    )


def _heatmap_schema(value: Heatmap) -> _HeatmapSchema:
    rows, columns = value.values.shape
    value_range = (
        None
        if value.value_range is None
        else (float(value.value_range[0]), float(value.value_range[1]))
    )
    return _HeatmapSchema(
        shape=(rows, columns),
        dtype=str(value.values.dtype),
        value_range=value_range,
    )


def _heatmap_image(value: Heatmap) -> Tensor:
    values = value.values.detach().to(
        device=torch.device("cpu"),
        dtype=torch.float32,
        copy=True,
    )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("observation heatmap values must be finite.")
    if value.value_range is None:
        lower = values.min()
        upper = values.max()
    else:
        lower = values.new_tensor(value.value_range[0])
        upper = values.new_tensor(value.value_range[1])
    scale = upper - lower
    if bool((scale > 0).item()):
        return ((values - lower) / scale).clamp(0.0, 1.0)
    return torch.zeros_like(values)


def _sum(trainer: pl.Trainer, value: Tensor, enabled: bool) -> Tensor:
    if not enabled or not _capability.is_distributed(trainer):
        return value.detach().clone()
    return _capability.strategy_reduce(
        trainer,
        value,
        reduce_op="sum",
        error=(
            "Distributed module observation requires trainer.strategy.reduce; "
            "set sync_distributed=False for rank-local diagnostics."
        ),
    )


def _sum_many(
    trainer: pl.Trainer,
    values: tuple[Tensor, ...],
    enabled: bool,
) -> tuple[Tensor, ...]:
    """Reduce a collection of tensors with as few collectives as possible.

    Observation histogram/profile state keeps several dense accumulators whose
    shapes differ (for example, a ``(groups, bins)`` mass tensor and scalar
    invalid count).  Concatenating their flattened values lets one collective
    reduce the complete state, after which the original shapes are restored.
    Tensors with different dtype/device are reduced in separate groups so this
    helper never silently promotes or moves an accumulator.
    """
    if not values:
        return ()
    if not enabled or not _capability.is_distributed(trainer):
        # Match _sum's rank-local behavior exactly: detached, independent
        # tensors are returned even when distributed synchronization is off.
        return tuple(value.detach().clone() for value in values)

    # torch.cat requires a common dtype and device.  Grouping by both keeps
    # mixed accumulator states correct while still reducing the usual
    # homogeneous histogram/profile state in a single collective.
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[int, Tensor]]] = {}
    for index, value in enumerate(values):
        groups.setdefault((value.device, value.dtype), []).append((index, value))

    reduced_values: list[Tensor | None] = [None] * len(values)
    for grouped in groups.values():
        packed_parts = tuple(value.detach().reshape(-1) for _index, value in grouped)
        packed_size = sum(part.numel() for part in packed_parts)
        packed = torch.cat(packed_parts, dim=0)
        reduced = _sum(trainer, packed, enabled=True)
        if not isinstance(reduced, Tensor):
            raise TypeError("trainer.strategy.reduce must return a Tensor.")
        if reduced.numel() != packed_size:
            raise RuntimeError(
                "trainer.strategy.reduce returned a tensor with an incompatible "
                "number of elements for batched observation reduction."
            )
        reduced_flat = reduced.reshape(-1)
        offset = 0
        for index, value in grouped:
            size = value.numel()
            reduced_values[index] = reduced_flat.narrow(0, offset, size).reshape(value.shape)
            offset += size

    # Every input contributes one output, but keep a defensive check here so a
    # future grouping change cannot leak ``None`` into state finalization.
    if any(value is None for value in reduced_values):
        raise RuntimeError("batched observation reduction did not restore all tensors.")
    return tuple(cast(Tensor, value) for value in reduced_values)


def _schema_signature(
    states: Mapping[str, _StateEntry],
    *,
    emitted_keys: set[str],
    device: torch.device,
    observation_error: bool = False,
) -> Tensor:
    if observation_error:
        return torch.tensor(
            [0, 0, 1, 0, 0],
            dtype=torch.int64,
            device=device,
        )
    unknown = emitted_keys.difference(states)
    if unknown:
        raise RuntimeError(
            "observation emitted keys are missing runtime state: "
            + ", ".join(sorted(unknown))
        )
    schema: list[object] = []
    for key in sorted(states):
        entry = states[key]
        state = entry.state
        if isinstance(state, _CurveState):
            schema.append(
                (key, "curve", state.reduction.value, _owner_state(entry.owner))
            )
        elif isinstance(state, _HeatmapState):
            if key in emitted_keys and (
                not state.pending or state.pending_schema is None
            ):
                raise RuntimeError(
                    f"heatmap observation {key!r} was emitted without a pending value."
                )
            schema.append(
                (
                    key,
                    "heatmap",
                    _owner_state(entry.owner),
                    state.pending_schema if state.pending else state.schema,
                )
            )
        elif isinstance(state, _ProfileState):
            schema.append(
                (
                    key,
                    "profile",
                    _owner_state(entry.owner),
                    state.half_life_steps,
                    state.schema,
                )
            )
        else:
            schema.append(
                (
                    key,
                    "histogram",
                    _owner_state(entry.owner),
                    state.half_life_steps,
                    _schema_state(state.schema),
                )
            )
    payload = (tuple(schema), tuple(sorted(emitted_keys)))
    digest = hashlib.blake2b(repr(payload).encode(), digest_size=16).digest()
    mask = (1 << 63) - 1
    return torch.tensor(
        [
            len(schema),
            len(emitted_keys),
            0,
            int.from_bytes(digest[:8], "little") & mask,
            int.from_bytes(digest[8:], "little") & mask,
        ],
        dtype=torch.int64,
        device=device,
    )


def _owner_state(owner: _StateOwner) -> dict[str, str]:
    return {
        "stage": owner.stage,
        "context": owner.context,
        "module_path": owner.module_path,
        "spec_name": owner.spec_name,
        "item": owner.item,
    }


def _owner_from_state(value: object, *, legacy: bool = False) -> _StateOwner:
    if not isinstance(value, Mapping):
        raise TypeError("observation owner state must be a mapping.")
    required = {"module_path", "spec_name", "item"}
    if legacy:
        if not required.issubset(value):
            raise ValueError("observation owner state fields are invalid.")
        stage: Literal["train", "validation"] = "train"
        context = ""
    else:
        if set(value) != {*required, "stage", "context"}:
            raise ValueError("observation owner state fields are invalid.")
        raw_stage = value.get("stage")
        if raw_stage not in {"train", "validation"}:
            raise ValueError("observation owner stage is invalid.")
        stage = cast(Literal["train", "validation"], raw_stage)
        raw_context = value.get("context")
        if not isinstance(raw_context, str):
            raise TypeError("observation owner context must be a string.")
        context = raw_context
    if not legacy and context != context.strip("/"):
        raise ValueError("observation owner state fields are invalid.")
    module_path = value.get("module_path")
    if not isinstance(module_path, str):
        raise TypeError("observation owner module_path must be a string.")
    return _StateOwner(
        stage=stage,
        context=context,
        module_path=module_path,
        spec_name=validate_tag_part(
            value.get("spec_name"),
            "observation owner spec_name",
        ),
        item=validate_tag_part(value.get("item"), "observation owner item"),
    )


def _schema_state(schema: _HistogramSchema) -> dict[str, Any]:
    return {
        "kind": schema.kind,
        "sizes": schema.sizes,
        "group_names": schema.group_names,
        "bucket_limits": schema.bucket_limits,
        "ranges": schema.ranges,
    }


def _schema_from_state(value: object) -> _HistogramSchema:
    if not isinstance(value, Mapping):
        raise TypeError("observation histogram schema must be a mapping.")
    kind = value.get("kind")
    if kind not in {"bins", "distribution"}:
        raise ValueError("observation histogram schema kind is invalid.")
    sizes = _tuple_ints(value.get("sizes"), "observation histogram sizes")
    group_names = _tuple_strings(
        value.get("group_names"),
        "observation histogram group_names",
    )
    if len(group_names) != len(sizes):
        raise ValueError("observation histogram group_names must align with sizes.")
    normalized_names = tuple(name.strip("/") for name in group_names)
    if len(group_names) > 1 and any(not name for name in normalized_names):
        raise ValueError("grouped observation histogram names must be non-empty.")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("observation histogram group_names must be unique.")
    if kind == "distribution" and len(sizes) != 1:
        raise ValueError("distribution histogram state must contain exactly one group.")
    bucket_limits = _tuple_float_tuples(
        value.get("bucket_limits"),
        "observation histogram bucket_limits",
    )
    ranges = _tuple_float_tuples(value.get("ranges"), "observation histogram ranges")
    if len(bucket_limits) != len(sizes) or len(ranges) != len(sizes):
        raise ValueError("observation histogram schema group metadata must align.")
    for size, limits, bounds in zip(sizes, bucket_limits, ranges, strict=False):
        if len(limits) != size or len(bounds) != 2:
            raise ValueError("observation histogram schema bin metadata is invalid.")
        if any(right <= left for left, right in zip(limits, limits[1:], strict=False)):
            raise ValueError("observation histogram bucket limits must increase.")
        if bounds[1] < bounds[0]:
            raise ValueError("observation histogram ranges must be ordered.")
    return _HistogramSchema(
        kind=cast(Literal["bins", "distribution"], kind),
        sizes=sizes,
        group_names=group_names,
        bucket_limits=bucket_limits,
        ranges=cast(tuple[tuple[float, float], ...], ranges),
    )


def _profile_schema_from_state(value: object) -> _ProfileSchema:
    if not isinstance(value, Mapping):
        raise TypeError("observation profile schema must be a mapping.")
    edges = _tuple_floats(value.get("edges"), "observation profile edges")
    if len(edges) < 2:
        raise ValueError("observation profile edges must contain at least two values.")
    if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
        raise ValueError("observation profile edges must be strictly increasing.")
    return _ProfileSchema(edges)


def _state_tensor(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    *,
    non_negative: bool = False,
) -> Tensor:
    value = state.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"observation EMA state {name!r} must be a Tensor.")
    if tuple(value.shape) != shape:
        raise ValueError(f"observation EMA state {name!r} has incompatible shape.")
    result = value.detach().float().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"observation EMA state {name!r} must be finite.")
    if non_negative and bool((result < 0).any().item()):
        raise ValueError(f"observation EMA state {name!r} must be non-negative.")
    return result


def _tuple_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(_positive_int(item, name) for item in value)


def _tuple_strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence.")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain strings.")
    return tuple(cast(list[str] | tuple[str, ...], value))


def _tuple_float_tuples(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence.")
    result: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, (tuple, list)):
            raise TypeError(f"{name} entries must be sequences.")
        result.append(tuple(_real(item, name) for item in row))
    return tuple(result)


def _tuple_floats(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(_real(item, name) for item in value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_real(value: object, name: str) -> float:
    result = _real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _encode_history(states: Mapping[str, _StateEntry]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    histograms: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    owners: dict[str, dict[str, str]] = {}
    for key, entry in states.items():
        state = entry.state
        if not isinstance(state, (_HistogramState, _ProfileState)) or state.ema_step is None:
            continue
        if isinstance(state, _HistogramState):
            histograms[key] = state.state_dict()
        else:
            profiles[key] = state.state_dict()
        owners[key] = _owner_state(entry.owner)
    return histograms, profiles, owners


def _decode_history(
    state_dict: Mapping[str, Any], *, version: int, max_dense_bins: int,
    last_global_step: int | None, validation_step: int,
) -> dict[str, _StateEntry]:
    histograms = state_dict.get("histograms")
    if not isinstance(histograms, Mapping):
        raise TypeError("ObservationCallback histograms state must be a mapping.")
    profiles = {} if version == 2 else state_dict.get("profiles")
    if not isinstance(profiles, Mapping):
        raise TypeError("ObservationCallback profiles state must be a mapping.")
    owners = state_dict.get("owners")
    if not isinstance(owners, Mapping):
        raise TypeError("ObservationCallback owners state must be a mapping.")
    if set(histograms).intersection(profiles):
        raise ValueError("ObservationCallback history keys must be unique.")
    if set(histograms).union(profiles) != set(owners):
        raise ValueError(
            "ObservationCallback history and owner keys must match."
        )
    loaded: dict[str, _StateEntry] = {}
    for key, payload in histograms.items():
        if not isinstance(key, str) or not key:
            raise TypeError("ObservationCallback histogram keys must be strings.")
        if not isinstance(payload, Mapping):
            raise TypeError(f"ObservationCallback histogram {key!r} must be a mapping.")
        histogram = _HistogramState.from_state_dict(payload)
        if histogram.schema.dense_bins > max_dense_bins:
            raise ValueError(
                f"restored observation {key!r} requires "
                f"{histogram.schema.dense_bins} dense bins, exceeding "
                f"max_dense_bins={max_dense_bins}."
            )
        owner = _owner_from_state(owners[key], legacy=version < 4)
        expected_step = (
            last_global_step if owner.stage == "train" else validation_step
        )
        if expected_step is None or histogram.ema_step != expected_step:
            raise ValueError(
                f"restored observation {key!r} EMA step does not match callback step."
            )
        loaded[key] = _StateEntry(
            histogram,
            tag="",
            owner=owner,
        )
    for key, payload in profiles.items():
        if not isinstance(key, str) or not key:
            raise TypeError("ObservationCallback profile keys must be strings.")
        if not isinstance(payload, Mapping):
            raise TypeError(f"ObservationCallback profile {key!r} must be a mapping.")
        profile = _ProfileState.from_state_dict(payload)
        if profile.schema.bins > max_dense_bins:
            raise ValueError(
                f"restored observation {key!r} requires "
                f"{profile.schema.bins} dense bins, exceeding "
                f"max_dense_bins={max_dense_bins}."
            )
        owner = _owner_from_state(owners[key], legacy=version < 4)
        expected_step = (
            last_global_step if owner.stage == "train" else validation_step
        )
        if expected_step is None or profile.ema_step != expected_step:
            raise ValueError(
                f"restored observation {key!r} EMA step does not match callback step."
            )
        loaded[key] = _StateEntry(
            profile,
            tag="",
            owner=owner,
        )
    return loaded
