from __future__ import annotations

import hashlib
import math
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, NoReturn, cast
from weakref import ReferenceType, ref

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from anytrain import observation
from anytrain.observation import (
    BackwardObservation,
    BackwardObservationWarning,
    Bins,
    Curve,
    Distribution,
    ForwardEvent,
    Heatmap,
    Observation,
    ObservationResult,
    ObservationSpec,
    OutputObservation,
    Profile,
    Reduction,
    StateObservation,
)
from anytrain.observation._binding import claim, release, subscribe_event

from .. import _capability, experiment
from ._naming import join_tag, unique_tag_items, validate_tag_part


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


_SCHEMA_ERROR_INDEX = 2
_ObservationError = tuple[str, str]


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
        self.last_sum = _sum(trainer, self.pending_sum, sync_distributed)
        self.last_count = _sum(trainer, self.pending_count, sync_distributed)
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
        reduced = tuple(_sum(trainer, value, sync_distributed) for value in values)
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
        reduced = tuple(_sum(trainer, value, sync_distributed) for value in values)
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


@dataclass(frozen=True)
class _Collector:
    identifier: int
    module_path: str
    module: nn.Module
    spec: ObservationSpec

    @property
    def state_prefix(self) -> str:
        return join_tag(self.module_path or "<root>", self.spec.name)

    @property
    def tag_prefix(self) -> str:
        if not self.module_path:
            return self.spec.name
        return join_tag(self.module_path, self.spec.name)


class _ForwardCollectorHook:
    def __init__(
        self,
        callback: ObservationCallback,
        module: nn.Module,
        collector_id: int,
    ) -> None:
        self.callback: ReferenceType[ObservationCallback] = ref(callback)
        self.module_id = id(module)
        self.collector_id = collector_id

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: object,
    ) -> None:
        callback = self.callback()
        if callback is None or id(module) != self.module_id:
            _remove_observer_hooks(module, self.callback, self.collector_id)
            return
        collector = callback._collectors_by_id.get(self.collector_id)
        if collector is None:
            _remove_observer_hooks(module, self.callback, self.collector_id)
            return
        callback._forward(collector, module, inputs, kwargs, output)


class _ForwardEventSink:
    def __init__(
        self,
        callback: ObservationCallback,
        module: nn.Module,
        collector_id: int,
    ) -> None:
        self.callback: ReferenceType[ObservationCallback] = ref(callback)
        self.module_id = id(module)
        self.collector_id = collector_id

    def __call__(self, module: nn.Module, value: ObservationResult) -> bool:
        callback = self.callback()
        if callback is None or id(module) != self.module_id:
            return False
        collector = callback._collectors_by_id.get(self.collector_id)
        if collector is None:
            return False
        callback._event(collector, value)
        return True


class ObservationCallback(pl.Callback):
    """Discover registered module observations and log current/EMA statistics."""

    def __init__(
        self,
        *,
        every_n_steps: int | None = 100,
        tag: str = "module",
        half_life_steps: float | None = None,
        sync_distributed: bool = False,
        max_dense_bins: int = 1_000_000,
        max_heatmap_elements: int = 65_536,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        self.every_n_steps = every_n_steps
        self.tag = validate_tag_part(tag)
        self.half_life_steps = (
            None
            if half_life_steps is None
            else _positive_real(half_life_steps, "half_life_steps")
        )
        if not isinstance(sync_distributed, bool):
            raise TypeError("sync_distributed must be bool.")
        self.sync_distributed = sync_distributed
        self.max_dense_bins = _positive_int(max_dense_bins, "max_dense_bins")
        self.max_heatmap_elements = _positive_int(
            max_heatmap_elements,
            "max_heatmap_elements",
        )
        self._collectors: list[_Collector] = []
        self._collectors_by_id: dict[int, _Collector] = {}
        self._handles: list[RemovableHandle] = []
        self._claimed: list[nn.Module] = []
        self._states: dict[str, _StateEntry] = {}
        self._output_tag_owners: dict[str, str] = {}
        self._emitted_keys: set[str] = set()
        self._enabled = False
        self._warning_rank = False
        self._defer_observation_errors = False
        self._pending_observation_error: _ObservationError | None = None
        self._backward_warning_emitted = False
        self._forward_active = False
        self._render_active = False
        self._pending_observation_time = 0.0
        self._observation_time_sum = 0.0
        self._observation_time_steps = 0
        self._last_global_step: int | None = None
        self._validation_step = 0
        self._restored_last_global_step: int | None = None
        self._active_stage: Literal["train", "validation"] | None = None
        self.observed_modules: tuple[str, ...] = ()

    @property
    def state_key(self) -> str:
        return f"{type(self).__qualname__}[tag={self.tag}]"

    def _require_active_stage(self) -> Literal["train", "validation"]:
        if self._active_stage is None:
            raise RuntimeError("observation forward has no active Trainer stage.")
        return self._active_stage

    def _context(self, stage: Literal["train", "validation"]) -> str:
        prefix = join_tag(*observation.context_prefix())
        stage_prefix = "train" if stage == "train" else "val"
        if not prefix:
            return "" if stage == "train" else stage_prefix
        if prefix == stage_prefix or prefix.startswith(f"{stage_prefix}/"):
            return prefix
        return join_tag(stage_prefix, prefix)

    def _stage_keys(self, stage: Literal["train", "validation"]) -> tuple[str, ...]:
        return tuple(
            key for key, entry in self._states.items() if entry.owner.stage == stage
        )

    def _pending_heatmaps(
        self,
        stage: Literal["train", "validation"],
    ) -> tuple[_HeatmapState, ...]:
        pending: list[_HeatmapState] = []
        for key in self._stage_keys(stage):
            state = self._states[key].state
            if isinstance(state, _HeatmapState) and state.pending:
                pending.append(state)
        return tuple(pending)

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._enabled or self._collectors or self._claimed:
            return
        self._warning_rank = _capability.is_global_zero(trainer)
        self._defer_observation_errors = (
            self.sync_distributed and _capability.is_distributed(trainer)
        )
        self._enabled = self.sync_distributed or self._warning_rank
        if not self._enabled:
            return
        try:
            self._attach(pl_module)
            self._bind_restored_tags()
        except BaseException:
            self._detach()
            raise

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self._enabled:
            return
        step = _capability.global_step(trainer)
        restored = self._restored_last_global_step
        if restored is not None and restored != step:
            raise ValueError(
                "ObservationCallback checkpoint global step does not match Trainer: "
                f"got {restored} and {step}."
            )
        if restored is None:
            ema_steps: list[int] = []
            for entry in self._states.values():
                state = entry.state
                if (
                    entry.owner.stage == "train"
                    and isinstance(state, (_HistogramState, _ProfileState))
                    and state.ema_step is not None
                ):
                    ema_steps.append(state.ema_step)
            if ema_steps and step < max(ema_steps):
                raise ValueError(
                    "ObservationCallback cannot reuse history after Trainer global_step "
                    "was reset; create a fresh callback for a new training run."
                )
        self._last_global_step = step
        self._restored_last_global_step = None
        for entry in self._states.values():
            if not isinstance(entry.state, _HeatmapState):
                entry.state._to(pl_module.device)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        if not self._enabled:
            return
        self._active_stage = "train"
        self._forward_active = True
        step = _capability.global_step(trainer)
        # Manual optimization (for example GAN discriminator/generator
        # updates) can commit more than one optimizer step before batch end.
        # The final step is not known yet, so keep render-only collectors
        # active for the whole batch and decide whether to log at batch end.
        manual_optimization = not bool(
            getattr(pl_module, "automatic_optimization", True)
        )
        self._render_active = (
            manual_optimization and self.every_n_steps is not None
        ) or (
            self.every_n_steps is not None
            and (step + 1) % self.every_n_steps == 0
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not self._enabled:
            return
        started_at = self._start_observation_timer()
        self._forward_active = False
        step = _capability.global_step(trainer)
        previous = self._last_global_step
        if previous is None:
            self._last_global_step = step
            self._active_stage = None
            return
        optimizer_steps = step - previous
        if optimizer_steps < 0:
            raise RuntimeError("trainer.global_step moved backwards during observation.")
        if optimizer_steps == 0:
            self._add_observation_time(started_at)
            self._active_stage = None
            return
        if optimizer_steps != 1 and bool(
            getattr(pl_module, "automatic_optimization", True)
        ):
            raise RuntimeError(
                "ObservationCallback cannot partition one train batch across multiple "
                "optimizer steps."
            )
        self._poll("train")
        self._validate_distributed_schema(trainer, pl_module)
        for key in sorted(self._stage_keys("train")):
            state = self._states[key].state
            if isinstance(state, _CurveState):
                state.finalize(trainer, sync_distributed=self.sync_distributed)
            elif isinstance(state, (_HistogramState, _ProfileState)):
                state.finalize(
                    trainer,
                    step=step,
                    sync_distributed=self.sync_distributed,
                )
        self._last_global_step = step
        self._emitted_keys.clear()
        self._render_active = False
        self._active_stage = None
        _validate_histograms(self._states, device=pl_module.device)
        _validate_profiles(self._states, device=pl_module.device)
        if self.every_n_steps is None or step % self.every_n_steps != 0:
            self._finish_observation_step(started_at)
            return
        self._log(
            trainer,
            pl_module,
            stage="train",
            step=step,
            started_at=started_at,
            on_step=True,
            on_epoch=False,
        )

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if not self._enabled or bool(getattr(trainer, "sanity_checking", False)):
            return
        self._active_stage = "validation"
        self._forward_active = True
        self._render_active = True

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self._enabled or self._active_stage != "validation":
            return
        self._forward_active = False
        self._poll("validation")
        self._validate_distributed_schema(trainer, pl_module)
        self._validation_step += 1
        for key in sorted(self._stage_keys("validation")):
            state = self._states[key].state
            if isinstance(state, _CurveState):
                state.finalize(trainer, sync_distributed=self.sync_distributed)
            elif isinstance(state, (_HistogramState, _ProfileState)):
                state.finalize(
                    trainer,
                    step=self._validation_step,
                    sync_distributed=self.sync_distributed,
                )
        self._emitted_keys.clear()
        self._render_active = False
        self._active_stage = None
        _validate_histograms(self._states, device=pl_module.device)
        _validate_profiles(self._states, device=pl_module.device)
        self._log(
            trainer,
            pl_module,
            stage="validation",
            step=_capability.global_step(trainer),
            started_at=None,
            on_step=False,
            on_epoch=True,
        )

    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: Tensor,
    ) -> None:
        del trainer, pl_module, loss
        if not self._enabled:
            return
        # Activation checkpointing may replay module forward during backward.
        self._forward_active = False

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not self._enabled:
            return
        self._forward_active = self._active_stage == "train"

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not self._enabled:
            return
        self._discard_pending()

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not self._enabled:
            return
        self._discard_pending()
        self._detach()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        if not self._enabled:
            return
        self._discard_pending()
        self._detach()

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        del trainer, pl_module, stage
        if not self._enabled:
            return
        self._discard_pending()
        self._detach()

    def state_dict(self) -> dict[str, Any]:
        histograms: dict[str, dict[str, Any]] = {}
        profiles: dict[str, dict[str, Any]] = {}
        owners: dict[str, dict[str, str]] = {}
        for key, entry in self._states.items():
            state = entry.state
            if not isinstance(state, (_HistogramState, _ProfileState)) or state.ema_step is None:
                continue
            if isinstance(state, _HistogramState):
                histograms[key] = state.state_dict()
            else:
                profiles[key] = state.state_dict()
            owners[key] = _owner_state(entry.owner)
        return {
            "version": 4,
            "tag": self.tag,
            "every_n_steps": self.every_n_steps,
            "half_life_steps": self.half_life_steps,
            "sync_distributed": self.sync_distributed,
            "max_dense_bins": self.max_dense_bins,
            "max_heatmap_elements": self.max_heatmap_elements,
            "last_global_step": self._last_global_step,
            "validation_step": self._validation_step,
            "histograms": histograms,
            "profiles": profiles,
            "owners": owners,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        version = state_dict.get("version")
        if version not in {2, 3, 4}:
            raise ValueError("unsupported ObservationCallback state version.")
        if state_dict.get("tag") != self.tag:
            raise ValueError("ObservationCallback state tag mismatch.")
        expected = {
            "half_life_steps": self.half_life_steps,
            "sync_distributed": self.sync_distributed,
        }
        for name, value in expected.items():
            if state_dict.get(name) != value:
                raise ValueError(f"ObservationCallback state {name} mismatch.")
        last_global_step = state_dict.get("last_global_step")
        if last_global_step is not None:
            last_global_step = _non_negative_int(
                last_global_step,
                "ObservationCallback last_global_step",
            )
        validation_step = (
            0
            if version < 4
            else _non_negative_int(
                state_dict.get("validation_step"),
                "ObservationCallback validation_step",
            )
        )
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
            if histogram.schema.dense_bins > self.max_dense_bins:
                raise ValueError(
                    f"restored observation {key!r} requires "
                    f"{histogram.schema.dense_bins} dense bins, exceeding "
                    f"max_dense_bins={self.max_dense_bins}."
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
            if profile.schema.bins > self.max_dense_bins:
                raise ValueError(
                    f"restored observation {key!r} requires "
                    f"{profile.schema.bins} dense bins, exceeding "
                    f"max_dense_bins={self.max_dense_bins}."
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
        self._states = loaded
        self._validation_step = validation_step
        self._output_tag_owners.clear()
        self._emitted_keys.clear()
        self._restored_last_global_step = last_global_step
        if self._collectors:
            self._bind_restored_tags()

    def _attach(self, pl_module: pl.LightningModule) -> None:
        registered: list[tuple[str, nn.Module, tuple[ObservationSpec, ...]]] = []
        for name, module in pl_module.named_modules():
            declared = observation.registry.specs(module)
            if declared:
                registered.append((name, module, declared))
        for _name, module, declared in registered:
            claim(module, declared)
            self._claimed.append(module)

        covered: list[str] = []
        observed: list[str] = []
        for module_path, module, specs in registered:
            if any(_is_descendant(module_path, parent) for parent in covered):
                continue
            supported: list[ObservationSpec] = []
            for spec in specs:
                if isinstance(spec, BackwardObservation):
                    self._warn_backward_observation()
                else:
                    supported.append(spec)
            if not supported:
                continue
            observed.append(module_path or "module")
            for spec in supported:
                collector = _Collector(len(self._collectors), module_path, module, spec)
                self._collectors.append(collector)
                self._collectors_by_id[collector.identifier] = collector
                self._attach_collector(collector)
            if any(spec.covers_descendants for spec in supported):
                covered.append(module_path)
        self.observed_modules = tuple(observed)

    def _warn_backward_observation(self) -> None:
        if not self._warning_rank or self._backward_warning_emitted:
            return
        self._backward_warning_emitted = True
        warnings.warn(
            "BackwardObservation is reserved but is not implemented by "
            "ObservationCallback yet. Use "
            "anytrain.lightning.GradientProbeLoggerCallback to observe loss "
            "gradients with respect to selected parameters.",
            BackwardObservationWarning,
            stacklevel=5,
        )

    def _attach_collector(self, collector: _Collector) -> None:
        spec = collector.spec
        if isinstance(spec, OutputObservation):
            handle = collector.module.register_forward_hook(
                _ForwardCollectorHook(self, collector.module, collector.identifier),
                with_kwargs=True,
            )
            self._handles.append(handle)
        elif isinstance(spec, ForwardEvent):
            self._handles.append(
                subscribe_event(
                    collector.module,
                    spec,
                    _ForwardEventSink(
                        self,
                        collector.module,
                        collector.identifier,
                    ),
                )
            )

    def _forward(
        self,
        collector: _Collector,
        module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: object,
    ) -> None:
        if not self._forward_active:
            return
        stage = self._require_active_stage()
        spec = cast(OutputObservation, collector.spec)
        if spec.history is None and not self._render_active:
            return
        started_at = self._start_observation_timer()
        try:
            self._collect_and_record(
                collector,
                lambda: spec.collect(module, inputs, kwargs, output),
                stage=stage,
            )
        finally:
            self._add_observation_time(started_at)

    def _event(
        self,
        collector: _Collector,
        value: ObservationResult,
    ) -> None:
        if not self._forward_active:
            return
        stage = self._require_active_stage()
        spec = cast(ForwardEvent, collector.spec)
        if spec.history is None and not self._render_active:
            return
        started_at = self._start_observation_timer()
        try:
            self._collect_and_record(collector, lambda: value, stage=stage)
        finally:
            self._add_observation_time(started_at)

    def _poll(self, stage: Literal["train", "validation"]) -> None:
        for collector in self._collectors:
            spec = collector.spec
            if isinstance(spec, StateObservation):
                if spec.history is None and not self._render_active:
                    continue
                self._collect_and_record(
                    collector,
                    lambda spec=spec, module=collector.module: spec.collect(module),
                    stage=stage,
                )

    def _collect_and_record(
        self,
        collector: _Collector,
        collect: Callable[[], ObservationResult],
        *,
        stage: Literal["train", "validation"],
    ) -> None:
        if self._pending_observation_error is not None:
            return
        try:
            self._record(collector, collect(), stage=stage)
        except Exception as error:
            if not self._defer_observation_errors:
                raise
            self._pending_observation_error = (
                "runtime",
                f"observation collector {collector.state_prefix!r} failed: "
                f"{type(error).__name__}: {error}",
            )

    def _record(
        self,
        collector: _Collector,
        result: ObservationResult,
        *,
        stage: Literal["train", "validation"],
    ) -> None:
        if result is None:
            return
        if isinstance(result, (Curve, Distribution, Profile, Bins, Heatmap)):
            values: Mapping[str, Observation] = {"value": result}
        elif isinstance(result, Mapping):
            values = result
        else:
            raise TypeError("observation collector must return an Observation, mapping, or None.")
        for item, value in unique_tag_items(
            values.items(),
            name="observation value name",
        ):
            if not isinstance(value, (Curve, Distribution, Profile, Bins, Heatmap)):
                raise TypeError("observation mapping values must be Observation objects.")
            if isinstance(value, Heatmap):
                if collector.spec.history is not None:
                    raise ValueError(
                        "Heatmap observations do not support EMAHistory; declare a "
                        "separate no-history observation."
                    )
                if not self._render_active:
                    continue
            context = self._context(stage)
            key = join_tag(context, collector.state_prefix, item)
            tag = join_tag(context, self.tag, collector.tag_prefix, item)
            self._update_state(
                key,
                tag,
                value,
                collector.spec,
                owner=_collector_owner(
                    collector,
                    item,
                    stage=stage,
                    context=context,
                ),
            )

    def _update_state(
        self,
        key: str,
        tag: str,
        value: Observation,
        spec: ObservationSpec,
        *,
        owner: _StateOwner,
    ) -> None:
        entry = self._states.get(key)
        reduction = _observation_reduction(spec, value)
        if isinstance(value, Curve):
            if entry is None:
                state = _CurveState(value, reduction=reduction)
                self._bind_state_identity(key, tag, state, owner)
                self._states[key] = _StateEntry(state, tag, owner)
                self._emitted_keys.add(key)
                return
            if not isinstance(entry.state, _CurveState):
                raise TypeError(f"observation {key!r} changed value type.")
            self._bind_state_identity(key, tag, entry.state, owner)
            entry.state.update(value, reduction=reduction)
            self._emitted_keys.add(key)
            return

        if isinstance(value, Heatmap):
            if entry is None:
                state = _HeatmapState(
                    value,
                    keep_image=self._warning_rank,
                    max_elements=self.max_heatmap_elements,
                )
                self._bind_state_identity(key, tag, state, owner)
                self._states[key] = _StateEntry(state, tag, owner)
                self._emitted_keys.add(key)
                return
            if not isinstance(entry.state, _HeatmapState):
                raise TypeError(f"observation {key!r} changed value type.")
            self._bind_state_identity(key, tag, entry.state, owner)
            entry.state.update(
                value,
                keep_image=self._warning_rank,
                max_elements=self.max_heatmap_elements,
            )
            self._emitted_keys.add(key)
            return

        history = spec.history
        if history is None:
            raise ValueError(
                f"observation {key!r} requires EMAHistory for Distribution, Profile, or Bins."
            )
        half_life_steps = (
            float(history.half_life_steps)
            if self.half_life_steps is None
            else self.half_life_steps
        )
        dense_bins = (
            len(value.sizes) * max(value.sizes)
            if isinstance(value, Bins)
            else len(value.edges) - 1
        )
        if dense_bins > self.max_dense_bins:
            raise ValueError(
                f"observation {key!r} requires {dense_bins} dense bins, exceeding "
                f"max_dense_bins={self.max_dense_bins}; use a factorized observation or raise "
                "the limit explicitly."
            )
        if isinstance(value, Profile):
            delta = _profile_delta(value)
            if entry is None:
                state = _ProfileState(delta, half_life_steps=half_life_steps)
                self._bind_state_identity(key, tag, state, owner)
                self._states[key] = _StateEntry(state, tag, owner)
                self._emitted_keys.add(key)
                return
            if not isinstance(entry.state, _ProfileState):
                raise TypeError(f"observation {key!r} changed value type.")
            self._bind_state_identity(key, tag, entry.state, owner)
            entry.state.update(delta, half_life_steps=half_life_steps)
            self._emitted_keys.add(key)
            return

        delta = (
            _distribution_delta(value)
            if isinstance(value, Distribution)
            else _bins_delta(value)
        )
        if entry is None:
            state = _HistogramState(delta, half_life_steps=half_life_steps)
            self._bind_state_identity(key, tag, state, owner)
            self._states[key] = _StateEntry(
                state,
                tag,
                owner,
            )
            self._emitted_keys.add(key)
            return
        if not isinstance(entry.state, _HistogramState):
            raise TypeError(f"observation {key!r} changed value type.")
        self._bind_state_identity(key, tag, entry.state, owner)
        entry.state.update(delta, half_life_steps=half_life_steps)
        self._emitted_keys.add(key)

    def _bind_state_identity(
        self,
        key: str,
        tag: str,
        state: _CurveState | _HeatmapState | _HistogramState | _ProfileState,
        owner: _StateOwner,
    ) -> None:
        entry = self._states.get(key)
        if entry is not None and entry.owner != owner:
            raise ValueError(
                f"observation sources {entry.owner!r} and {owner!r} resolve to the "
                f"same state key {key!r}."
            )
        if entry is not None and entry.tag and entry.tag != tag:
            raise ValueError(
                f"observation {key!r} changed logger tag from {entry.tag!r} to {tag!r}."
            )
        output_tags = _state_output_tags(tag, state)
        for output_tag in output_tags:
            previous_key = self._output_tag_owners.get(output_tag)
            if previous_key is not None and previous_key != key:
                raise ValueError(
                    f"observations {previous_key!r} and {key!r} resolve to the same "
                    f"logger tag {output_tag!r}."
                )
        for output_tag in output_tags:
            self._output_tag_owners[output_tag] = key

    def _log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        stage: Literal["train", "validation"],
        step: int,
        started_at: float | None,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        metrics: dict[str, Tensor] = {}
        stage_keys = self._stage_keys(stage)
        has_pending_heatmap = bool(self._pending_heatmaps(stage))
        image_writer: experiment.Image | None = None
        image_error: tuple[Literal["runtime"], str] | None = None
        if has_pending_heatmap and _capability.is_global_zero(trainer):
            try:
                image_writer = experiment.image(trainer)
            except Exception as error:
                image_error = (
                    "runtime",
                    "heatmap image writer discovery failed: "
                    f"{type(error).__name__}: {error}",
                )
        self._validate_heatmap_logging(
            trainer,
            pl_module,
            stage=stage,
            image_writer=image_writer,
            image_error=image_error,
        )
        write_error = self._write_heatmaps(
            trainer,
            stage=stage,
            image_writer=image_writer,
            step=step,
        )
        self._validate_heatmap_write(
            trainer,
            pl_module,
            pending=has_pending_heatmap,
            error=write_error,
        )
        raw_writer: experiment.HistogramRaw | None = None
        raw_error: _ObservationError | None = None
        has_raw_histogram = any(
            isinstance(self._states[key].state, _HistogramState)
            and bool(self._states[key].tag)
            for key in stage_keys
        )
        if has_raw_histogram and _capability.is_global_zero(trainer):
            try:
                raw_writer = experiment.histogram_raw(trainer)
            except Exception as error:
                raw_error = (
                    "runtime",
                    "raw histogram writer discovery failed: "
                    f"{type(error).__name__}: {error}",
                )
        write_error = raw_error
        for key in sorted(stage_keys):
            entry = self._states[key]
            if not entry.tag:
                continue
            state = entry.state
            if isinstance(state, _CurveState):
                value = state.value()
                if value is not None:
                    _merge_metrics(metrics, {entry.tag: value})
                continue
            if isinstance(state, _HeatmapState):
                continue
            if isinstance(state, _ProfileState):
                _merge_metrics(
                    metrics,
                    _profile_metrics(entry.tag, "step", state.current()),
                )
                _merge_metrics(
                    metrics,
                    _profile_metrics(entry.tag, "ema", state.ema()),
                )
                continue
            current = state.current()
            ema = state.ema()
            _merge_metrics(metrics, _histogram_metrics(entry.tag, "step", current))
            _merge_metrics(metrics, _histogram_metrics(entry.tag, "ema", ema))
            if raw_writer is not None and write_error is None:
                try:
                    _write_histogram(raw_writer, entry.tag, "step", current, step=step)
                    _write_histogram(raw_writer, entry.tag, "ema", ema, step=step)
                except Exception as error:
                    write_error = (
                        "runtime",
                        f"raw histogram write failed for {entry.tag!r}: "
                        f"{type(error).__name__}: {error}",
                    )
        self._synchronize_runtime_error(
            trainer,
            pl_module,
            pending=has_raw_histogram,
            error=write_error,
            context="raw histogram logging",
            collective_error=(
                "Distributed raw histogram logging requires "
                "trainer.strategy.all_gather; set sync_distributed=False for "
                "rank-local diagnostics."
            ),
        )
        if stage == "train":
            self._finish_observation_step(started_at)
        if stage == "train" and self._observation_time_steps > 0:
            _merge_metrics(
                metrics,
                {
                    join_tag(self.tag, "observation_time"): torch.tensor(
                        self._observation_time_sum / self._observation_time_steps,
                        dtype=torch.float32,
                        device=pl_module.device,
                    )
                },
            )
            self._observation_time_sum = 0.0
            self._observation_time_steps = 0
        if metrics:
            pl_module.log_dict(
                metrics,
                on_step=on_step,
                on_epoch=on_epoch,
                logger=True,
                sync_dist=False,
            )

    def _write_heatmaps(
        self,
        trainer: pl.Trainer,
        *,
        stage: Literal["train", "validation"],
        image_writer: experiment.Image | None,
        step: int,
    ) -> tuple[Literal["runtime"], str] | None:
        error: tuple[Literal["runtime"], str] | None = None
        for key in sorted(self._stage_keys(stage)):
            entry = self._states[key]
            state = entry.state
            if not isinstance(state, _HeatmapState) or not state.pending:
                continue
            if _capability.is_global_zero(trainer) and error is None:
                try:
                    value = state.value()
                    if value is None:
                        raise RuntimeError(
                            "global-zero heatmap observation did not retain an image."
                        )
                    if image_writer is None:
                        raise RuntimeError(
                            "heatmap image writer preflight was inconsistent."
                        )
                    _write_heatmap(
                        image_writer,
                        entry.tag,
                        value,
                        step=step,
                    )
                except Exception as caught:
                    error = (
                        "runtime",
                        f"heatmap image write failed for {entry.tag!r}: "
                        f"{type(caught).__name__}: {caught}",
                    )
            state.discard_pending()
        return error

    def _bind_restored_tags(self) -> None:
        self._output_tag_owners.clear()
        collectors = {
            (collector.module_path, collector.spec.name.strip("/")): collector
            for collector in self._collectors
        }
        for key, entry in self._states.items():
            owner = entry.owner
            collector = collectors.get((owner.module_path, owner.spec_name))
            if collector is None:
                raise ValueError(
                    f"restored observation {key!r} does not match the current module tree."
                )
            expected_key = join_tag(owner.context, collector.state_prefix, owner.item)
            if expected_key != key:
                raise ValueError(
                    f"restored observation {key!r} has incompatible source identity."
                )
            if isinstance(entry.state, (_HistogramState, _ProfileState)):
                history = collector.spec.history
                if history is None:
                    raise ValueError(
                        f"restored observation {key!r} no longer declares EMA history."
                    )
                half_life_steps = (
                    float(history.half_life_steps)
                    if self.half_life_steps is None
                    else self.half_life_steps
                )
                if entry.state.half_life_steps != half_life_steps:
                    raise ValueError(
                        f"restored observation {key!r} EMA half-life changed."
                    )
            tag = join_tag(owner.context, self.tag, collector.tag_prefix, owner.item)
            self._bind_state_identity(
                key,
                tag,
                entry.state,
                owner,
            )
            entry.tag = tag

    def _validate_distributed_schema(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.sync_distributed or not _capability.is_distributed(trainer):
            return
        signature, local_error = self._distributed_schema_signature(pl_module)
        gathered = _all_gather(
            trainer,
            signature,
            error=(
                "Distributed module observation requires trainer.strategy.all_gather; "
                "set sync_distributed=False for rank-local diagnostics."
            ),
        )
        if gathered.numel() % signature.numel() != 0:
            raise RuntimeError(
                "trainer.strategy.all_gather returned an invalid observation schema payload."
            )
        rows = gathered.detach().cpu().reshape(-1, signature.numel())
        if bool((rows[:, _SCHEMA_ERROR_INDEX] > 0).any().item()):
            _raise_synchronized_error(
                local_error,
                context="module observation",
            )
        expected = signature.detach().cpu()
        expected[_SCHEMA_ERROR_INDEX] = 0
        rows[:, _SCHEMA_ERROR_INDEX] = 0
        if any(not torch.equal(row, expected) for row in rows):
            raise RuntimeError(
                "sync_distributed=True requires every rank to emit the same observation "
                "keys and schemas for each optimizer step."
            )

    def _distributed_schema_signature(
        self,
        pl_module: pl.LightningModule,
    ) -> tuple[Tensor, _ObservationError | None]:
        local_error = self._pending_observation_error
        try:
            signature = _schema_signature(
                self._states,
                emitted_keys=self._emitted_keys,
                device=pl_module.device,
                observation_error=local_error is not None,
            )
        except Exception as error:
            if not self._defer_observation_errors:
                raise
            local_error = (
                "runtime",
                "observation schema materialization failed: "
                f"{type(error).__name__}: {error}",
            )
            signature = _schema_signature(
                {},
                emitted_keys=set(),
                device=pl_module.device,
                observation_error=True,
            )
        return signature, local_error

    def _validate_heatmap_logging(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        stage: Literal["train", "validation"],
        image_writer: experiment.Image | None,
        image_error: tuple[Literal["runtime"], str] | None,
    ) -> None:
        pending = self._pending_heatmaps(stage)
        if not pending:
            return
        errors = [state.pending_error for state in pending if state.pending_error]
        if image_error is not None:
            errors.append(image_error)
        elif self._warning_rank and image_writer is None:
            errors.append(
                (
                    "runtime",
                    "Heatmap observation requires the Trainer logger experiment "
                    "to provide add_image().",
                )
            )
        if self._warning_rank and any(
            state.pending_image is None and state.pending_error is None
            for state in pending
        ):
            errors.append(
                (
                    "runtime",
                    "global-zero heatmap observation did not retain an image.",
                )
            )

        if self.sync_distributed and _capability.is_distributed(trainer):
            local_error = torch.tensor(
                [int(bool(errors))],
                dtype=torch.int64,
                device=pl_module.device,
            )
            gathered = _all_gather(
                trainer,
                local_error,
                error=(
                    "Distributed Heatmap observation requires "
                    "trainer.strategy.all_gather; set sync_distributed=False for "
                    "rank-local diagnostics."
                ),
            )
            if bool((gathered.detach() > 0).any().item()):
                if errors:
                    detail = "; ".join(message for _kind, message in errors)
                    raise RuntimeError(
                        "distributed Heatmap observation failed: " + detail
                    )
                raise RuntimeError(
                    "distributed Heatmap observation failed on another rank."
                )
            return

        if not errors:
            return
        kind, message = errors[0]
        if kind == "value":
            raise ValueError(message)
        raise RuntimeError(message)

    def _validate_heatmap_write(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        pending: bool,
        error: tuple[Literal["runtime"], str] | None,
    ) -> None:
        if not pending:
            return
        self._synchronize_runtime_error(
            trainer,
            pl_module,
            pending=pending,
            error=error,
            context="Heatmap image logging",
            collective_error=(
                "Distributed Heatmap image logging requires "
                "trainer.strategy.all_gather; set sync_distributed=False for "
                "rank-local diagnostics."
            ),
        )

    def _synchronize_runtime_error(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        pending: bool,
        error: _ObservationError | None,
        context: str,
        collective_error: str,
    ) -> None:
        if not pending:
            return
        if self.sync_distributed and _capability.is_distributed(trainer):
            local_error = torch.tensor(
                [int(error is not None)],
                dtype=torch.int64,
                device=pl_module.device,
            )
            gathered = _all_gather(trainer, local_error, error=collective_error)
            if bool((gathered.detach() > 0).any().item()):
                _raise_synchronized_error(error, context=context)
            return
        if error is not None:
            raise RuntimeError(f"{context} failed: {error[1]}")

    def _discard_pending(self) -> None:
        self._active_stage = None
        self._forward_active = False
        self._render_active = False
        self._pending_observation_time = 0.0
        self._emitted_keys.clear()
        self._pending_observation_error = None
        for entry in self._states.values():
            entry.state.discard_pending()

    def _detach(self) -> None:
        self._enabled = False
        self._warning_rank = False
        self._defer_observation_errors = False
        self._pending_observation_error = None
        self._active_stage = None
        self._forward_active = False
        self._render_active = False
        self._pending_observation_time = 0.0
        self._observation_time_sum = 0.0
        self._observation_time_steps = 0
        while self._handles:
            self._handles.pop().remove()
        while self._claimed:
            release(self._claimed.pop())
        self._collectors.clear()
        self._collectors_by_id.clear()
        self._output_tag_owners.clear()
        self._emitted_keys.clear()

    def _start_observation_timer(self) -> float | None:
        if (
            self._active_stage != "train"
            or self.every_n_steps is None
            or not self._collectors
        ):
            return None
        return time.perf_counter()

    def _add_observation_time(self, started_at: float | None) -> None:
        if started_at is None:
            return
        self._pending_observation_time += time.perf_counter() - started_at

    def _finish_observation_step(self, started_at: float | None) -> None:
        self._add_observation_time(started_at)
        if self.every_n_steps is None or not self._collectors:
            return
        self._observation_time_sum += self._pending_observation_time
        self._observation_time_steps += 1
        self._pending_observation_time = 0.0


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


def _state_output_tags(
    tag: str,
    state: _CurveState | _HeatmapState | _HistogramState | _ProfileState,
) -> tuple[str, ...]:
    if isinstance(state, (_CurveState, _HeatmapState)):
        return (tag,)
    if isinstance(state, _ProfileState):
        profile_tags: list[str] = []
        for scope in ("step", "ema"):
            scope_tag = join_tag(tag, scope)
            profile_tags.extend(
                (
                    join_tag(scope_tag, "observations"),
                    join_tag(scope_tag, "active_ratio"),
                )
            )
            for index in range(state.schema.bins):
                bin_tag = join_tag(scope_tag, f"bin_{index}")
                profile_tags.extend(
                    (
                        join_tag(bin_tag, "count"),
                        join_tag(bin_tag, "mean"),
                        join_tag(bin_tag, "std"),
                    )
                )
        return tuple(profile_tags)
    result: list[str] = []
    for group_name in state.schema.group_names:
        for scope in ("step", "ema"):
            scope_tag = join_tag(tag, group_name, scope)
            result.append(scope_tag)
            metrics = [
                "observations",
                "active_ratio",
                "entropy",
                "perplexity",
            ]
            if state.schema.kind == "bins":
                metrics.extend(
                    (
                        "occupied_count",
                        "occupied_ratio",
                        "dead_count",
                        "dead_ratio",
                    )
                )
            result.extend(join_tag(scope_tag, metric) for metric in metrics)
    if len(set(result)) != len(result):
        raise RuntimeError("observation state produced duplicate logger tags.")
    return tuple(result)


def _observation_reduction(
    spec: ObservationSpec,
    value: Observation,
) -> Reduction:
    if isinstance(spec, (OutputObservation, ForwardEvent)):
        reduction = spec.reduction
    elif isinstance(value, (Curve, Heatmap)):
        reduction = Reduction.Last
    else:
        reduction = Reduction.Merge

    if reduction == Reduction.Merge and spec.history is None:
        raise ValueError("Merge observation reduction requires EMAHistory.")
    if reduction != Reduction.Merge and spec.history is not None:
        raise ValueError("only Merge observation reduction supports EMAHistory.")

    if reduction == Reduction.Last and isinstance(value, (Curve, Heatmap)):
        return reduction
    if reduction == Reduction.Mean and isinstance(value, Curve):
        return reduction
    if reduction == Reduction.Merge and isinstance(
        value,
        (Distribution, Profile, Bins),
    ):
        return reduction
    raise ValueError(
        f"observation reduction {reduction.value!r} does not support "
        f"{type(value).__name__}."
    )


def _merge_metrics(target: dict[str, Tensor], values: Mapping[str, Tensor]) -> None:
    collisions = sorted(set(target).intersection(values))
    if collisions:
        raise RuntimeError(
            "observation metrics produced duplicate logger tags: "
            + ", ".join(repr(tag) for tag in collisions)
        )
    target.update(values)


def _histogram_metrics(
    tag: str,
    scope: str,
    snapshot: _HistogramSnapshot,
) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    tiny = torch.finfo(snapshot.mass.dtype).tiny
    for group, size in enumerate(snapshot.schema.sizes):
        group_tag = join_tag(tag, snapshot.schema.group_names[group], scope)
        observations = snapshot.observations[group]
        total = snapshot.total[group]
        mass = snapshot.mass[group, :size]
        probabilities = torch.where(
            observations > 0,
            mass / observations.clamp_min(tiny),
            torch.zeros_like(mass),
        )
        safe = probabilities.clamp_min(tiny)
        entropy = -torch.where(
            probabilities > 0,
            probabilities * safe.log(),
            torch.zeros_like(probabilities),
        ).sum()
        values[join_tag(group_tag, "observations")] = observations
        values[join_tag(group_tag, "active_ratio")] = torch.where(
            total > 0,
            observations / total.clamp_min(tiny),
            torch.zeros_like(observations),
        )
        values[join_tag(group_tag, "entropy")] = entropy
        values[join_tag(group_tag, "perplexity")] = torch.where(
            observations > 0,
            entropy.exp(),
            torch.zeros_like(entropy),
        )
        if snapshot.schema.kind == "bins":
            denominator = mass.new_tensor(float(size))
            occupied = (mass > 0).sum().to(dtype=mass.dtype)
            dead = denominator - occupied
            values[join_tag(group_tag, "occupied_count")] = occupied
            values[join_tag(group_tag, "occupied_ratio")] = occupied / denominator
            values[join_tag(group_tag, "dead_count")] = dead
            values[join_tag(group_tag, "dead_ratio")] = dead / denominator
    return values


def _profile_metrics(
    tag: str,
    scope: str,
    snapshot: _ProfileSnapshot,
) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    scope_tag = join_tag(tag, scope)
    observations = snapshot.counts.sum()
    tiny = torch.finfo(snapshot.counts.dtype).tiny
    values[join_tag(scope_tag, "observations")] = observations
    values[join_tag(scope_tag, "active_ratio")] = torch.where(
        snapshot.total > 0,
        observations / snapshot.total.clamp_min(tiny),
        torch.zeros_like(observations),
    )
    for index in range(snapshot.schema.bins):
        bin_tag = join_tag(scope_tag, f"bin_{index}")
        count = snapshot.counts[index]
        mean = torch.where(
            count > 0,
            snapshot.sums[index] / count.clamp_min(tiny),
            torch.zeros_like(count),
        )
        variance = torch.where(
            count > 0,
            snapshot.sum_squares[index] / count.clamp_min(tiny) - mean.square(),
            torch.zeros_like(count),
        ).clamp_min(0)
        values[join_tag(bin_tag, "count")] = count
        values[join_tag(bin_tag, "mean")] = mean
        values[join_tag(bin_tag, "std")] = variance.sqrt()
    return values


def _validate_histograms(
    states: Mapping[str, _StateEntry],
    *,
    device: torch.device,
) -> None:
    entries: list[tuple[str, _HistogramState]] = []
    for entry in states.values():
        state = entry.state
        if isinstance(state, _HistogramState):
            entries.append((entry.tag, state))
    if not entries:
        return
    total = torch.stack(
        [state.last_invalid.to(device) for _tag, state in entries]
    ).sum()
    if not bool((total > 0).item()):
        return
    for tag, state in entries:
        invalid = float(state.last_invalid.detach().cpu().item())
        if invalid > 0:
            raise ValueError(
                f"observation {tag!r} contained {invalid:g} non-finite, "
                "out-of-range, or invalid active values."
            )
    raise RuntimeError("observation invalid-count aggregation was inconsistent.")


def _validate_profiles(
    states: Mapping[str, _StateEntry],
    *,
    device: torch.device,
) -> None:
    entries: list[tuple[str, _ProfileState]] = []
    for entry in states.values():
        state = entry.state
        if isinstance(state, _ProfileState):
            entries.append((entry.tag, state))
    if not entries:
        return
    total = torch.stack(
        [state.last_invalid.to(device) for _tag, state in entries]
    ).sum()
    if not bool((total > 0).item()):
        return
    for tag, state in entries:
        invalid = float(state.last_invalid.detach().cpu().item())
        if invalid > 0:
            raise ValueError(
                f"observation {tag!r} contained {invalid:g} non-finite or "
                "out-of-range profile pairs."
            )
    raise RuntimeError("observation profile invalid-count aggregation was inconsistent.")


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


def _write_heatmap(
    writer: experiment.Image,
    tag: str,
    image: Tensor,
    *,
    step: int,
) -> None:
    writer.add_image(
        tag,
        image,
        step,
        dataformats="HW",
    )


def _write_histogram(
    writer: experiment.HistogramRaw,
    tag: str,
    scope: str,
    snapshot: _HistogramSnapshot,
    *,
    step: int,
) -> None:
    for group, size in enumerate(snapshot.schema.sizes):
        observations = float(snapshot.observations[group].detach().cpu().item())
        if observations <= 0:
            continue
        group_tag = join_tag(tag, snapshot.schema.group_names[group], scope)
        counts = snapshot.mass[group, :size].detach().float().cpu().tolist()
        minimum, maximum = snapshot.schema.ranges[group]
        writer.add_histogram_raw(
            group_tag,
            min=minimum,
            max=maximum,
            num=observations,
            sum=float(snapshot.sums[group].detach().cpu().item()),
            sum_squares=float(snapshot.sum_squares[group].detach().cpu().item()),
            bucket_limits=list(snapshot.schema.bucket_limits[group]),
            bucket_counts=counts,
            global_step=step,
        )


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


def _all_gather(
    trainer: pl.Trainer,
    value: Tensor,
    *,
    error: str,
) -> Tensor:
    strategy = getattr(trainer, "strategy", None)
    all_gather = getattr(strategy, "all_gather", None)
    if not callable(all_gather):
        raise RuntimeError(error)
    result = all_gather(value)
    if not isinstance(result, Tensor):
        raise TypeError("trainer.strategy.all_gather must return a Tensor.")
    return result


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


def _raise_synchronized_error(
    error: _ObservationError | None,
    *,
    context: str,
) -> NoReturn:
    if error is not None:
        raise RuntimeError(f"distributed {context} failed: {error[1]}")
    raise RuntimeError(f"distributed {context} failed on another rank.")


def _remove_observer_hooks(
    module: nn.Module,
    callback: ReferenceType[ObservationCallback],
    collector_id: int,
) -> None:
    for hook_id, hook in tuple(module._forward_hooks.items()):
        if not isinstance(hook, _ForwardCollectorHook):
            continue
        if hook.callback != callback or hook.collector_id != collector_id:
            continue
        module._forward_hooks.pop(hook_id, None)
        module._forward_hooks_with_kwargs.pop(hook_id, None)
        module._forward_hooks_always_called.pop(hook_id, None)


def _is_descendant(path: str, parent: str) -> bool:
    if path == parent:
        return False
    if not parent:
        return True
    return path.startswith(f"{parent}.")


def _collector_owner(
    collector: _Collector,
    item: str,
    *,
    stage: Literal["train", "validation"],
    context: str,
) -> _StateOwner:
    return _StateOwner(
        stage=stage,
        context=context,
        module_path=collector.module_path,
        spec_name=collector.spec.name.strip("/"),
        item=item,
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


__all__ = ["ObservationCallback"]
