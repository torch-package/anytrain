from __future__ import annotations

import ast
import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol, cast

import lightning.pytorch as pl
import torch

from .. import _capability
from ._data import UnitBatch, padding_ratio, units_per_second
from ._spec import LRSchedule, UnitClock

_LEGACY_NO_MATCH = object()


@dataclass(frozen=True)
class _Measurement:
    elapsed: float
    valid: float
    padded: float | None
    padded_valid: float | None
    batches: int


@dataclass(frozen=True)
class _ResolvedUnits:
    valid: float
    padded: float | None
    padded_valid: float | None
    batches: int


@dataclass(frozen=True)
class UnitScheduleUpdate:
    """One training-batch progress update from a unit-clock schedule."""

    batch_index: int | None
    global_step: int
    completed_unit_epochs: int | None
    previous_unit_epochs: int | None
    advanced_unit_epochs: int
    validation_due: bool

    def crossed_unit_epoch_interval(self, every_n_epochs: int) -> bool:
        if isinstance(every_n_epochs, bool) or not isinstance(every_n_epochs, int):
            raise TypeError("every_n_epochs must be an integer.")
        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be positive.")
        if self.completed_unit_epochs is None or self.previous_unit_epochs is None:
            raise RuntimeError("unit epochs require clock.units_per_epoch.")
        return (
            self.completed_unit_epochs // every_n_epochs
            > self.previous_unit_epochs // every_n_epochs
        )


class UnitScheduleConsumer(Protocol):
    def on_unit_schedule_update(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        update: UnitScheduleUpdate,
    ) -> None: ...


class _AllGatherStrategy(Protocol):
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor: ...


@dataclass
class _PendingUnits:
    valid: float = 0.0
    padded: float = 0.0
    padded_valid: float = 0.0
    batches: int = 0
    padded_batches: int = 0

    def add(self, units: _ResolvedUnits) -> None:
        self.valid += units.valid
        self.batches += 1
        if units.padded is not None:
            self.padded += units.padded
            self.padded_valid += units.valid
            self.padded_batches += 1

    def clear(self) -> None:
        self.valid = 0.0
        self.padded = 0.0
        self.padded_valid = 0.0
        self.batches = 0
        self.padded_batches = 0


class UnitScheduleCallback(pl.Callback):
    """Own unit-clock progress, LR stepping, throughput logs, and unit stopping."""

    def __init__(self, schedule: LRSchedule) -> None:
        super().__init__()
        clock = schedule.clock
        if not isinstance(clock, UnitClock):
            raise TypeError("UnitScheduleCallback requires a UnitClock schedule.")
        self.schedule = schedule
        self._clock = clock
        self.total_valid = 0.0
        self.total_padded = 0.0
        self._update_started_at: float | None = None
        self._batch_active = False
        self._pending = _PendingUnits()
        self._last_global_step: int | None = None
        self._base_lrs: list[list[float]] | None = None
        self._last_lrs: list[list[float]] | None = None
        self._next_log_at: float | None = None
        self._measurements: deque[_Measurement] = deque()
        self._measurement_batches = 0
        self._unit_epochs_advanced = 0
        self._unit_schedule_consumers: list[UnitScheduleConsumer] = []
        self._unit_validation_intervals: set[int] = set()

    @property
    def state_key(self) -> str:
        return self._generate_state_key(unit=self.clock.unit)

    @property
    def clock(self) -> UnitClock:
        return self._clock

    @property
    def completed_unit_epochs(self) -> int:
        units_per_epoch = self.clock.units_per_epoch
        if units_per_epoch is None:
            raise RuntimeError("unit epochs require clock.units_per_epoch.")
        return int(math.floor(self.total_valid / units_per_epoch))

    @property
    def unit_epochs_advanced(self) -> int:
        return self._unit_epochs_advanced

    def crossed_unit_epoch_interval(self, every_n_epochs: int) -> bool:
        _validate_unit_epoch_interval(every_n_epochs)
        current = self.completed_unit_epochs
        previous = current - self.unit_epochs_advanced
        return current // every_n_epochs > previous // every_n_epochs

    def register_unit_schedule_consumer(
        self,
        consumer: UnitScheduleConsumer,
        *,
        validation_every_n_epochs: int | None = None,
    ) -> None:
        """Register a callback that consumes every unit-clock update."""

        if not callable(getattr(consumer, "on_unit_schedule_update", None)):
            raise TypeError("unit schedule consumer must expose on_unit_schedule_update.")
        if validation_every_n_epochs is not None:
            _validate_unit_epoch_interval(validation_every_n_epochs)
            self._unit_validation_intervals.add(validation_every_n_epochs)
        if not any(candidate is consumer for candidate in self._unit_schedule_consumers):
            self._unit_schedule_consumers.append(consumer)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._unit_epochs_advanced = 0
        self._reject_lightning_schedulers(trainer)
        if self._base_lrs is None:
            self._base_lrs = _current_lrs(_optimizers(trainer))
        self._last_global_step = _capability.global_step(trainer)
        self._next_log_at = _next_log_at(
            self.total_valid,
            self.clock.log_every_n_units,
        )
        self._apply_lr(trainer, self.schedule.lr_ratio(self.total_valid))
        stop_at = self.schedule.stop_unit()
        if stop_at is not None and self.total_valid >= stop_at:
            trainer.should_stop = True
        self._sync_cuda(pl_module)
        del pl_module

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        self._unit_epochs_advanced = 0
        if self._batch_active:
            raise RuntimeError("A training batch started before the previous batch ended.")
        self._batch_active = True
        if self._update_started_at is None:
            self._update_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._batch_active:
            return
        self._batch_active = False

        units = self.clock.provider(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        self._pending.add(self._local_units(units))

        global_step = _capability.global_step(trainer)
        optimizer_steps = self._optimizer_steps(global_step)
        if optimizer_steps == 0:
            self._notify_unit_schedule_consumers(trainer, pl_module, batch_idx)
            return

        self._sync_cuda(pl_module)
        if self._update_started_at is None:
            raise RuntimeError("unit schedule update timer was not started.")
        elapsed = time.perf_counter() - self._update_started_at
        self._update_started_at = None
        if elapsed <= 0:
            raise RuntimeError(f"non-positive measured unit schedule time={elapsed}.")

        previous_unit_epochs = (
            self.completed_unit_epochs
            if self.clock.units_per_epoch is not None
            else 0
        )
        measurement = self._resolve_pending_update(
            trainer,
            pl_module,
            elapsed=elapsed,
        )
        self.total_valid += measurement.valid
        if measurement.padded is not None:
            self.total_padded += measurement.padded
        if self.clock.units_per_epoch is not None:
            self._unit_epochs_advanced = (
                self.completed_unit_epochs - previous_unit_epochs
            )
        self._append_measurement(measurement)

        self._apply_lr(trainer, self.schedule.lr_ratio(self.total_valid))
        self._last_global_step = global_step

        stop_at = self.schedule.stop_unit()
        should_stop = stop_at is not None and self.total_valid >= stop_at
        if should_stop:
            trainer.should_stop = True

        self._log_if_due(
            pl_module,
            measurement,
            force=should_stop or self.unit_epochs_advanced > 0,
        )
        self._notify_unit_schedule_consumers(trainer, pl_module, batch_idx)

    def _notify_unit_schedule_consumers(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
    ) -> None:
        completed: int | None
        if self.clock.units_per_epoch is None:
            completed = None
            previous = None
        else:
            completed = self.completed_unit_epochs
            previous = completed - self.unit_epochs_advanced
        validation_due = (
            completed is not None
            and previous is not None
            and any(
                completed // interval > previous // interval
                for interval in self._unit_validation_intervals
            )
        )
        update = UnitScheduleUpdate(
            batch_index=_total_batch_index(trainer, fallback=batch_idx),
            global_step=_capability.global_step(trainer),
            completed_unit_epochs=completed,
            previous_unit_epochs=previous,
            advanced_unit_epochs=self.unit_epochs_advanced,
            validation_due=validation_due,
        )
        for consumer in tuple(self._unit_schedule_consumers):
            consumer.on_unit_schedule_update(trainer, pl_module, update)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if self._batch_active:
            raise RuntimeError("training epoch ended while a unit batch was still active.")
        if self._pending.batches > 0:
            raise RuntimeError(
                "training epoch ended with unit batches that did not reach an "
                "optimizer-step boundary."
            )
        self._update_started_at = None

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._reset_incomplete_update()

    def state_dict(self) -> dict[str, Any]:
        return {
            "unit": self.clock.unit,
            "total_valid": self.total_valid,
            "total_padded": self.total_padded,
            "base_lrs": self._base_lrs,
            "last_global_step": self._last_global_step,
            "last_lrs": self._last_lrs,
            "schedule": self.schedule.fingerprint(),
            "stop_at": self.schedule.stop_unit(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError("UnitScheduleCallback state must be a dict.")
        if state_dict.get("unit") != self.clock.unit:
            raise ValueError("UnitScheduleCallback state unit does not match callback unit.")
        raw_schedule = state_dict.get("schedule")
        if not _schedule_fingerprint_matches(raw_schedule, self.schedule):
            raise ValueError("UnitScheduleCallback state schedule does not match callback.")
        saved_stop_at = _saved_stop_at(state_dict, raw_schedule, self.schedule)
        _validate_resume_stop_at(saved_stop_at, self.schedule.stop_unit())
        total_valid = _non_negative_float(state_dict.get("total_valid"), "total_valid")
        total_padded = _non_negative_float(state_dict.get("total_padded"), "total_padded")
        raw_base_lrs = state_dict.get("base_lrs")
        base_lrs = None if raw_base_lrs is None else _lr_groups(raw_base_lrs, "base_lrs")
        last_step = state_dict.get("last_global_step")
        if last_step is not None and (isinstance(last_step, bool) or not isinstance(last_step, int)):
            raise TypeError("UnitScheduleCallback last_global_step must be an integer or None.")
        raw_last_lrs = state_dict.get("last_lrs")
        last_lrs = None if raw_last_lrs is None else _lr_groups(raw_last_lrs, "last_lrs")

        self.total_valid = total_valid
        self.total_padded = total_padded
        self._base_lrs = base_lrs
        self._last_global_step = last_step
        self._last_lrs = last_lrs
        self._unit_epochs_advanced = 0

    def _local_units(self, units: object) -> _ResolvedUnits:
        if not isinstance(units, UnitBatch):
            raise TypeError(f"unit provider must return UnitBatch, got {type(units)!r}.")
        if units.unit != self.clock.unit:
            raise ValueError(
                f"unit provider returned unit {units.unit!r}, expected {self.clock.unit!r}."
            )
        valid = float(units.valid)
        padded = None if units.padded is None else float(units.padded)
        return _ResolvedUnits(
            valid=valid,
            padded=padded,
            padded_valid=None if padded is None else valid,
            batches=1,
        )

    def _resolve_pending_update(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        elapsed: float,
    ) -> _Measurement:
        if self._pending.batches == 0:
            raise RuntimeError("optimizer step completed without any pending unit batches.")
        pending = self._pending
        local = _ResolvedUnits(
            valid=pending.valid,
            padded=None if pending.padded_batches == 0 else pending.padded,
            padded_valid=(
                None if pending.padded_batches == 0 else pending.padded_valid
            ),
            batches=pending.batches,
        )
        try:
            if not self.clock.sync_distributed or _capability.world_size(trainer) <= 1:
                return _Measurement(
                    elapsed=elapsed,
                    valid=local.valid,
                    padded=local.padded,
                    padded_valid=local.padded_valid,
                    batches=local.batches,
                )
            return _distributed_update(
                trainer,
                pl_module,
                local,
                elapsed=elapsed,
                batches=pending.batches,
                padded_batches=pending.padded_batches,
            )
        finally:
            pending.clear()

    def _optimizer_steps(self, global_step: int) -> int:
        if self._last_global_step is None:
            self._last_global_step = global_step
            return 0
        optimizer_steps = global_step - self._last_global_step
        if optimizer_steps < 0:
            raise RuntimeError("trainer.global_step moved backwards during unit scheduling.")
        return optimizer_steps

    def _append_measurement(self, measurement: _Measurement) -> None:
        self._measurements.append(measurement)
        self._measurement_batches += measurement.batches
        capacity = self.clock.measure_window_batches
        while self._measurement_batches > capacity and len(self._measurements) > 1:
            removed = self._measurements.popleft()
            self._measurement_batches -= removed.batches

    def _apply_lr(self, trainer: pl.Trainer, ratio: float) -> None:
        optimizers = _optimizers(trainer)
        if self._base_lrs is None:
            self._base_lrs = _current_lrs(optimizers)
        if len(optimizers) != len(self._base_lrs):
            raise RuntimeError("optimizer count changed during unit scheduling.")
        self._validate_lr_owner(optimizers)
        for optimizer, base_lrs in zip(optimizers, self._base_lrs, strict=False):
            if len(optimizer.param_groups) != len(base_lrs):
                raise RuntimeError("optimizer param group count changed during unit scheduling.")
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=False):
                group["lr"] = float(base_lr) * ratio
        self._last_lrs = _current_lrs(optimizers)

    def _validate_lr_owner(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if self.clock.allow_external_lr_changes or self._last_lrs is None:
            return
        current = _current_lrs(optimizers)
        if len(current) != len(self._last_lrs):
            raise RuntimeError("optimizer count changed during unit scheduling.")
        for optimizer_index, (actual_group, expected_group) in enumerate(zip(current, self._last_lrs, strict=False)):
            if len(actual_group) != len(expected_group):
                raise RuntimeError("optimizer param group count changed during unit scheduling.")
            for group_index, (actual, expected) in enumerate(zip(actual_group, expected_group, strict=False)):
                if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                    raise RuntimeError(
                        "optimizer lr was modified outside UnitScheduleCallback "
                        f"(optimizer={optimizer_index}, group={group_index})."
                    )

    def _log_if_due(
        self,
        pl_module: pl.LightningModule,
        measurement: _Measurement,
        *,
        force: bool = False,
    ) -> None:
        every = self.clock.log_every_n_units
        if every is not None:
            next_log_at = self._next_log_at
            if next_log_at is None:
                next_log_at = _next_log_at(self.total_valid, every)
                if next_log_at is None:
                    raise RuntimeError("unit log interval unexpectedly resolved to None.")
                self._next_log_at = next_log_at
            if not force and self.total_valid < next_log_at:
                return
            while self.total_valid >= next_log_at:
                next_log_at += every
            self._next_log_at = next_log_at

        window_time = math.fsum(item.elapsed for item in self._measurements)
        window_valid = math.fsum(item.valid for item in self._measurements)
        unit = self.clock.unit
        metrics = {
            f"progress/{unit}": measurement.valid,
            f"progress/{unit}_total": self.total_valid,
            f"progress/{unit}_per_second": units_per_second(
                valid_units=measurement.valid,
                step_time=measurement.elapsed,
            ),
            f"progress/{unit}_per_second_window": units_per_second(
                valid_units=window_valid,
                step_time=window_time,
            ),
            "progress/lr_ratio": self.schedule.lr_ratio(self.total_valid),
        }
        if self.clock.units_per_epoch is not None:
            completed = self.completed_unit_epochs
            metrics.update(
                {
                    "progress/unit_epoch": completed,
                    "progress/unit_epoch_progress": (
                        self.total_valid / self.clock.units_per_epoch - completed
                    ),
                }
            )
        if measurement.padded is not None:
            window_padded = math.fsum(
                item.padded for item in self._measurements if item.padded is not None
            )
            window_padded_valid = math.fsum(
                item.padded_valid
                for item in self._measurements
                if item.padded_valid is not None
            )
            metrics.update(
                {
                    f"progress/{unit}_padded": measurement.padded,
                    f"progress/{unit}_padded_total": self.total_padded,
                }
            )
            if measurement.padded > 0:
                if measurement.padded_valid is None:
                    raise RuntimeError("padded measurement is missing its valid unit count.")
                metrics["progress/padding_ratio"] = padding_ratio(
                    valid_units=measurement.padded_valid,
                    padded_units=measurement.padded,
                )
            if window_padded > 0:
                metrics["progress/padding_ratio_window"] = padding_ratio(
                    valid_units=window_padded_valid,
                    padded_units=window_padded,
                )
        pl_module.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

    def _reject_lightning_schedulers(self, trainer: pl.Trainer) -> None:
        configs = getattr(trainer, "lr_scheduler_configs", None)
        if configs:
            raise RuntimeError(
                "UnitScheduleCallback cannot run with Lightning lr_schedulers. "
                "Return only optimizer from configure_optimizers() for unit-based scheduling."
            )

    def _sync_cuda(self, pl_module: pl.LightningModule) -> None:
        device = _device(pl_module)
        if self.clock.sync_cuda and device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _reset_incomplete_update(self) -> None:
        self._batch_active = False
        self._update_started_at = None
        self._pending.clear()


def _distributed_update(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    units: _ResolvedUnits,
    *,
    elapsed: float,
    batches: int,
    padded_batches: int,
) -> _Measurement:
    world_size = _capability.world_size(trainer)
    payload = torch.tensor(
        [
            units.valid,
            0.0 if units.padded is None else units.padded,
            0.0 if units.padded_valid is None else units.padded_valid,
            float(batches),
            float(batches * batches),
            float(padded_batches),
            float(padded_batches * padded_batches),
            elapsed,
        ],
        dtype=torch.float64,
        device=_device(pl_module),
    )
    gathered = _strategy_all_gather(trainer, payload)
    if gathered.numel() != world_size * payload.numel():
        raise RuntimeError(
            "trainer.strategy.all_gather returned an invalid distributed unit payload."
        )
    rows = gathered.detach().reshape(world_size, payload.numel())
    aggregated = torch.cat(
        (rows[:, :-1].sum(dim=0), rows[:, -1].max().reshape(1))
    )
    values = aggregated.cpu().tolist()
    global_batches = int(values[3])
    global_batch_squares = int(values[4])
    if world_size * global_batch_squares != global_batches * global_batches:
        raise RuntimeError(
            "distributed unit scheduling observed different accumulation windows across ranks."
        )
    global_padded_batches = int(values[5])
    global_padded_batch_squares = int(values[6])
    if not 0 <= global_padded_batches <= global_batches:
        raise RuntimeError("distributed padded batch count is invalid.")
    if (
        world_size * global_padded_batch_squares
        != global_padded_batches * global_padded_batches
    ):
        raise ValueError(
            "unit provider padded counts must be present on the same number of "
            "batches on all ranks."
        )
    return _Measurement(
        elapsed=float(values[7]),
        valid=float(values[0]),
        padded=None if global_padded_batches == 0 else float(values[1]),
        padded_valid=None if global_padded_batches == 0 else float(values[2]),
        batches=batches,
    )


def _strategy_all_gather(
    trainer: pl.Trainer,
    tensor: torch.Tensor,
) -> torch.Tensor:
    strategy = getattr(trainer, "strategy", None)
    all_gather = getattr(strategy, "all_gather", None)
    if not callable(all_gather):
        raise RuntimeError(
            "Distributed unit scheduling requires trainer.strategy.all_gather."
        )
    return cast(_AllGatherStrategy, strategy).all_gather(tensor)


def _optimizers(trainer: pl.Trainer) -> list[torch.optim.Optimizer]:
    raw = getattr(trainer, "optimizers", None)
    if raw is None:
        raise RuntimeError("UnitScheduleCallback requires trainer.optimizers.")
    optimizers = list(cast(Iterable[torch.optim.Optimizer], raw))
    if not optimizers:
        raise RuntimeError("UnitScheduleCallback requires trainer.optimizers.")
    return optimizers


def _current_lrs(optimizers: list[torch.optim.Optimizer]) -> list[list[float]]:
    return [[float(group["lr"]) for group in optimizer.param_groups] for optimizer in optimizers]


def _device(pl_module: pl.LightningModule) -> torch.device | None:
    try:
        return pl_module.device
    except (AttributeError, RuntimeError):
        return None


def _next_log_at(total_valid: float, every: float | None) -> float | None:
    if every is None:
        return None
    return (math.floor(total_valid / every) + 1) * every


def _validate_unit_epoch_interval(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("every_n_epochs must be an integer.")
    if value <= 0:
        raise ValueError("every_n_epochs must be positive.")
    return value


def _total_batch_index(trainer: pl.Trainer, *, fallback: int) -> int | None:
    fit_loop = getattr(trainer, "fit_loop", None)
    epoch_loop = getattr(fit_loop, "epoch_loop", None)
    value = getattr(epoch_loop, "total_batch_idx", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback if isinstance(fallback, int) and not isinstance(fallback, bool) else None
    return value


def register_unit_schedule_consumer(
    trainer: pl.Trainer,
    schedule: UnitScheduleCallback,
    consumer: UnitScheduleConsumer,
    *,
    validation_every_n_epochs: int | None = None,
) -> None:
    """Attach a runtime consumer without imposing callback-list ordering."""

    if not isinstance(schedule, UnitScheduleCallback):
        raise TypeError("unit schedule consumer requires a UnitScheduleCallback.")
    raw_callbacks = getattr(trainer, "callbacks", None)
    if not isinstance(raw_callbacks, list) or not any(
        callback is schedule for callback in raw_callbacks
    ):
        raise RuntimeError("unit schedule consumer requires the schedule on the Trainer.")
    schedule.register_unit_schedule_consumer(
        consumer,
        validation_every_n_epochs=validation_every_n_epochs,
    )


def _schedule_fingerprint_matches(raw: object, schedule: LRSchedule) -> bool:
    if raw == schedule.fingerprint():
        return True
    return _legacy_schedule_state(raw, schedule) is not _LEGACY_NO_MATCH


def _saved_stop_at(
    state_dict: dict[str, Any],
    raw_schedule: object,
    schedule: LRSchedule,
) -> float | None:
    if "stop_at" in state_dict:
        return _optional_stop_at(state_dict["stop_at"], "stop_at")
    legacy = _legacy_schedule_state(raw_schedule, schedule)
    if legacy is not _LEGACY_NO_MATCH:
        return cast(float | None, legacy)
    # Checkpoints written before the separate stop policy field used the
    # legacy fingerprint.  A matching current fingerprint without the field
    # is treated as an unbounded policy for forward compatibility.
    return None


def _validate_resume_stop_at(saved: float | None, current: float | None) -> None:
    """Permit only stop-budget extensions when resuming a checkpoint."""

    if saved is None:
        if current is not None:
            raise ValueError(
                "UnitScheduleCallback resume stop_at cannot introduce a finite "
                "budget when the checkpoint was unbounded."
            )
        return
    if current is not None and current < saved:
        raise ValueError(
            "UnitScheduleCallback resume stop_at cannot be smaller than the "
            "checkpoint budget."
        )


def _legacy_schedule_state(
    raw: object,
    schedule: LRSchedule,
) -> float | None | object:
    if not isinstance(raw, str):
        return _LEGACY_NO_MATCH
    try:
        parts = ast.literal_eval(raw)
        current_parts = ast.literal_eval(schedule.fingerprint())
    except (SyntaxError, ValueError):
        return _LEGACY_NO_MATCH
    if not isinstance(parts, tuple) or not isinstance(current_parts, tuple):
        return _LEGACY_NO_MATCH
    if len(current_parts) != 4:
        return _LEGACY_NO_MATCH

    if len(parts) == 6:
        core_parts = parts[:4]
        stop_part = parts[4]
        legacy_end_flag = parts[5]
    elif len(parts) == 5 and current_parts[2] is None:
        # Fingerprints written before units_per_epoch was added omitted the
        # third core field.  They remain readable only for schedules that do
        # not define logical unit epochs.
        core_parts = (parts[0], parts[1], None, parts[2])
        stop_part = parts[3]
        legacy_end_flag = parts[4]
    else:
        return _LEGACY_NO_MATCH

    if core_parts != current_parts:
        return _LEGACY_NO_MATCH
    if not isinstance(legacy_end_flag, bool):
        return _LEGACY_NO_MATCH
    try:
        saved = _optional_stop_at(stop_part, "legacy stop_at")
    except (TypeError, ValueError):
        return _LEGACY_NO_MATCH
    if saved is None and legacy_end_flag:
        return schedule.total_duration
    return saved


def _optional_stop_at(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"UnitScheduleCallback {name} must be a real number or None.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"UnitScheduleCallback {name} must be finite and positive.")
    return resolved


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _lr_groups(value: object, name: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"UnitScheduleCallback {name} must be a sequence of groups.")
    if not value:
        raise ValueError(f"UnitScheduleCallback {name} must contain an optimizer.")

    result: list[list[float]] = []
    for group in value:
        if not isinstance(group, (list, tuple)):
            raise TypeError(f"UnitScheduleCallback {name} groups must be sequences.")
        if not group:
            raise ValueError(f"UnitScheduleCallback {name} groups must not be empty.")
        result.append(
            [_non_negative_float(item, f"{name} learning rate") for item in group]
        )
    return result


__all__ = [
    "UnitScheduleCallback",
    "UnitScheduleConsumer",
    "UnitScheduleUpdate",
    "register_unit_schedule_consumer",
]
