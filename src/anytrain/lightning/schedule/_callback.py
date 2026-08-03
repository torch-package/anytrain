from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import lightning.pytorch as pl
import torch

from .. import _capability
from ._data import UnitBatch, padding_ratio, units_per_second
from ._spec import LRSchedule, UnitClock


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

    @property
    def state_key(self) -> str:
        return self._generate_state_key(unit=self.clock.unit)

    @property
    def clock(self) -> UnitClock:
        return self._clock

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reject_lightning_schedulers(trainer)
        if self._base_lrs is None:
            self._base_lrs = _current_lrs(_optimizers(trainer))
        self._last_global_step = _capability.global_step(trainer)
        self._next_log_at = _next_log_at(
            self.total_valid,
            self.clock.log_every_n_units,
        )
        self._apply_lr(trainer, self.schedule.lr_ratio(self.total_valid))
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
            return

        self._sync_cuda(pl_module)
        if self._update_started_at is None:
            raise RuntimeError("unit schedule update timer was not started.")
        elapsed = time.perf_counter() - self._update_started_at
        self._update_started_at = None
        if elapsed <= 0:
            raise RuntimeError(f"non-positive measured unit schedule time={elapsed}.")

        measurement = self._resolve_pending_update(
            trainer,
            pl_module,
            elapsed=elapsed,
        )
        self.total_valid += measurement.valid
        if measurement.padded is not None:
            self.total_padded += measurement.padded
        self._append_measurement(measurement)

        self._apply_lr(trainer, self.schedule.lr_ratio(self.total_valid))
        self._last_global_step = global_step

        stop_at = self.schedule.stop_unit()
        should_stop = stop_at is not None and self.total_valid >= stop_at
        if should_stop:
            trainer.should_stop = True

        self._log_if_due(pl_module, measurement, force=should_stop)

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
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError("UnitScheduleCallback state must be a dict.")
        if state_dict.get("unit") != self.clock.unit:
            raise ValueError("UnitScheduleCallback state unit does not match callback unit.")
        if state_dict.get("schedule") != self.schedule.fingerprint():
            raise ValueError("UnitScheduleCallback state schedule does not match callback.")
        self.total_valid = _non_negative_float(state_dict.get("total_valid"), "total_valid")
        self.total_padded = _non_negative_float(state_dict.get("total_padded"), "total_padded")
        base_lrs = state_dict.get("base_lrs")
        if base_lrs is not None:
            self._base_lrs = [[float(value) for value in group] for group in base_lrs]
        last_step = state_dict.get("last_global_step")
        if last_step is not None and (isinstance(last_step, bool) or not isinstance(last_step, int)):
            raise TypeError("UnitScheduleCallback last_global_step must be an integer or None.")
        self._last_global_step = last_step
        last_lrs = state_dict.get("last_lrs")
        if last_lrs is not None:
            self._last_lrs = [[float(value) for value in group] for group in last_lrs]

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
        self._validate_lr_owner(optimizers)
        for optimizer, base_lrs in zip(optimizers, self._base_lrs):
            if len(optimizer.param_groups) != len(base_lrs):
                raise RuntimeError("optimizer param group count changed during unit scheduling.")
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = float(base_lr) * ratio
        self._last_lrs = _current_lrs(optimizers)

    def _validate_lr_owner(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if self.clock.allow_external_lr_changes or self._last_lrs is None:
            return
        current = _current_lrs(optimizers)
        if len(current) != len(self._last_lrs):
            raise RuntimeError("optimizer count changed during unit scheduling.")
        for optimizer_index, (actual_group, expected_group) in enumerate(zip(current, self._last_lrs)):
            if len(actual_group) != len(expected_group):
                raise RuntimeError("optimizer param group count changed during unit scheduling.")
            for group_index, (actual, expected) in enumerate(zip(actual_group, expected_group)):
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


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be a real number.")
    try:
        resolved = float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


__all__ = ["UnitScheduleCallback"]
