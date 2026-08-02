from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

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


@dataclass(frozen=True)
class _ResolvedUnits:
    valid: float
    padded: float | None


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
        self._step_started_at: float | None = None
        self._last_global_step: int | None = None
        self._base_lrs: list[list[float]] | None = None
        self._last_lrs: list[list[float]] | None = None
        self._next_log_at: float | None = None
        self._measurements: deque[_Measurement] = deque(
            maxlen=clock.measure_window_batches
        )

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
        self._sync_cuda()
        del pl_module

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if self._step_started_at is not None:
            raise RuntimeError("A training batch started before the previous batch ended.")
        self._sync_cuda()
        self._step_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._step_started_at is None:
            return

        self._sync_cuda()
        elapsed = time.perf_counter() - self._step_started_at
        self._step_started_at = None
        if elapsed <= 0:
            raise RuntimeError(f"non-positive measured unit schedule time={elapsed}.")

        units = self.clock.provider(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        resolved = self._resolve_units(trainer, pl_module, units)
        self.total_valid += resolved.valid
        if resolved.padded is not None:
            self.total_padded += resolved.padded
        measurement = _Measurement(
            elapsed=elapsed,
            valid=resolved.valid,
            padded=resolved.padded,
        )
        self._measurements.append(measurement)

        global_step = _capability.global_step(trainer)
        optimizer_steps = self._optimizer_steps(global_step)
        if optimizer_steps > 0:
            self._apply_lr(trainer, self.schedule.lr_ratio(self.total_valid))
            self._last_global_step = global_step

        stop_at = self.schedule.stop_unit()
        should_stop = stop_at is not None and self.total_valid >= stop_at
        if should_stop:
            trainer.should_stop = True

        if optimizer_steps > 0 or should_stop:
            self._log_if_due(pl_module, measurement, force=should_stop)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._step_started_at = None

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._step_started_at = None

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

    def _resolve_units(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        units: object,
    ) -> _ResolvedUnits:
        if not isinstance(units, UnitBatch):
            raise TypeError(f"unit provider must return UnitBatch, got {type(units)!r}.")
        if units.unit != self.clock.unit:
            raise ValueError(
                f"unit provider returned unit {units.unit!r}, expected {self.clock.unit!r}."
            )
        local = _ResolvedUnits(valid=float(units.valid), padded=None if units.padded is None else float(units.padded))
        if not self.clock.sync_distributed or _capability.world_size(trainer) <= 1:
            return local
        return _distributed_units(trainer, pl_module, local)

    def _optimizer_steps(self, global_step: int) -> int:
        if self._last_global_step is None:
            self._last_global_step = global_step
            return 0
        optimizer_steps = global_step - self._last_global_step
        if optimizer_steps < 0:
            raise RuntimeError("trainer.global_step moved backwards during unit scheduling.")
        return optimizer_steps

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
            metrics.update(
                {
                    f"progress/{unit}_padded": measurement.padded,
                    f"progress/{unit}_padded_total": self.total_padded,
                }
            )
            if measurement.padded > 0:
                metrics["progress/padding_ratio"] = padding_ratio(
                    valid_units=measurement.valid,
                    padded_units=measurement.padded,
                )
            if window_padded > 0:
                metrics["progress/padding_ratio_window"] = padding_ratio(
                    valid_units=window_valid,
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

    def _sync_cuda(self) -> None:
        if self.clock.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()


def _distributed_units(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    units: _ResolvedUnits,
) -> _ResolvedUnits:
    world_size = _capability.world_size(trainer)
    payload = torch.tensor(
        [
            units.valid,
            0.0 if units.padded is None else units.padded,
            float(units.padded is not None),
        ],
        dtype=torch.float64,
        device=_device(pl_module),
    )
    reduced = _capability.strategy_reduce(
        trainer,
        payload,
        reduce_op="sum",
        error="Distributed unit scheduling requires trainer.strategy.reduce.",
    )
    values = reduced.detach().cpu().tolist()
    present = int(values[2])
    if present not in (0, world_size):
        raise ValueError("unit provider padded counts must be present on all ranks or none.")
    return _ResolvedUnits(
        valid=float(values[0]),
        padded=None if present == 0 else float(values[1]),
    )


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
