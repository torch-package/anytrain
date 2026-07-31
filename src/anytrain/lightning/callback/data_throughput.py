from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from lightning import pytorch as pl

from anytrain.lightning.perf.data import (
    padding_ratio,
    require_unit_name,
    units_realtime,
)


@dataclass(frozen=True)
class DataUnits:
    """Homogeneous valid/padded counts for one train batch.

    ``unit`` names the metric suffix (``audio_seconds``, ``tokens``, ``frames``).
    Both fields use that unit; padding is ``1 - valid / padded``.
    """

    valid: float
    padded: float
    unit: str

    def __post_init__(self) -> None:
        require_unit_name(self.unit)
        padding_ratio(valid_units=self.valid, padded_units=self.padded)


class DataUnitsProvider(Protocol):
    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> DataUnits: ...


@dataclass(frozen=True)
class _Measurement:
    elapsed: float
    valid: float
    padded: float


class DataThroughputCallback(pl.Callback):
    """Log padding ratio and valid-unit realtime from a ``DataUnitsProvider``.

    Does not interpret batch schema. Pair with ``PerformanceCallback`` for MFU;
    both may log ``perf/step_time``.
    """

    def __init__(
        self,
        *,
        data_units: DataUnitsProvider | Callable[..., DataUnits],
        log_every_n_steps: int = 100,
        warmup_steps: int = 20,
        measure_window_steps: int = 100,
        sync_cuda: bool = True,
    ) -> None:
        _require_positive_int(log_every_n_steps, "log_every_n_steps")
        _require_non_negative_int(warmup_steps, "warmup_steps")
        _require_positive_int(measure_window_steps, "measure_window_steps")
        if not callable(data_units):
            raise TypeError("data_units must be callable.")
        if not isinstance(sync_cuda, bool):
            raise TypeError("sync_cuda must be a boolean.")

        self.data_units = data_units
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.measure_window_steps = measure_window_steps
        self.sync_cuda = sync_cuda
        self._step_started_at: float | None = None
        self._unit: str | None = None
        self._measurements: deque[_Measurement] = deque(maxlen=measure_window_steps)

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
        _sync_cuda(self.sync_cuda)
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

        _sync_cuda(self.sync_cuda)
        step_time = time.perf_counter() - self._step_started_at
        self._step_started_at = None
        if step_time <= 0:
            raise RuntimeError(f"non-positive measured step_time={step_time}.")

        units = self.data_units(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        if not isinstance(units, DataUnits):
            raise TypeError(
                f"data_units provider must return DataUnits, got {type(units)!r}."
            )
        if self._unit is None:
            self._unit = units.unit
        elif units.unit != self._unit:
            raise ValueError(
                f"data_units unit changed from {self._unit!r} to {units.unit!r} "
                "during training."
            )

        self._measurements.append(
            _Measurement(elapsed=step_time, valid=units.valid, padded=units.padded)
        )

        global_step = int(trainer.global_step)
        if global_step < self.warmup_steps:
            return
        if global_step % self.log_every_n_steps != 0:
            return

        window_time = sum(item.elapsed for item in self._measurements)
        window_valid = sum(item.valid for item in self._measurements)
        metrics = {
            "perf/step_time": step_time,
            "perf/step_time_window": window_time / len(self._measurements),
            "perf/padding_ratio": padding_ratio(
                valid_units=units.valid,
                padded_units=units.padded,
            ),
            f"perf/{units.unit}": units.valid,
            f"perf/{units.unit}_realtime": units_realtime(
                valid_units=units.valid,
                step_time=step_time,
            ),
            f"perf/{units.unit}_realtime_window": units_realtime(
                valid_units=window_valid,
                step_time=window_time,
            ),
        }
        pl_module.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

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


def _sync_cuda(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


__all__ = [
    "DataThroughputCallback",
    "DataUnits",
    "DataUnitsProvider",
]
