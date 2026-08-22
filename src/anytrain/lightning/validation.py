from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, TypedDict

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ..evaluator.weighted import Metric, reduce
from . import _capability
from .schedule import UnitScheduleCallback


def log(
    module: LightningModule,
    metrics: Mapping[str, Metric],
    *,
    prefix: str = "val",
) -> None:
    resolved_prefix = _prefix(prefix)
    if not isinstance(metrics, Mapping):
        raise TypeError("validation metrics must be a mapping.")
    if not metrics:
        raise ValueError("validation metrics must contain at least one value.")

    resolved: dict[str, Metric] = {}
    for raw_name, metric in metrics.items():
        name = _name(raw_name)
        if name in resolved:
            raise ValueError(f"Duplicate validation metric name {name!r}.")
        resolved[name] = metric
    for name, (value, count) in reduce(resolved).items():
        module.log(
            f"{resolved_prefix}{name}",
            value,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=_distributed_count(count),
        )


class Record(TypedDict):
    step: int
    metrics: dict[str, float]


class Report(TypedDict):
    sanity: list[Record]
    interval: list[Record]


class History(Callback):
    def __init__(self, *, prefix: str = "val", strict: bool = True) -> None:
        super().__init__()
        if not isinstance(strict, bool):
            raise TypeError("validation history strict mode must be boolean.")
        self.prefix = _prefix(prefix)
        self.strict = strict
        self.sanity: list[Record] = []
        self.interval: list[Record] = []

    @property
    def state_key(self) -> str:
        return self._generate_state_key(prefix=self.prefix)

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        metrics = {
            name: _scalar(name, value)
            for name, value in sorted(trainer.callback_metrics.items())
            if name.startswith(self.prefix)
        }
        if not metrics:
            if self.strict:
                raise RuntimeError(
                    f"validation completed without {self.prefix}* callback metrics."
                )
            return
        record = Record(step=_capability.global_step(trainer), metrics=metrics)
        values = self.sanity if trainer.sanity_checking else self.interval
        _append(values, record)

    def report(self) -> Report:
        return {
            "sanity": _copy(self.sanity),
            "interval": _copy(self.interval),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "strict": self.strict,
            "sanity": _copy(self.sanity),
            "interval": _copy(self.interval),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) != {"prefix", "strict", "sanity", "interval"}:
            raise TypeError(
                "validation history state requires prefix, strict, sanity, and interval."
            )
        state_prefix = state_dict["prefix"]
        if not isinstance(state_prefix, str):
            raise TypeError("validation history state prefix must be a string.")
        if state_prefix != self.prefix:
            raise ValueError("validation history state prefix does not match the callback.")
        state_strict = state_dict["strict"]
        if not isinstance(state_strict, bool):
            raise TypeError("validation history state strict mode must be boolean.")
        if state_strict is not self.strict:
            raise ValueError("validation history state strict mode does not match the callback.")
        sanity = _records(state_dict, "sanity", prefix=self.prefix)
        interval = _records(state_dict, "interval", prefix=self.prefix)
        self.sanity = sanity
        self.interval = interval


class UnitEpochInterval(Callback):
    """Run Lightning validation when the shared unit clock crosses an epoch boundary."""

    def __init__(
        self,
        schedule: UnitScheduleCallback,
        *,
        every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(schedule, UnitScheduleCallback):
            raise TypeError("unit epoch validation requires a UnitScheduleCallback.")
        if isinstance(every_n_epochs, bool) or not isinstance(every_n_epochs, int):
            raise TypeError("validation every_n_epochs must be an integer.")
        if every_n_epochs <= 0:
            raise ValueError("validation every_n_epochs must be positive.")
        if schedule.clock.units_per_epoch is None:
            raise ValueError("unit epoch validation requires clock.units_per_epoch.")
        self.schedule = schedule
        self.every_n_epochs = every_n_epochs

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            unit=self.schedule.clock.unit,
            every_n_epochs=self.every_n_epochs,
        )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.enable_validation:
            raise RuntimeError("unit epoch validation requires an enabled validation loop.")
        if trainer.check_val_every_n_epoch is not None:
            raise RuntimeError(
                "unit epoch validation requires check_val_every_n_epoch=None."
            )
        if trainer.world_size > 1 and not self.schedule.clock.sync_distributed:
            raise RuntimeError(
                "distributed unit epoch validation requires a globally synchronized unit clock."
            )
        raw_callbacks = getattr(trainer, "callbacks", None)
        if not isinstance(raw_callbacks, list):
            raise RuntimeError("unit epoch validation requires trainer callbacks.")
        callbacks = list(raw_callbacks)
        if callbacks.index(self.schedule) > callbacks.index(self):
            raise RuntimeError(
                "UnitScheduleCallback must run before unit epoch validation."
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        epoch_loop = trainer.fit_loop.epoch_loop
        current_iteration = epoch_loop.total_batch_idx
        due = self.schedule.crossed_unit_epoch_interval(self.every_n_epochs)
        # Lightning checks this value immediately after callback hooks. Setting the
        # next non-matching integer suppresses dataloader-epoch validation.
        trainer.val_check_batch = current_iteration + (1 if due else 2)


def _prefix(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("validation metric prefix must be a string.")
    normalized = value.strip("/")
    if not normalized or "//" in normalized:
        raise ValueError("validation metric prefix must contain non-empty path segments.")
    return f"{normalized}/"


def _distributed_count(count: int) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return count
    # Lightning mean-syncs its integer batch-size accumulator. Scaling every rank
    # keeps that mean integral when the global observation count is not divisible.
    return count * torch.distributed.get_world_size()


def _name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("validation metric names must be strings.")
    normalized = value.strip("/")
    if (
        not normalized
        or "//" in normalized
        or any(not part for part in normalized.split("/"))
    ):
        raise ValueError("validation metric names must contain non-empty path segments.")
    return normalized


def _scalar(name: str, value: object) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"validation metric {name!r} must be scalar.")
        resolved = float(value.detach().item())
    elif isinstance(value, Real) and not isinstance(value, bool):
        resolved = float(value)
    else:
        raise TypeError(f"validation metric {name!r} must be numeric or a scalar tensor.")
    if not math.isfinite(resolved):
        raise ValueError(f"validation metric {name!r} must be finite.")
    return resolved


def _append(values: list[Record], record: Record) -> None:
    if values and record["step"] < values[-1]["step"]:
        raise RuntimeError("validation steps must not move backwards.")
    if values and record["step"] == values[-1]["step"]:
        values[-1] = record
        return
    values.append(record)


def _copy(values: list[Record]) -> list[Record]:
    return [
        Record(step=value["step"], metrics=dict(value["metrics"]))
        for value in values
    ]


def _records(state: Mapping[str, Any], name: str, *, prefix: str) -> list[Record]:
    values = state[name]
    if not isinstance(values, list):
        raise TypeError(f"validation history state {name!r} must be a list.")
    records = [_record(value, prefix=prefix) for value in values]
    if any(
        current["step"] <= previous["step"]
        for previous, current in zip(records, records[1:], strict=False)
    ):
        raise ValueError(
            f"validation history state {name!r} must be strictly step ordered."
        )
    return records


def _record(value: object, *, prefix: str) -> Record:
    if not isinstance(value, Mapping) or set(value) != {"step", "metrics"}:
        raise TypeError("validation history records require step and metrics.")
    step = value["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TypeError("validation history record step must be a non-negative integer.")
    raw_metrics = value["metrics"]
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("validation history record metrics must be a mapping.")
    metrics: dict[str, float] = {}
    for name, metric in raw_metrics.items():
        resolved_name = _history_name(name)
        if not resolved_name.startswith(prefix):
            raise ValueError(
                f"validation history metric {resolved_name!r} must match prefix {prefix!r}."
            )
        metrics[resolved_name] = _scalar(resolved_name, metric)
    return Record(step=step, metrics=metrics)


def _history_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("validation history metric names must be non-empty strings.")
    return value


__all__ = ["History", "Metric", "Record", "Report", "UnitEpochInterval", "log"]
