from __future__ import annotations

import math
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
from lightning import pytorch as pl

from anytrain.lightning.perf import (
    PeakFlops,
    count_parameters,
    infer_peak_flops,
    model_flops_utilization,
)

from .. import _capability

if TYPE_CHECKING:
    from torch.utils.flop_counter import FlopCounterMode


class FlopsProvider(Protocol):
    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> float: ...


@dataclass(frozen=True)
class _Measurement:
    elapsed: float
    flops: float | None
    optimizer_steps: int


@dataclass(frozen=True)
class _Totals:
    current_time: float
    window_time: float
    current_flops: float | None
    window_flops: float | None
    hardware_peak_flops: float | None
    world_size: int


class PerformanceCallback(pl.Callback):
    def __init__(
        self,
        *,
        model_flops_per_step: float | None = None,
        model_flops_per_batch: FlopsProvider | None = None,
        profile_flops: bool = False,
        compute_dtype: torch.dtype | str | None = None,
        hardware_peak_flops: float | None = None,
        log_every_n_steps: int = 100,
        warmup_steps: int = 20,
        measure_window_steps: int = 100,
        sync_cuda: bool = True,
        sync_distributed: bool = True,
    ) -> None:
        _require_positive_int(log_every_n_steps, "log_every_n_steps")
        _require_non_negative_int(warmup_steps, "warmup_steps")
        _require_positive_int(measure_window_steps, "measure_window_steps")
        if model_flops_per_step is not None:
            _require_positive(model_flops_per_step, "model_flops_per_step")
        if model_flops_per_batch is not None and not callable(model_flops_per_batch):
            raise TypeError("model_flops_per_batch must be callable.")
        if not isinstance(profile_flops, bool):
            raise TypeError("profile_flops must be a boolean.")
        flops_sources = sum(
            (
                model_flops_per_step is not None,
                model_flops_per_batch is not None,
                profile_flops,
            )
        )
        if flops_sources > 1:
            raise ValueError(
                "model_flops_per_step, model_flops_per_batch, and profile_flops "
                "are mutually exclusive."
            )
        if hardware_peak_flops is not None:
            _require_positive(hardware_peak_flops, "hardware_peak_flops")
        for name, value in (
            ("sync_cuda", sync_cuda),
            ("sync_distributed", sync_distributed),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")

        self.model_flops_per_step = model_flops_per_step
        self.model_flops_per_batch = model_flops_per_batch
        self.profile_flops = profile_flops
        self.compute_dtype = compute_dtype
        self.hardware_peak_flops = hardware_peak_flops
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.measure_window_steps = measure_window_steps
        self.sync_cuda = sync_cuda
        self.sync_distributed = sync_distributed
        self.hardware: PeakFlops | None = None
        self._step_started_at: float | None = None
        self._last_global_step: int | None = None
        self._pending_time = 0.0
        self._pending_flops = 0.0
        self._flop_counter: FlopCounterMode | None = None
        self._measurements: deque[_Measurement] = deque(maxlen=measure_window_steps)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset(global_step=_capability.global_step(trainer))
        _reset_peak_memory(pl_module)
        self.hardware = _infer_hardware(
            trainer=trainer,
            pl_module=pl_module,
            compute_dtype=self.compute_dtype,
            hardware_peak_flops=self.hardware_peak_flops,
        )

        metrics = {
            "perf/model_params": float(count_parameters(pl_module)),
            "perf/model_trainable_params": float(count_parameters(pl_module, trainable_only=True)),
        }
        if self.model_flops_per_step is not None:
            metrics["perf/model_flops_per_step"] = float(self.model_flops_per_step)
        if self.hardware is not None:
            metrics["perf/hardware_peak_flops"] = float(self.hardware.flops)

        pl_module.log_dict(metrics, on_step=False, on_epoch=True, logger=True)
        _log_metadata(
            trainer,
            _metadata(
                self.hardware,
                model_flops_source=_model_flops_source(
                    model_flops_per_step=self.model_flops_per_step,
                    model_flops_per_batch=self.model_flops_per_batch,
                    profile_flops=self.profile_flops,
                ),
            ),
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        if self._step_started_at is not None:
            raise RuntimeError("A training batch started before the previous batch ended.")
        if self._last_global_step is None:
            self._last_global_step = _capability.global_step(trainer)
        _sync_distributed(trainer, self.sync_distributed)
        _sync_cuda(self.sync_cuda, pl_module)
        if self.profile_flops:
            counter = _flop_counter()
            counter.__enter__()
            self._flop_counter = counter
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

        self._record_batch(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        step = self._close_optimizer_step(trainer)
        if step is None:
            return
        global_step, measurement = step
        if not self._should_log(global_step):
            return

        metrics = self._metrics(trainer, pl_module, measurement)
        pl_module.log_dict(metrics, on_step=True, on_epoch=False, logger=True, sync_dist=False)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        self._discard_pending(global_step=_capability.global_step(trainer))

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        self._discard_pending(global_step=_capability.global_step(trainer))

    def _record_batch(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        profiled_flops = self._stop_flop_counter()
        _sync_cuda(self.sync_cuda, pl_module)
        if self._step_started_at is None:
            raise RuntimeError("Cannot record a performance batch before it starts.")
        batch_time = time.perf_counter() - self._step_started_at
        self._step_started_at = None
        _require_positive(batch_time, "measured batch time")
        self._pending_time += batch_time
        if profiled_flops is not None:
            self._pending_flops += _flops(profiled_flops, "profiled train batch FLOPs")
        elif self.model_flops_per_batch is not None:
            self._pending_flops += _flops(
                self.model_flops_per_batch(
                    trainer=trainer,
                    pl_module=pl_module,
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx,
                ),
                "model_flops_per_batch result",
            )

    def _close_optimizer_step(
        self,
        trainer: pl.Trainer,
    ) -> tuple[int, _Measurement] | None:
        global_step = _capability.global_step(trainer)
        if self._last_global_step is None:
            self._last_global_step = global_step
            return None
        optimizer_steps = global_step - self._last_global_step
        if optimizer_steps < 0:
            raise RuntimeError("trainer.global_step moved backwards during training.")
        if optimizer_steps == 0:
            return None

        flops: float | None = None
        if self.profile_flops or self.model_flops_per_batch is not None:
            flops = self._pending_flops
        elif self.model_flops_per_step is not None:
            flops = float(self.model_flops_per_step) * optimizer_steps

        measurement = _Measurement(
            elapsed=self._pending_time,
            flops=flops,
            optimizer_steps=optimizer_steps,
        )
        self._measurements.append(measurement)
        self._pending_time = 0.0
        self._pending_flops = 0.0
        self._last_global_step = global_step
        return global_step, measurement

    def _should_log(self, global_step: int) -> bool:
        if global_step < self.warmup_steps:
            return False
        return global_step % self.log_every_n_steps == 0

    def _metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        measurement: _Measurement,
    ) -> dict[str, float]:
        optimizer_steps = measurement.optimizer_steps
        window_steps = sum(item.optimizer_steps for item in self._measurements)
        totals = _totals(
            trainer=trainer,
            pl_module=pl_module,
            current=measurement,
            measurements=self._measurements,
            hardware_peak_flops=None if self.hardware is None else self.hardware.flops,
        )
        metrics = {
            "perf/step_time": totals.current_time / optimizer_steps,
            "perf/step_time_window": totals.window_time / window_steps,
        }
        if totals.current_flops is not None and totals.window_flops is not None:
            metrics["perf/model_flops_per_step"] = (
                totals.current_flops / totals.world_size / optimizer_steps
            )
            metrics["perf/model_flops_per_step_window"] = (
                totals.window_flops / totals.world_size / window_steps
            )
        if totals.hardware_peak_flops is not None and totals.window_flops is not None:
            metrics["perf/mfu"] = model_flops_utilization(
                model_flops_per_step=totals.window_flops,
                step_time=totals.window_time,
                hardware_peak_flops=totals.hardware_peak_flops,
            )
        metrics.update(_memory_metrics(trainer, pl_module))
        return metrics

    def _reset(self, *, global_step: int) -> None:
        self._discard_pending(global_step=global_step)
        self._measurements.clear()

    def _discard_pending(self, *, global_step: int) -> None:
        self._stop_flop_counter()
        self._step_started_at = None
        self._last_global_step = global_step
        self._pending_time = 0.0
        self._pending_flops = 0.0

    def _stop_flop_counter(self) -> float | None:
        counter = self._flop_counter
        if counter is None:
            return None
        self._flop_counter = None
        counter.__exit__(None, None, None)
        return float(counter.get_total_flops())


def _window_flops(measurements: deque[_Measurement]) -> float | None:
    if not measurements or measurements[0].flops is None:
        return None
    values = [item.flops for item in measurements]
    if any(value is None for value in values):
        raise RuntimeError("Performance measurement window mixed configured and missing FLOPs.")
    return math.fsum(value for value in values if value is not None)


def _totals(
    *,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    current: _Measurement,
    measurements: deque[_Measurement],
    hardware_peak_flops: float | None,
) -> _Totals:
    window_time = math.fsum(item.elapsed for item in measurements)
    window_flops = _window_flops(measurements)
    world_size = _capability.world_size(trainer)
    if world_size <= 1:
        return _Totals(
            current_time=current.elapsed,
            window_time=window_time,
            current_flops=current.flops,
            window_flops=window_flops,
            hardware_peak_flops=hardware_peak_flops,
            world_size=1,
        )

    device = _device(pl_module)
    summed = torch.tensor(
        [
            0.0 if current.flops is None else current.flops,
            0.0 if window_flops is None else window_flops,
            0.0 if hardware_peak_flops is None else hardware_peak_flops,
            float(current.flops is not None),
            float(window_flops is not None),
            float(hardware_peak_flops is not None),
        ],
        dtype=torch.float64,
        device=device,
    )
    maximum = torch.tensor(
        [current.elapsed, *(item.elapsed for item in measurements)],
        dtype=torch.float64,
        device=device,
    )
    reduce_error = "Distributed performance measurement requires trainer.strategy.reduce."
    summed = _capability.strategy_reduce(
        trainer,
        summed,
        reduce_op="sum",
        error=reduce_error,
    )
    maximum = _capability.strategy_reduce(
        trainer,
        maximum,
        reduce_op="max",
        error=reduce_error,
    )
    sum_values = summed.detach().cpu().tolist()
    max_values = maximum.detach().cpu().tolist()

    return _Totals(
        current_time=float(max_values[0]),
        window_time=math.fsum(float(value) for value in max_values[1:]),
        current_flops=float(sum_values[0]) if sum_values[3] == world_size else None,
        window_flops=float(sum_values[1]) if sum_values[4] == world_size else None,
        hardware_peak_flops=float(sum_values[2]) if sum_values[5] == world_size else None,
        world_size=world_size,
    )


def _infer_hardware(
    *,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    compute_dtype: torch.dtype | str | None,
    hardware_peak_flops: float | None,
) -> PeakFlops | None:
    dtype = compute_dtype
    if dtype is None:
        dtype = _dtype_from_trainer(trainer)
    device = _device(pl_module)
    try:
        peak = infer_peak_flops(
            dtype=dtype,
            device=device,
            hardware_peak_flops=hardware_peak_flops,
        )
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        return None
    if peak is None:
        warnings.warn(
            "Cannot infer hardware peak FLOPs for the current device and compute dtype; "
            "set hardware_peak_flops to enable perf/mfu.",
            RuntimeWarning,
            stacklevel=2,
        )
    return peak


def _dtype_from_trainer(trainer: pl.Trainer) -> torch.dtype | str:
    precision = str(trainer.precision).lower()
    if "bf16" in precision:
        return torch.bfloat16
    if "16" in precision:
        return torch.float16
    if "64" in precision:
        return torch.float64
    if torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return torch.float32


def _device(pl_module: pl.LightningModule) -> torch.device | None:
    try:
        return pl_module.device
    except (AttributeError, RuntimeError):
        return None


def _sync_cuda(sync_cuda: bool, pl_module: pl.LightningModule) -> None:
    device = _device(pl_module)
    if sync_cuda and device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _sync_distributed(trainer: pl.Trainer, enabled: bool) -> None:
    if not enabled or not _capability.is_distributed(trainer):
        return
    _capability.strategy_barrier(
        trainer,
        error="distributed performance measurement requires strategy.barrier.",
    )


def _reset_peak_memory(pl_module: pl.LightningModule) -> None:
    device = _device(pl_module)
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_metrics(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
) -> dict[str, float]:
    device = _device(pl_module)
    if device is None or device.type != "cuda":
        return {}
    values = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        ],
        dtype=torch.float64,
        device=device,
    )
    if _capability.is_distributed(trainer):
        values = _capability.strategy_reduce(
            trainer,
            values,
            reduce_op="max",
            error="distributed memory measurement requires trainer.strategy.reduce.",
        )
    allocated, reserved = values.detach().cpu().tolist()
    return {
        "perf/gpu_peak_memory_allocated_bytes": float(allocated),
        "perf/gpu_peak_memory_reserved_bytes": float(reserved),
    }


def _metadata(
    hardware: PeakFlops | None,
    *,
    model_flops_source: str,
) -> dict[str, str | float]:
    if hardware is None:
        return {
            "perf/model_flops_source": model_flops_source,
            "perf/hardware_peak_flops_source": "unresolved",
        }
    return {
        "perf/model_flops_source": model_flops_source,
        "perf/hardware_peak_flops_source": hardware.source,
        "perf/hardware_device_name": hardware.device_name,
        "perf/hardware_compute_dtype": hardware.dtype,
        "perf/hardware_peak_flops": hardware.flops,
    }


def _model_flops_source(
    *,
    model_flops_per_step: float | None,
    model_flops_per_batch: FlopsProvider | None,
    profile_flops: bool,
) -> str:
    if profile_flops:
        return "profile"
    if model_flops_per_batch is not None:
        return "provider"
    if model_flops_per_step is not None:
        return "static"
    return "unresolved"


def _flop_counter() -> FlopCounterMode:
    from torch.utils.flop_counter import FlopCounterMode

    return FlopCounterMode(display=False)


def _log_metadata(trainer: pl.Trainer, metadata: dict[str, str | float]) -> None:
    _capability.log_hyperparams(trainer, metadata)


def _flops(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    _require_positive(converted, name)
    return converted


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "FlopsProvider",
    "PerformanceCallback",
]
