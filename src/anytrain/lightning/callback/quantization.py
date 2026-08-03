from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Union

import lightning.pytorch as pl
from torch import Tensor

from anytrain.module.quantization.output import QuantizeOutput
from anytrain.module.quantization.stats import (
    QuantizationUsage,
    codebook_usage,
    quantization_loss_metrics,
    quantization_usage_metrics,
    valid_indices_by_codebook,
)

from .. import _capability, experiment
from ._cadence import StepCadence
from ._naming import unique_tag_items, validate_tag_part

BatchOutputs = Union[Tensor, Mapping[str, Any], None]

QuantizationMonitorsFn = Callable[
    [BatchOutputs, Any, pl.LightningModule],
    Mapping[str, "QuantizationMonitor"],
]

CodebookUsageMonitorsFn = Callable[
    [BatchOutputs, Any, pl.LightningModule],
    Mapping[str, "CodebookUsageMonitor"],
]


@dataclass(frozen=True)
class CodebookUsageMonitor:
    indices: Tensor
    codebook_sizes: Sequence[int]
    active_codebook_mask: Tensor | None = None

    def resolved_codebook_sizes(self) -> tuple[int, ...]:
        return _codebook_sizes(self.codebook_sizes, "codebook_sizes")

    def observed_codebook_sizes(self) -> tuple[int, ...]:
        return _observed_codebook_sizes(
            self.indices,
            self.resolved_codebook_sizes(),
            source="CodebookUsageMonitor.indices",
        )


@dataclass(frozen=True)
class QuantizationMonitor:
    output: QuantizeOutput
    quantizer: object | None = None
    codebook_sizes: Sequence[int] | None = None

    def resolved_codebook_sizes(self) -> tuple[int, ...]:
        value = self.codebook_sizes
        if value is not None:
            return _codebook_sizes(value, "codebook_sizes")

        value = getattr(self.quantizer, "codebook_sizes", None)
        if value is None:
            raise ValueError("QuantizationMonitor requires codebook_sizes or a quantizer.")

        return _codebook_sizes(value, "quantizer.codebook_sizes")

    def observed_codebook_sizes(self) -> tuple[int, ...]:
        return _observed_codebook_sizes(
            self.output.indices,
            self.resolved_codebook_sizes(),
            source="QuantizeOutput.indices",
        )

    def usage_monitor(self) -> CodebookUsageMonitor:
        return CodebookUsageMonitor(
            indices=self.output.indices,
            codebook_sizes=self.observed_codebook_sizes(),
            active_codebook_mask=self.output.active_codebook_mask,
        )


class CodebookUsageLoggerCallback(pl.Callback):
    def __init__(
        self,
        monitors: CodebookUsageMonitorsFn,
        *,
        metrics: Iterable[str] = ("usage", "active"),
        every_n_steps: int | None = 100,
        tag: str = "codebook",
        log_histograms: bool = False,
    ) -> None:
        super().__init__()
        self.monitors = monitors
        self.metrics = _usage_metrics(metrics)
        self.every_n_steps = every_n_steps
        self._cadence = StepCadence(every_n_steps)
        self.tag = validate_tag_part(tag)
        self.log_histograms = log_histograms

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: BatchOutputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch_idx
        if not self.should_log(trainer, pl_module):
            return

        monitors = self.monitors(outputs, batch, pl_module)
        if not isinstance(monitors, Mapping):
            raise TypeError("CodebookUsageLoggerCallback monitors must return a mapping.")

        values: dict[str, Tensor] = {}
        for name, monitor in unique_tag_items(monitors.items()):
            monitor = _validate_usage_monitor(name, monitor)
            prefix = f"{self.tag}/{name}"
            values.update(_usage_scalar_metrics(trainer, prefix, monitor, self.metrics))
            if self.log_histograms:
                _log_usage_histograms(trainer, prefix, monitor)

        if values:
            pl_module.log_dict(
                values,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )

    def should_log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> bool:
        del pl_module
        return self._cadence.should_log(trainer)


class QuantizationLoggerCallback(pl.Callback):
    def __init__(
        self,
        monitors: QuantizationMonitorsFn,
        *,
        metrics: Iterable[str] = ("loss", "usage", "active"),
        every_n_steps: int | None = 100,
        tag: str = "quant",
        log_histograms: bool = False,
    ) -> None:
        super().__init__()
        self.monitors = monitors
        self.metrics = _metrics(metrics)
        self.every_n_steps = every_n_steps
        self._cadence = StepCadence(every_n_steps)
        self.tag = validate_tag_part(tag)
        self.log_histograms = log_histograms

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: BatchOutputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch_idx
        if not self.should_log(trainer, pl_module):
            return

        monitors = self.monitors(outputs, batch, pl_module)
        if not isinstance(monitors, Mapping):
            raise TypeError("QuantizationLoggerCallback monitors must return a mapping.")

        values: dict[str, Tensor] = {}
        for name, monitor in unique_tag_items(monitors.items()):
            monitor = _validate_monitor(name, monitor)
            prefix = f"{self.tag}/{name}"
            values.update(self._scalar_metrics(trainer, prefix, monitor))
            if self.log_histograms:
                self._log_histograms(trainer, prefix, monitor)

        if values:
            pl_module.log_dict(
                values,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )

    def should_log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> bool:
        del pl_module
        return self._cadence.should_log(trainer)

    def _scalar_metrics(
        self,
        trainer: pl.Trainer,
        prefix: str,
        monitor: QuantizationMonitor,
    ) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        if "loss" in self.metrics:
            for name, value in quantization_loss_metrics(
                monitor.output,
                prefix="loss",
            ).items():
                values[f"{prefix}/{name}"] = _reduce_scalar(trainer, value.detach())

        usage_metrics = self.metrics & {"usage", "active"}
        if usage_metrics:
            values.update(
                _usage_scalar_metrics(
                    trainer,
                    prefix,
                    monitor.usage_monitor(),
                    usage_metrics,
                    distributed_error=(
                        "Quantization usage logging in distributed training requires "
                        "trainer.strategy.reduce."
                    ),
                )
            )
        return values

    def _log_histograms(
        self,
        trainer: pl.Trainer,
        prefix: str,
        monitor: QuantizationMonitor,
    ) -> None:
        _log_usage_histograms(trainer, prefix, monitor.usage_monitor())


def _usage_scalar_metrics(
    trainer: pl.Trainer,
    prefix: str,
    monitor: CodebookUsageMonitor,
    metrics: Iterable[str],
    *,
    distributed_error: str = (
        "Codebook usage logging in distributed training requires trainer.strategy.reduce."
    ),
) -> dict[str, Tensor]:
    usage = codebook_usage(
        monitor.indices,
        codebook_sizes=monitor.observed_codebook_sizes(),
        active_codebook_mask=monitor.active_codebook_mask,
    )
    usage = _reduce_usage(trainer, usage, error=distributed_error)
    return {
        f"{prefix}/{name}": value
        for name, value in quantization_usage_metrics(usage, metrics=metrics).items()
    }


def _log_usage_histograms(
    trainer: pl.Trainer,
    prefix: str,
    monitor: CodebookUsageMonitor,
) -> None:
    if not _capability.is_global_zero(trainer):
        return
    writer = experiment.histogram(trainer)
    if writer is None:
        return
    step = _capability.global_step(trainer)
    indices_by_book = valid_indices_by_codebook(
        monitor.indices,
        codebook_sizes=monitor.observed_codebook_sizes(),
        active_codebook_mask=monitor.active_codebook_mask,
    )
    for book_index, indices in enumerate(indices_by_book):
        if indices.numel() == 0:
            continue
        writer.add_histogram(
            f"{prefix}/book_{book_index}/indices",
            indices.detach().float().cpu(),
            step,
        )


def _reduce_usage(
    trainer: pl.Trainer,
    usage: QuantizationUsage,
    *,
    error: str,
) -> QuantizationUsage:
    if not _capability.is_distributed(trainer):
        return usage
    return QuantizationUsage(
        counts=_capability.strategy_reduce(
            trainer,
            usage.counts,
            reduce_op="sum",
            error=error,
        ),
        active_count=_capability.strategy_reduce(
            trainer,
            usage.active_count,
            reduce_op="sum",
            error=error,
        ),
        total_count=_capability.strategy_reduce(
            trainer,
            usage.total_count,
            reduce_op="sum",
            error=error,
        ),
        codebook_sizes=usage.codebook_sizes,
    )


def _reduce_scalar(trainer: pl.Trainer, value: Tensor) -> Tensor:
    if not _capability.is_distributed(trainer):
        return value
    return _capability.strategy_reduce(
        trainer,
        value,
        reduce_op="mean",
        error="Quantization scalar logging in distributed training requires trainer.strategy.reduce.",
    )


def _validate_usage_monitor(name: object, monitor: object) -> CodebookUsageMonitor:
    _non_empty_string(name, "monitor name")
    if not isinstance(monitor, CodebookUsageMonitor):
        raise TypeError(
            "CodebookUsageLoggerCallback monitor values must be CodebookUsageMonitor."
        )
    return monitor


def _validate_monitor(name: object, monitor: object) -> QuantizationMonitor:
    _non_empty_string(name, "monitor name")
    if not isinstance(monitor, QuantizationMonitor):
        raise TypeError("QuantizationLoggerCallback monitor values must be QuantizationMonitor.")
    return monitor


def _metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"loss", "usage", "active"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown quantization metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("metrics must contain at least one value.")
    return result


def _usage_metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"usage", "active"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown codebook usage metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("metrics must contain at least one value.")
    return result


def _codebook_sizes(values: Sequence[int], name: str) -> tuple[int, ...]:
    sizes = tuple(_positive_int_value(size, name) for size in values)
    if not sizes:
        raise ValueError(f"{name} must not be empty.")
    return sizes


def _observed_codebook_sizes(
    indices: Tensor,
    sizes: tuple[int, ...],
    *,
    source: str,
) -> tuple[int, ...]:
    configured = len(sizes)
    if configured == 1:
        return sizes
    if indices.ndim == 0:
        raise ValueError(f"multi-codebook {source} must include codebook dim.")
    observed = int(indices.shape[-1])
    if observed <= 0 or observed > configured:
        raise ValueError(
            f"{source} last dimension must be in [1, num_codebooks]: "
            f"got {observed}, num_codebooks={configured}."
        )
    return sizes[:observed]


def _positive_int_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError(f"{name} must be a non-empty string.")


__all__ = [
    "CodebookUsageLoggerCallback",
    "CodebookUsageMonitor",
    "CodebookUsageMonitorsFn",
    "QuantizationLoggerCallback",
    "QuantizationMonitor",
    "QuantizationMonitorsFn",
]
