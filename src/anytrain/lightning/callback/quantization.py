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
    quantization_loss_metrics,
    quantization_usage,
    quantization_usage_metrics,
    valid_indices_by_codebook,
)

from .. import _capability, experiment

QuantizationMonitorsFn = Callable[
    [Union[Tensor, Mapping[str, Any], None], Any, pl.LightningModule],
    Mapping[str, "QuantizationMonitor"],
]


@dataclass(frozen=True)
class QuantizationMonitor:
    output: QuantizeOutput
    quantizer: object | None = None
    codebook_sizes: Sequence[int] | None = None

    def resolved_codebook_sizes(self) -> tuple[int, ...]:
        value = self.codebook_sizes
        if value is not None:
            sizes = tuple(_positive_int_value(size, "codebook_sizes") for size in value)
            if not sizes:
                raise ValueError("codebook_sizes must not be empty.")
            return sizes

        value = getattr(self.quantizer, "codebook_sizes", None)
        if value is None:
            raise ValueError("QuantizationMonitor requires codebook_sizes or a quantizer.")

        sizes = tuple(_positive_int_value(size, "quantizer.codebook_sizes") for size in value)
        if not sizes:
            raise ValueError("quantizer.codebook_sizes must not be empty.")
        return sizes

    def observed_codebook_sizes(self) -> tuple[int, ...]:
        sizes = self.resolved_codebook_sizes()
        configured = len(sizes)
        if configured == 1:
            return sizes
        if self.output.indices.ndim == 0:
            raise ValueError("multi-codebook QuantizeOutput.indices must include codebook dim.")
        observed = int(self.output.indices.shape[-1])
        if observed <= 0 or observed > configured:
            raise ValueError(
                "QuantizeOutput.indices last dimension must be in [1, num_codebooks]: "
                f"got {observed}, num_codebooks={configured}."
            )
        return sizes[:observed]


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
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        _non_empty_string(tag, "tag")
        self.monitors = monitors
        self.metrics = _metrics(metrics)
        self.every_n_steps = every_n_steps
        self.tag = tag.strip("/")
        self.log_histograms = log_histograms
        self._last_logged_step: int | None = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
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
        for name, monitor in monitors.items():
            monitor = _validate_monitor(name, monitor)
            prefix = f"{self.tag}/{_tag_part(name)}"
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
        if self.every_n_steps is None:
            return True
        step = _capability.global_step(trainer)
        if step % self.every_n_steps != 0:
            return False
        if self._last_logged_step == step:
            return False
        self._last_logged_step = step
        return True

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
            usage = quantization_usage(
                monitor.output,
                codebook_sizes=monitor.observed_codebook_sizes(),
            )
            usage = _reduce_usage(trainer, usage)
            for name, value in quantization_usage_metrics(
                usage,
                metrics=usage_metrics,
            ).items():
                values[f"{prefix}/{name}"] = value
        return values

    def _log_histograms(
        self,
        trainer: pl.Trainer,
        prefix: str,
        monitor: QuantizationMonitor,
    ) -> None:
        if not _capability.is_global_zero(trainer):
            return
        writer = experiment.histogram(trainer)
        if writer is None:
            return
        step = _capability.global_step(trainer)
        indices_by_book = valid_indices_by_codebook(
            monitor.output.indices,
            codebook_sizes=monitor.observed_codebook_sizes(),
            active_codebook_mask=monitor.output.active_codebook_mask,
        )
        for book_index, indices in enumerate(indices_by_book):
            if indices.numel() == 0:
                continue
            writer.add_histogram(
                f"{prefix}/book_{book_index}/indices",
                indices.detach().float().cpu(),
                step,
            )


def _reduce_usage(trainer: pl.Trainer, usage: QuantizationUsage) -> QuantizationUsage:
    if not _capability.is_distributed(trainer):
        return usage
    error = "Quantization usage logging in distributed training requires trainer.strategy.reduce."
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


def _tag_part(value: object) -> str:
    text = str(value).strip("/")
    _non_empty_string(text, "monitor name")
    return text


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


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
    "QuantizationLoggerCallback",
    "QuantizationMonitor",
    "QuantizationMonitorsFn",
]
