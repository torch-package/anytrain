from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Union

import lightning.pytorch as pl
from torch import Tensor

from anytrain.module.quantization.output import QuantizeOutput
from anytrain.module.quantization.stats import (
    QuantizationUsage,
    quantization_loss_metrics,
    quantization_usage,
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
    codebook_size: int | None = None
    num_codebooks: int | None = None

    def resolved_codebook_size(self) -> int:
        value = self.codebook_size
        if value is None:
            value = getattr(self.quantizer, "codebook_size", None)
        if value is None:
            raise ValueError("QuantizationMonitor requires codebook_size or a quantizer.")
        value = int(value)
        if value <= 0:
            raise ValueError(f"codebook_size must be positive, got {value}.")
        return value

    def resolved_num_codebooks(self) -> int:
        value = self.num_codebooks
        if value is None:
            value = getattr(self.quantizer, "num_codebooks", 1)
        if value is None:
            value = 1
        value = int(value)
        if value <= 0:
            raise ValueError(f"num_codebooks must be positive, got {value}.")
        return value

    def observed_num_codebooks(self) -> int:
        configured = self.resolved_num_codebooks()
        if configured == 1:
            return 1
        if self.output.indices.ndim == 0:
            raise ValueError("multi-codebook QuantizeOutput.indices must include codebook dim.")
        observed = int(self.output.indices.shape[-1])
        if observed <= 0 or observed > configured:
            raise ValueError(
                "QuantizeOutput.indices last dimension must be in [1, num_codebooks]: "
                f"got {observed}, num_codebooks={configured}."
            )
        return observed


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
            for name, value in quantization_loss_metrics(monitor.output).items():
                values[f"{prefix}/loss/{name}"] = _reduce_scalar(trainer, value.detach())

        if "usage" in self.metrics or "active" in self.metrics:
            usage = quantization_usage(
                monitor.output,
                codebook_size=monitor.resolved_codebook_size(),
                num_codebooks=monitor.observed_num_codebooks(),
            )
            usage = _reduce_usage(trainer, usage)
            values.update(_usage_metrics(prefix, usage, self.metrics))
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
            codebook_size=monitor.resolved_codebook_size(),
            num_codebooks=monitor.observed_num_codebooks(),
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


def _usage_metrics(
    prefix: str,
    usage: QuantizationUsage,
    enabled: set[str],
) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    for book_index in range(usage.num_codebooks):
        book = f"{prefix}/book_{book_index}"
        if "usage" in enabled:
            values.update(
                {
                    f"{book}/usage/observations": usage.observations[book_index],
                    f"{book}/usage/perplexity": usage.perplexity[book_index],
                    f"{book}/usage/entropy": usage.entropy[book_index],
                    f"{book}/usage/unique_count": usage.unique_count[book_index],
                    f"{book}/usage/unique_ratio": usage.unique_ratio[book_index],
                    f"{book}/usage/dead_count": usage.dead_count[book_index],
                    f"{book}/usage/dead_ratio": usage.dead_ratio[book_index],
                }
            )
        if "active" in enabled:
            values.update(
                {
                    f"{book}/active/count": usage.active_count[book_index],
                    f"{book}/active/ratio": usage.active_ratio[book_index],
                }
            )
    return values


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


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError(f"{name} must be a non-empty string.")


__all__ = [
    "QuantizationLoggerCallback",
    "QuantizationMonitor",
    "QuantizationMonitorsFn",
]
