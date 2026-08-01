from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Union

import lightning.pytorch as pl
from torch import Tensor

from anytrain.module.moe import (
    MoEOutput,
    SparseMoELossOutput,
    sparse_moe_loss_metrics,
    sparse_moe_routing_metrics,
)
from anytrain.module.moe.routing import SparseExpertRouting

from .. import _capability

SparseMoEMonitorsFn = Callable[
    [Union[Tensor, Mapping[str, Any], None], Any, pl.LightningModule],
    Mapping[str, "SparseMoEMonitor"],
]


@dataclass(frozen=True)
class SparseMoEMonitor:
    source: SparseExpertRouting | MoEOutput
    loss: SparseMoELossOutput | None = None

    def routing(self) -> SparseExpertRouting:
        if isinstance(self.source, SparseExpertRouting):
            return self.source
        if isinstance(self.source, MoEOutput):
            return self.source.routing
        raise TypeError("SparseMoEMonitor source must be SparseExpertRouting or MoEOutput.")

    def tokens_per_expert(self) -> Tensor | None:
        if isinstance(self.source, MoEOutput):
            return self.source.tokens_per_expert
        return None

    def dropped_mask(self) -> Tensor | None:
        if isinstance(self.source, MoEOutput):
            return self.source.dropped_mask
        return None


class SparseMoELoggerCallback(pl.Callback):
    def __init__(
        self,
        monitors: SparseMoEMonitorsFn,
        *,
        metrics: Iterable[str] = ("loss", "usage", "probability", "tokens", "dropped"),
        every_n_steps: int | None = 100,
        tag: str = "moe",
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        _non_empty_string(tag, "tag")
        self.monitors = monitors
        self.metrics = _metrics(metrics)
        self.every_n_steps = every_n_steps
        self.tag = tag.strip("/")
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
            raise TypeError("SparseMoELoggerCallback monitors must return a mapping.")

        values: dict[str, Tensor] = {}
        for name, monitor in monitors.items():
            monitor = _validate_monitor(name, monitor)
            prefix = f"{self.tag}/{_tag_part(name)}"
            values.update(self._scalar_metrics(prefix, monitor))

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

    def _scalar_metrics(self, prefix: str, monitor: SparseMoEMonitor) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        routing_metrics = self.metrics - {"loss"}
        if routing_metrics:
            for name, value in sparse_moe_routing_metrics(
                monitor.routing(),
                tokens=monitor.tokens_per_expert(),
                dropped_mask=monitor.dropped_mask(),
                metrics=routing_metrics,
            ).items():
                values[f"{prefix}/{name}"] = value.detach()

        if "loss" in self.metrics and monitor.loss is not None:
            for name, value in sparse_moe_loss_metrics(monitor.loss, prefix="loss").items():
                values[f"{prefix}/{name}"] = value.detach()
        return values


def _validate_monitor(name: object, monitor: object) -> SparseMoEMonitor:
    _non_empty_string(name, "monitor name")
    if not isinstance(monitor, SparseMoEMonitor):
        raise TypeError("SparseMoELoggerCallback monitor values must be SparseMoEMonitor.")
    return monitor


def _metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"loss", "usage", "probability", "tokens", "dropped"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown sparse MoE metrics: {sorted(unknown)!r}.")
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
    "SparseMoELoggerCallback",
    "SparseMoEMonitor",
    "SparseMoEMonitorsFn",
]
