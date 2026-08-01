from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Union

import lightning.pytorch as pl
from torch import Tensor

from anytrain.module.dirichlet_tempering import (
    AdaptiveDirichletTempering,
    adt_stats_metrics,
)
from anytrain.module.moe import (
    MoE,
    MoELossOutput,
    MoEOutput,
    expert_probability_mean,
    expert_route_weight_mean,
    expert_usage,
    moe_routing_metrics,
    tokens_per_expert,
)
from anytrain.module.moe.routing import ExpertRouting

from .. import _capability, experiment
from ._naming import join_tag, module_monitor_names, validate_tag_part

MoEMonitorsFn = Callable[
    [Union[Tensor, Mapping[str, Any], None], Any, pl.LightningModule],
    Mapping[str, "MoEMonitor"],
]


@dataclass(frozen=True)
class MoEMonitor:
    source: ExpertRouting | MoEOutput
    loss: MoELossOutput | None = None
    module: MoE | None = None

    def routing(self) -> ExpertRouting:
        if isinstance(self.source, ExpertRouting):
            return self.source
        if isinstance(self.source, MoEOutput):
            return self.source.routing
        raise TypeError("MoEMonitor source must be ExpertRouting or MoEOutput.")

    def tokens_per_expert(self) -> Tensor | None:
        if isinstance(self.source, MoEOutput):
            return self.source.tokens_per_expert
        return None

    def dropped_mask(self) -> Tensor | None:
        if isinstance(self.source, MoEOutput):
            return self.source.dropped_mask
        return None


class MoELoggerCallback(pl.Callback):
    def __init__(
        self,
        monitors: MoEMonitorsFn | None = None,
        *,
        metrics: Iterable[str] = (
            "loss",
            "usage",
            "gate_probability",
            "route_weight",
            "tokens",
            "dropped",
        ),
        adt_metrics: Iterable[str] = ("variance", "alpha", "temperature"),
        every_n_steps: int | None = 100,
        tag: str = "moe",
        log_histograms: bool = False,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        validate_tag_part(tag)
        self.monitors = monitors
        self.metrics = _metrics(metrics)
        self.adt_metrics = _adt_metric_groups(adt_metrics)
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

        if self.monitors is None:
            monitors = _discover_monitors(outputs)
        else:
            monitors = self.monitors(outputs, batch, pl_module)
            if not isinstance(monitors, Mapping):
                raise TypeError("MoELoggerCallback monitors must return a mapping.")

        moe_modules = _moe_modules_by_monitor_name(pl_module)
        values: dict[str, Tensor] = {}
        for name, monitor in monitors.items():
            monitor = _validate_monitor(name, monitor)
            monitor_name = validate_tag_part(name, "monitor name")
            prefix = join_tag(self.tag, monitor_name)
            moe_module = monitor.module or moe_modules.get(monitor_name)
            values.update(self._scalar_metrics(prefix, monitor, moe_module=moe_module))
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
        prefix: str,
        monitor: MoEMonitor,
        *,
        moe_module: MoE | None,
    ) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        routing_metrics = self.metrics - {"loss"}
        if routing_metrics:
            for name, value in moe_routing_metrics(
                monitor.routing(),
                tokens=monitor.tokens_per_expert(),
                dropped_mask=monitor.dropped_mask(),
                metrics=routing_metrics,
            ).items():
                values[f"{prefix}/{name}"] = value.detach()

        if "loss" in self.metrics and monitor.loss is not None:
            for name, value in _loss_metrics(monitor.loss, prefix="loss").items():
                values[f"{prefix}/{name}"] = value.detach()
        values.update(self._adt_metrics(prefix, moe_module))
        return values

    def _adt_metrics(self, prefix: str, moe_module: MoE | None) -> dict[str, Tensor]:
        if moe_module is None:
            return {}
        gate = getattr(moe_module, "gate", None)
        if not isinstance(gate, AdaptiveDirichletTempering):
            return {}
        return {
            join_tag(prefix, "adt", name): value.detach()
            for name, value in adt_stats_metrics(
                gate.monitor,
                metrics=self.adt_metrics,
            ).items()
        }

    def _log_histograms(
        self,
        trainer: pl.Trainer,
        prefix: str,
        monitor: MoEMonitor,
    ) -> None:
        if not _capability.is_global_zero(trainer):
            return
        writer = experiment.histogram(trainer)
        if writer is None:
            return
        step = _capability.global_step(trainer)
        for name, values in _histogram_values(monitor, self.metrics).items():
            writer.add_histogram(
                join_tag(prefix, name),
                values.detach().float().cpu(),
                step,
            )


def _loss_metrics(loss: MoELossOutput, *, prefix: str = "loss") -> dict[str, Tensor]:
    prefix = prefix.strip("/")
    if prefix:
        prefix = f"{prefix}/"
    return {
        f"{prefix}total": loss.total.detach(),
        f"{prefix}balance": loss.balance_loss.detach(),
        f"{prefix}z": loss.z_loss.detach(),
        f"{prefix}weighted_balance": loss.weighted_balance.detach(),
        f"{prefix}weighted_z": loss.weighted_z.detach(),
    }


def _validate_monitor(name: object, monitor: object) -> MoEMonitor:
    validate_tag_part(name, "monitor name")
    if not isinstance(monitor, MoEMonitor):
        raise TypeError("MoELoggerCallback monitor values must be MoEMonitor.")
    if monitor.module is not None and not isinstance(monitor.module, MoE):
        raise TypeError("MoEMonitor.module must be MoE or None.")
    return monitor


def _discover_monitors(outputs: object) -> dict[str, MoEMonitor]:
    if isinstance(outputs, MoEMonitor):
        return {"module": outputs}
    if isinstance(outputs, (MoEOutput, ExpertRouting)):
        return {"module": MoEMonitor(outputs)}
    if not isinstance(outputs, Mapping):
        return {}

    result: dict[str, MoEMonitor] = {}
    for key, value in outputs.items():
        name = validate_tag_part(key, "monitor name")
        if isinstance(value, MoEMonitor):
            result[name] = value
        elif isinstance(value, (MoEOutput, ExpertRouting)):
            result[name] = MoEMonitor(value, loss=_matching_loss(outputs, name))
    return result


def _matching_loss(outputs: Mapping[str, object], name: str) -> MoELossOutput | None:
    candidates = (
        f"{name}_loss",
        f"{name}_aux_loss",
        f"{name}_moe_loss",
        f"{name}/loss",
    )
    for candidate in candidates:
        value = outputs.get(candidate)
        if isinstance(value, MoELossOutput):
            return value
    return None


def _moe_modules_by_monitor_name(pl_module: pl.LightningModule) -> dict[str, MoE]:
    module_names = module_monitor_names(pl_module, MoE)
    modules_by_id = _moe_modules_by_id(pl_module)
    return {
        name: modules_by_id[module_id]
        for module_id, name in module_names.items()
        if module_id in modules_by_id
    }


def _moe_modules_by_id(pl_module: pl.LightningModule) -> dict[int, MoE]:
    named_modules = getattr(pl_module, "named_modules", None)
    if not callable(named_modules):
        return {}
    result: dict[int, MoE] = {}
    for _name, module in named_modules():
        if isinstance(module, MoE):
            result.setdefault(id(module), module)
    return result


def _metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"loss", "usage", "gate_probability", "route_weight", "tokens", "dropped"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown MoE metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("metrics must contain at least one value.")
    return result


def _adt_metric_groups(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(value.strip() for value in values)
    allowed = {"variance", "alpha", "temperature"}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"unknown MoE ADT-only metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("adt_metrics must contain at least one value.")
    return result


def _histogram_values(monitor: MoEMonitor, metrics: Iterable[str]) -> dict[str, Tensor]:
    enabled = _metrics(metrics)
    routing = monitor.routing()
    values: dict[str, Tensor] = {}
    if "usage" in enabled:
        values["usage"] = expert_usage(routing)
    if "gate_probability" in enabled:
        values["gate_probability"] = expert_probability_mean(routing)
    if "route_weight" in enabled:
        values["route_weight"] = expert_route_weight_mean(routing)
    if "tokens" in enabled:
        assignment_counts = monitor.tokens_per_expert()
        values["tokens"] = (
            assignment_counts if assignment_counts is not None else tokens_per_expert(routing)
        )
    return values


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")

__all__ = [
    "MoELoggerCallback",
    "MoEMonitor",
    "MoEMonitorsFn",
]
