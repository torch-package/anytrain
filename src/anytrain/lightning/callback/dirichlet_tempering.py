from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Union, cast

import lightning.pytorch as pl
from torch import Tensor

from anytrain.module.dirichlet_tempering import (
    AdaptiveDirichletTempering,
    ADTStats,
    adt_stats_metrics,
)

from .. import _capability, experiment
from ._naming import join_tag, module_monitor_names, validate_tag_part

ADTStatsSource = Union[AdaptiveDirichletTempering, ADTStats]
ADTStatsMonitorLike = Union["ADTStatsMonitor", ADTStatsSource]
ADTStatsMonitors = Union[
    Mapping[str, ADTStatsMonitorLike],
    ADTStatsMonitorLike,
    Iterable[ADTStatsMonitorLike],
]
ADTStatsMonitorsFn = Callable[
    [Union[Tensor, Mapping[str, Any], None], Any, pl.LightningModule],
    ADTStatsMonitors,
]


@dataclass(frozen=True)
class ADTStatsMonitor:
    source: ADTStatsSource
    name: str | None = None

    def stats(self) -> ADTStats:
        if isinstance(self.source, ADTStats):
            return self.source
        if isinstance(self.source, AdaptiveDirichletTempering):
            return self.source.monitor
        raise TypeError(
            "ADTStatsMonitor source must be AdaptiveDirichletTempering or ADTStats."
        )


class ADTStatsLoggerCallback(pl.Callback):
    def __init__(
        self,
        monitors: ADTStatsMonitorsFn | None = None,
        *,
        metrics: Iterable[str] = ("variance", "alpha", "temperature"),
        every_n_steps: int | None = 100,
        tag: str = "adt",
        log_histograms: bool = False,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        self.monitors = monitors
        self.metrics = _metrics(metrics)
        self.every_n_steps = every_n_steps
        self.tag = validate_tag_part(tag)
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
            monitors = _discover_monitors(pl_module)
        else:
            monitors = _normalize_monitors(
                self.monitors(outputs, batch, pl_module),
                pl_module,
            )

        values: dict[str, Tensor] = {}
        for name, monitor in monitors:
            prefix = join_tag(self.tag, name)
            stats = monitor.stats()
            values.update(self._scalar_metrics(prefix, stats))
            if self.log_histograms:
                self._log_histograms(trainer, prefix, stats)

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

    def _scalar_metrics(self, prefix: str, stats: ADTStats) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        for name, value in adt_stats_metrics(stats, metrics=self.metrics).items():
            values[join_tag(prefix, name)] = value.detach()
        return values

    def _log_histograms(
        self,
        trainer: pl.Trainer,
        prefix: str,
        stats: ADTStats,
    ) -> None:
        if not _capability.is_global_zero(trainer):
            return
        writer = experiment.histogram(trainer)
        if writer is None:
            return
        step = _capability.global_step(trainer)
        for name, values in _histogram_values(stats, self.metrics).items():
            writer.add_histogram(
                join_tag(prefix, name),
                values.detach().float().cpu(),
                step,
            )


def _normalize_monitors(
    monitors: object,
    pl_module: pl.LightningModule,
) -> list[tuple[str, ADTStatsMonitor]]:
    module_names = module_monitor_names(pl_module, AdaptiveDirichletTempering)
    if isinstance(monitors, Mapping):
        return [
            (
                _monitor_name(
                    monitor,
                    explicit_name=name,
                    module_names=module_names,
                    index=index,
                ),
                monitor,
            )
            for index, (name, monitor) in enumerate(
                (key, _as_monitor(value)) for key, value in monitors.items()
            )
        ]

    if _is_monitor_like(monitors):
        monitor = _as_monitor(monitors)
        return [(_monitor_name(monitor, module_names=module_names, index=0), monitor)]

    if isinstance(monitors, Iterable) and not isinstance(monitors, (str, bytes)):
        result: list[tuple[str, ADTStatsMonitor]] = []
        for index, value in enumerate(monitors):
            monitor = _as_monitor(value)
            result.append(
                (
                    _monitor_name(monitor, module_names=module_names, index=index),
                    monitor,
                )
            )
        return result

    raise TypeError(
        "ADTStatsLoggerCallback monitors must return an ADTStatsMonitor, "
        "AdaptiveDirichletTempering, ADTStats, an iterable of those, or a mapping."
    )


def _discover_monitors(
    pl_module: pl.LightningModule,
) -> list[tuple[str, ADTStatsMonitor]]:
    named_modules = getattr(pl_module, "named_modules", None)
    if not callable(named_modules):
        return []

    module_names = module_monitor_names(pl_module, AdaptiveDirichletTempering)
    result: list[tuple[str, ADTStatsMonitor]] = []
    seen: set[int] = set()
    modules = cast(Iterable[tuple[str, object]], named_modules())
    for name, module in modules:
        if isinstance(module, AdaptiveDirichletTempering):
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            monitor_name = module_names.get(
                module_id,
                validate_tag_part(name or "module"),
            )
            result.append((monitor_name, ADTStatsMonitor(module, name=monitor_name)))
    return result


def _as_monitor(value: object) -> ADTStatsMonitor:
    if isinstance(value, ADTStatsMonitor):
        return value
    if isinstance(value, (AdaptiveDirichletTempering, ADTStats)):
        return ADTStatsMonitor(value)
    raise TypeError(
        "ADTStatsLoggerCallback monitor entries must be ADTStatsMonitor, "
        "AdaptiveDirichletTempering, or ADTStats."
    )


def _is_monitor_like(value: object) -> bool:
    return isinstance(value, (ADTStatsMonitor, AdaptiveDirichletTempering, ADTStats))


def _monitor_name(
    monitor: ADTStatsMonitor,
    *,
    module_names: Mapping[int, str],
    index: int,
    explicit_name: object | None = None,
) -> str:
    if monitor.name is not None:
        return validate_tag_part(monitor.name, "monitor name")
    if explicit_name is not None:
        return validate_tag_part(explicit_name, "monitor name")
    if isinstance(monitor.source, AdaptiveDirichletTempering):
        module_name = module_names.get(id(monitor.source))
        if module_name is not None:
            return module_name
    return f"monitor_{index}"


def _metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"variance", "alpha", "temperature"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown ADT-only stats metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("metrics must contain at least one value.")
    return result


def _histogram_values(stats: ADTStats, metrics: Iterable[str]) -> dict[str, Tensor]:
    enabled = _metrics(metrics)
    values: dict[str, Tensor] = {}
    if "usage" in enabled:
        values["usage"] = stats.expert_utilization
    if "variance" in enabled:
        values["variance"] = stats.expert_vars
    if "alpha" in enabled:
        values["alpha"] = stats.alpha
    if "temperature" in enabled:
        values["temperature"] = stats.temperature
    return values


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "ADTStatsLoggerCallback",
    "ADTStatsMonitor",
    "ADTStatsMonitorsFn",
]
