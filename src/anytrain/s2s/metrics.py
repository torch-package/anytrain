from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from anytrain.lightning import StepInterval, _capability, experiment

from .data import GenerationRequest, Route
from .model import GenerationOutput, LossStat, TokenLossStats
from .training import REPORT_KEY, BatchUnits, StepReport

_GLOBAL_FIELDS = 6
_ROUTE_FIELDS = 13


def generation_metrics(
    request: GenerationRequest,
    output: GenerationOutput,
) -> dict[str, float]:
    """Return backend-neutral latency metrics for one semantic generation."""

    if not isinstance(request, GenerationRequest):
        raise TypeError("generation metrics request must be GenerationRequest.")
    if not isinstance(output, GenerationOutput):
        raise TypeError("generation metrics output must be GenerationOutput.")
    prefix = f"generation/{request.route.tag}"
    values = {
        f"{prefix}/count": 1.0,
        f"{prefix}/status/{output.status.value}": 1.0,
    }
    if output.error is not None:
        values[f"{prefix}/error/{output.error.code}"] = 1.0
    timing = output.timing
    if timing is None:
        return values
    values[f"{prefix}/total_seconds"] = timing.total_seconds
    input_audio = request.source.audio
    if input_audio is not None:
        values[f"{prefix}/input_seconds"] = input_audio.duration_seconds
        values[f"{prefix}/rtf"] = timing.total_seconds / input_audio.duration_seconds
    for stage in timing.stages:
        values[f"{prefix}/stage/{stage.name}_seconds"] = stage.seconds
        if input_audio is not None:
            values[f"{prefix}/stage/{stage.name}_rtf"] = (
                stage.seconds / input_audio.duration_seconds
            )
    return values


class S2SMetricsCallback(Callback):
    """Accumulate exact S2S sums/counts and log one reduced metric window."""

    def __init__(
        self,
        routes: Sequence[Route],
        *,
        every_n_steps: int,
        sync_distributed: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(routes, (str, bytes)) or not isinstance(routes, Sequence):
            raise TypeError("S2S metric routes must be a sequence of Route values.")
        values = tuple(routes)
        if not values:
            raise ValueError("S2S metric routes must not be empty.")
        if any(not isinstance(route, Route) for route in values):
            raise TypeError("S2S metric routes must contain only Route values.")
        if len(values) != len(set(values)):
            raise ValueError("S2S metric routes must be unique.")
        if not isinstance(sync_distributed, bool):
            raise TypeError("sync_distributed must be a boolean.")
        self.routes = values
        self.interval = StepInterval(every_n_steps=every_n_steps)
        self.sync_distributed = sync_distributed
        self._window = _Window(values)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        if not isinstance(outputs, Mapping):
            return
        report = outputs.get(REPORT_KEY)
        if report is None:
            return
        if not isinstance(report, StepReport):
            raise TypeError(f"training output {REPORT_KEY!r} must be StepReport.")
        self._window.update(report)
        step = _capability.global_step(trainer)
        if step > 0 and self.interval.should_run(step):
            self._flush(trainer, step=step)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self._window:
            self._flush(trainer, step=_capability.global_step(trainer))

    def _flush(self, trainer: Trainer, *, step: int) -> None:
        packed = self._window.pack()
        if self.sync_distributed and _capability.world_size(trainer) > 1:
            packed = _capability.strategy_reduce(
                trainer,
                packed,
                reduce_op="sum",
                error="Distributed S2S metrics require trainer.strategy.reduce.",
            )
        if _capability.is_global_zero(trainer):
            writer = experiment.scalar(trainer)
            if writer is None:
                raise RuntimeError(
                    "S2SMetricsCallback requires a logger with add_scalar capability."
                )
            for tag, value in _window_metrics(packed, self.routes).items():
                writer.add_scalar(tag, value, step)
        self._window.reset()


class _RouteWindow:
    def __init__(self) -> None:
        self.units: BatchUnits | None = None
        self.loss: TokenLossStats | None = None
        self.loss_exact = True

    def update(self, units: BatchUnits, loss: TokenLossStats | None) -> None:
        self.units = units if self.units is None else self.units + units
        if not self.loss_exact:
            return
        if loss is None:
            self.loss = None
            self.loss_exact = False
            return
        self.loss = loss if self.loss is None else self.loss + loss


class _Window:
    def __init__(self, routes: tuple[Route, ...]) -> None:
        self.routes = routes
        self.loss: TokenLossStats | None = None
        self.route = {route: _RouteWindow() for route in routes}

    def __bool__(self) -> bool:
        return self.loss is not None

    def update(self, report: StepReport) -> None:
        unknown = tuple(value.route for value in report.routes if value.route not in self.route)
        if unknown:
            names = ", ".join(route.tag for route in unknown)
            raise ValueError(f"S2S metrics observed unconfigured routes: {names}.")
        self.loss = report.loss if self.loss is None else self.loss + report.loss
        for value in report.routes:
            self.route[value.route].update(value.units, value.loss)

    def reset(self) -> None:
        self.loss = None
        self.route = {route: _RouteWindow() for route in self.routes}

    def pack(self) -> Tensor:
        if self.loss is None:
            raise RuntimeError("cannot pack an empty S2S metric window.")
        reference = self.loss.total.loss_sum
        values = _loss_values(self.loss, reference)
        for route in self.routes:
            window = self.route[route]
            values.extend(_unit_values(window.units, reference))
            values.extend(_loss_values(window.loss, reference))
            values.append(_inexact_value(window.loss_exact, reference))
        return torch.stack(values)


def _unit_values(units: BatchUnits | None, reference: Tensor) -> list[Tensor]:
    if units is None:
        return [_zero(reference) for _ in range(6)]
    return [
        units.rows.to(device=reference.device, dtype=torch.float64),
        units.input_tokens.to(device=reference.device, dtype=torch.float64),
        units.padded_tokens.to(device=reference.device, dtype=torch.float64),
        units.text_tokens.to(device=reference.device, dtype=torch.float64),
        units.audio_tokens.to(device=reference.device, dtype=torch.float64),
        units.audio_seconds.to(device=reference.device, dtype=torch.float64),
    ]


def _loss_values(stats: TokenLossStats | None, reference: Tensor) -> list[Tensor]:
    if stats is None:
        return [_zero(reference) for _ in range(6)]
    values: list[Tensor] = []
    for stat in (stats.total, stats.text, stats.audio):
        values.extend(_stat_values(stat, reference))
    return values


def _stat_values(stat: LossStat | None, reference: Tensor) -> list[Tensor]:
    if stat is None:
        return [_zero(reference), _zero(reference)]
    return [
        stat.loss_sum.detach().to(device=reference.device, dtype=torch.float64),
        stat.count.detach().to(device=reference.device, dtype=torch.float64),
    ]


def _zero(reference: Tensor) -> Tensor:
    return torch.zeros((), device=reference.device, dtype=torch.float64)


def _inexact_value(loss_exact: bool, reference: Tensor) -> Tensor:
    return torch.tensor(
        0.0 if loss_exact else 1.0,
        device=reference.device,
        dtype=torch.float64,
    )


def _window_metrics(packed: Tensor, routes: tuple[Route, ...]) -> dict[str, float]:
    expected = _GLOBAL_FIELDS + len(routes) * _ROUTE_FIELDS
    if packed.ndim != 1 or packed.numel() != expected:
        raise ValueError(f"packed S2S metrics must have shape [{expected}].")
    values = tuple(float(value) for value in packed.detach().cpu().tolist())
    metrics: dict[str, float] = {}
    _add_losses(metrics, "loss/train", values[:_GLOBAL_FIELDS])

    rows_by_task: dict[str, float] = {}
    route_values: list[tuple[Route, tuple[float, ...]]] = []
    offset = _GLOBAL_FIELDS
    for route in routes:
        current = values[offset : offset + _ROUTE_FIELDS]
        offset += _ROUTE_FIELDS
        route_values.append((route, current))
        rows_by_task[route.task] = rows_by_task.get(route.task, 0.0) + current[0]
    total_rows = sum(rows_by_task.values())

    for task, rows in rows_by_task.items():
        if total_rows > 0:
            metrics[f"data/train/task_fraction/{task}"] = rows / total_rows
    for route, current in route_values:
        rows, input_tokens, padded_tokens, text_tokens, audio_tokens, audio_seconds = current[:6]
        supervised_tokens = text_tokens + audio_tokens
        tag = route.tag
        metrics[f"data/train/rows/{tag}"] = rows
        metrics[f"data/train/input_tokens/{tag}"] = input_tokens
        metrics[f"data/train/padded_tokens/{tag}"] = padded_tokens
        metrics[f"data/train/supervised_tokens/{tag}"] = supervised_tokens
        metrics[f"data/train/text_tokens/{tag}"] = text_tokens
        metrics[f"data/train/audio_tokens/{tag}"] = audio_tokens
        metrics[f"data/train/audio_seconds/{tag}"] = audio_seconds
        if total_rows > 0:
            metrics[f"data/train/trace_fraction/{tag}"] = rows / total_rows
        allocated = input_tokens + padded_tokens
        if allocated > 0:
            metrics[f"data/train/padding_ratio/{tag}"] = padded_tokens / allocated
        if input_tokens > 0:
            metrics[f"data/train/supervision_ratio/{tag}"] = (
                supervised_tokens / input_tokens
            )
        if current[12] == 0:
            _add_losses(metrics, f"loss/train/{tag}", current[6:12])
    return metrics


def _add_losses(
    metrics: dict[str, float],
    prefix: str,
    values: Sequence[float],
) -> None:
    names = ("total", "text", "audio")
    for index, name in enumerate(names):
        loss_sum, count = values[2 * index : 2 * index + 2]
        if count <= 0:
            continue
        tag = prefix if name == "total" else f"{prefix}/{name}"
        metrics[tag] = loss_sum / count


__all__ = ["S2SMetricsCallback", "generation_metrics"]
