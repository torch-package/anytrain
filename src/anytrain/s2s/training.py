from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor

from .data import Modality, Route, TokenBatch
from .model import TokenLossStats, TokenScorer, score_tokens

REPORT_KEY = "s2s_report"


@dataclass(eq=False, frozen=True, slots=True)
class BatchUnits:
    rows: Tensor
    input_tokens: Tensor
    padded_tokens: Tensor
    text_tokens: Tensor
    audio_tokens: Tensor
    audio_seconds: Tensor

    def __post_init__(self) -> None:
        integers = (
            ("rows", self.rows),
            ("input_tokens", self.input_tokens),
            ("padded_tokens", self.padded_tokens),
            ("text_tokens", self.text_tokens),
            ("audio_tokens", self.audio_tokens),
        )
        device: torch.device | None = None
        for name, value in integers:
            _integer_scalar(value, name)
            if value.device.type == "cpu" and bool(value < 0):
                raise ValueError(f"{name} must be non-negative.")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError("batch unit tensors must share a device.")
        if not isinstance(self.audio_seconds, Tensor):
            raise TypeError("audio_seconds must be a Tensor.")
        if self.audio_seconds.ndim != 0:
            raise ValueError("audio_seconds must be a scalar Tensor.")
        if not torch.is_floating_point(self.audio_seconds):
            raise TypeError("audio_seconds must use a floating-point dtype.")
        if device is not None and self.audio_seconds.device != device:
            raise ValueError("batch unit tensors must share a device.")
        if self.audio_seconds.device.type == "cpu" and (
            not bool(torch.isfinite(self.audio_seconds))
            or bool(self.audio_seconds < 0)
        ):
            raise ValueError("audio_seconds must be finite and non-negative.")

    @property
    def supervised_tokens(self) -> Tensor:
        return self.text_tokens + self.audio_tokens

    def __add__(self, other: BatchUnits) -> BatchUnits:
        if not isinstance(other, BatchUnits):
            return NotImplemented
        if self.rows.device != other.rows.device:
            raise ValueError("batch units must share a device before addition.")
        return BatchUnits(
            rows=self.rows + other.rows,
            input_tokens=self.input_tokens + other.input_tokens,
            padded_tokens=self.padded_tokens + other.padded_tokens,
            text_tokens=self.text_tokens + other.text_tokens,
            audio_tokens=self.audio_tokens + other.audio_tokens,
            audio_seconds=self.audio_seconds + other.audio_seconds,
        )


@dataclass(eq=False, frozen=True, slots=True)
class RouteReport:
    route: Route
    units: BatchUnits
    loss: TokenLossStats | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, Route):
            raise TypeError("route report route must be Route.")
        if not isinstance(self.units, BatchUnits):
            raise TypeError("route report units must be BatchUnits.")
        if self.loss is not None and not isinstance(self.loss, TokenLossStats):
            raise TypeError("route report loss must be TokenLossStats or None.")


@dataclass(eq=False, frozen=True, slots=True)
class StepReport:
    loss: TokenLossStats
    routes: tuple[RouteReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.loss, TokenLossStats):
            raise TypeError("step report loss must be TokenLossStats.")
        if not isinstance(self.routes, tuple) or not self.routes:
            raise ValueError("step report routes must be a non-empty tuple.")
        if any(not isinstance(route, RouteReport) for route in self.routes):
            raise TypeError("step report routes must contain RouteReport values.")
        names = tuple(route.route for route in self.routes)
        if len(names) != len(set(names)):
            raise ValueError("step report routes must be unique.")


class TrainingOutput(TypedDict):
    loss: Tensor
    s2s_report: StepReport


def token_objective(provider: TokenScorer, batch: TokenBatch) -> TrainingOutput:
    """Score one token batch and produce the only payload the S2S logger consumes."""

    stats = score_tokens(provider, batch)
    return {
        "loss": stats.total.mean(),
        "s2s_report": step_report(batch, stats),
    }


def step_report(batch: TokenBatch, loss: TokenLossStats) -> StepReport:
    if not isinstance(batch, TokenBatch):
        raise TypeError("step report batch must be TokenBatch.")
    if not isinstance(loss, TokenLossStats):
        raise TypeError("step report loss must be TokenLossStats.")
    if loss.modalities != batch.targets.modalities:
        raise ValueError("step report loss modalities must exactly match batch targets.")
    routes = tuple(dict.fromkeys(batch.routes))
    exact_route_loss = loss.detached() if len(routes) == 1 else None
    reports = tuple(
        RouteReport(
            route=route,
            units=_units(batch, route),
            loss=exact_route_loss,
        )
        for route in routes
    )
    return StepReport(loss=loss.detached(), routes=reports)


def _units(batch: TokenBatch, route: Route) -> BatchUnits:
    device = batch.input.input_ids.device
    selected = torch.tensor(
        [value == route for value in batch.routes],
        dtype=torch.bool,
        device=device,
    )
    attention = batch.input.attention_mask[selected]
    rows = selected.sum(dtype=torch.int64)
    input_tokens = attention.sum(dtype=torch.int64)
    padded_tokens = (~attention).sum(dtype=torch.int64)
    text_tokens = _target_count(batch, Modality.TEXT, selected)
    audio_tokens = _target_count(batch, Modality.AUDIO, selected)
    audio_seconds = (
        torch.zeros((), dtype=torch.float64, device=device)
        if batch.audio_seconds is None
        else batch.audio_seconds[selected].sum(dtype=torch.float64)
    )
    return BatchUnits(
        rows=rows,
        input_tokens=input_tokens,
        padded_tokens=padded_tokens,
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
        audio_seconds=audio_seconds,
    )


def _target_count(batch: TokenBatch, modality: Modality, rows: Tensor) -> Tensor:
    mask = batch.targets.mask(modality)
    if mask is None:
        return torch.zeros((), dtype=torch.int64, device=rows.device)
    return mask[rows].sum(dtype=torch.int64)


def _integer_scalar(value: object, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar Tensor.")
    if value.dtype is not torch.int64:
        raise TypeError(f"{name} must use int64 dtype.")


__all__ = [
    "REPORT_KEY",
    "BatchUnits",
    "RouteReport",
    "StepReport",
    "TrainingOutput",
    "step_report",
    "token_objective",
]
