from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from lightning import pytorch as pl
from torch import Tensor

from .. import _capability


@dataclass
class _MetricWindow:
    capacity: int
    first: deque[Tensor] = field(init=False)
    last: deque[Tensor] = field(init=False)
    steps: int = 0

    def __post_init__(self) -> None:
        self.first = deque(maxlen=self.capacity)
        self.last = deque(maxlen=self.capacity)

    def append(self, value: Tensor) -> None:
        if len(self.first) < self.capacity:
            self.first.append(value)
        self.last.append(value)
        self.steps += 1


def window_summary(
    values: Sequence[float],
    window: int = 20,
) -> dict[str, float | int | None]:
    return _summary(values, values, steps=len(values), window=_window(window))


class LossSummaryCallback(pl.Callback):
    def __init__(
        self,
        *,
        loss_key: str = "loss",
        loss_names: Iterable[str] = (),
        window_capacity: int = 20,
    ) -> None:
        super().__init__()
        _non_empty_string(loss_key, "loss_key")
        _positive_int(window_capacity, "window_capacity")
        self.loss_key = loss_key
        self.loss_names = _loss_names(loss_names, loss_key=loss_key)
        self.window_capacity = window_capacity
        self._values: dict[str, _MetricWindow] = {}
        self._synchronized = False

    @property
    def values(self) -> dict[str, list[float]]:
        """Return a bounded, host-materialized compatibility view of the values."""
        return {
            name: _materialized_values(values)
            for name, values in self._values.items()
        }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if not isinstance(outputs, Mapping):
            return
        selected = tuple(
            (name, _scalar_tensor(outputs, name))
            for name in (self.loss_key, *self.loss_names)
        )
        for name, value in selected:
            self._append(name, value)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._synchronize(trainer, pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        if not self._values:
            self._synchronized = True

    def report(
        self,
        window: int | None = None,
    ) -> dict[str, dict[str, float | int | None]]:
        if not self._synchronized:
            raise RuntimeError(
                "Loss summary report is unavailable before training-end synchronization."
            )
        selected_window = self.window_capacity if window is None else window
        return {
            name: _window_report(values, selected_window)
            for name, values in self._values.items()
        }

    def _append(self, name: str, value: Tensor) -> None:
        scalar = value.detach().float()
        values = self._values.get(name)
        if values is None:
            values = _MetricWindow(self.window_capacity)
            self._values[name] = values
        values.append(scalar)
        self._synchronized = False

    def _synchronize(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._synchronized:
            return
        world_size = _capability.world_size(trainer)
        if world_size <= 1:
            self._synchronized = True
            return

        names = tuple(sorted(self._values))
        schema = _schema(
            names,
            window_capacity=self.window_capacity,
            device=_summary_device(self._values, pl_module),
        )
        reduce_error = "Distributed loss summary requires trainer.strategy.reduce."
        minimum_schema = _capability.strategy_reduce(
            trainer,
            schema.clone(),
            reduce_op="min",
            error=reduce_error,
        )
        maximum_schema = _capability.strategy_reduce(
            trainer,
            schema.clone(),
            reduce_op="max",
            error=reduce_error,
        )
        if not torch.equal(minimum_schema, maximum_schema):
            raise RuntimeError(
                "Loss summary metric names differ across ranks, or window capacity is inconsistent."
            )

        for name in names:
            values = self._values[name]
            if values.steps == 0:
                continue
            local_count = torch.tensor(
                values.steps,
                dtype=torch.int64,
                device=values.first[0].device,
            )
            minimum = _capability.strategy_reduce(
                trainer,
                local_count.clone(),
                reduce_op="min",
                error=reduce_error,
            )
            maximum = _capability.strategy_reduce(
                trainer,
                local_count.clone(),
                reduce_op="max",
                error=reduce_error,
            )
            if _tensor_int(minimum) != _tensor_int(maximum):
                raise RuntimeError(
                    f"Loss summary metric {name!r} has different observation counts across ranks."
                )
            values.first = _reduced_window(trainer, values.first, values.capacity)
            values.last = _reduced_window(trainer, values.last, values.capacity)

        self._synchronized = True


def _materialized_values(values: _MetricWindow) -> list[float]:
    if values.steps <= values.capacity:
        tensors: Sequence[Tensor] = values.first
    else:
        overlap = max(0, 2 * values.capacity - values.steps)
        tensors = (*values.first, *tuple(values.last)[overlap:])
    return _tensor_floats(tensors)


def _window_report(
    values: _MetricWindow,
    window: int,
) -> dict[str, float | int | None]:
    window = _window(window)
    if values.steps == 0:
        return {"steps": 0}
    if window > values.capacity and values.steps > values.capacity:
        raise ValueError(
            "window exceeds the retained loss summary capacity; "
            "increase window_capacity when constructing the callback."
        )

    first_values = _tensor_floats(values.first)
    last_values = _tensor_floats(values.last)
    return _summary(first_values, last_values, steps=values.steps, window=window)


def _summary(
    first_values: Sequence[float],
    last_values: Sequence[float],
    *,
    steps: int,
    window: int,
) -> dict[str, float | int | None]:
    if steps == 0:
        return {"steps": 0}
    size = min(window, steps)
    first_mean = sum(first_values[:size]) / size
    last_mean = sum(last_values[-size:]) / size
    return {
        "steps": steps,
        "window": size,
        "first": first_values[0],
        "last": last_values[-1],
        "first_mean": first_mean,
        "last_mean": last_mean,
        "last_to_first": None if first_mean == 0 else last_mean / first_mean,
    }


def _window(value: int) -> int:
    _positive_int(value, "window")
    return value


def _schema(
    names: Sequence[str],
    *,
    window_capacity: int,
    device: torch.device,
) -> Tensor:
    digest = hashlib.sha256()
    digest.update(window_capacity.to_bytes(8, byteorder="big"))
    for name in names:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return torch.tensor(list(digest.digest()), dtype=torch.int64, device=device)


def _summary_device(
    values: Mapping[str, _MetricWindow],
    pl_module: pl.LightningModule,
) -> torch.device:
    for name in sorted(values):
        metric = values[name]
        if metric.first:
            return metric.first[0].device
    device = getattr(pl_module, "device", None)
    if isinstance(device, torch.device):
        return device
    raise RuntimeError(
        "Distributed loss summary requires pl_module.device when no metrics were observed."
    )


def _reduced_window(
    trainer: pl.Trainer,
    values: Sequence[Tensor],
    capacity: int,
) -> deque[Tensor]:
    reduced = _capability.strategy_reduce(
        trainer,
        torch.stack(tuple(values)),
        reduce_op="mean",
        error="Distributed loss summary requires trainer.strategy.reduce.",
    )
    return deque((value.detach() for value in reduced.unbind(0)), maxlen=capacity)


def _tensor_floats(values: Sequence[Tensor]) -> list[float]:
    return [float(value) for value in torch.stack(tuple(values)).detach().cpu().tolist()]


def _tensor_int(value: Tensor) -> int:
    return int(value.detach().cpu().item())


def _scalar_tensor(outputs: Mapping[str, Any], name: str) -> Tensor:
    value = outputs[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name!r} output must be a tensor.")
    if value.ndim != 0:
        raise ValueError(f"{name!r} output must be a scalar tensor.")
    return value


def _loss_names(values: Iterable[str], *, loss_key: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("loss_names must be an iterable of strings.")
    names = tuple(values)
    for index, name in enumerate(names):
        _non_empty_string(name, f"loss_names[{index}]")
    if loss_key in names:
        raise ValueError("loss_names must not contain loss_key.")
    if len(set(names)) != len(names):
        raise ValueError("loss_names must be unique.")
    return names


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "LossSummaryCallback",
    "window_summary",
]
