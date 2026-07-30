from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from lightning import pytorch as pl
from torch import Tensor

from anytrain.loss import LossItem, iter_loss_items
from anytrain.stats import TimeBucketedMean, time_bucketed_mean

from .. import experiment

LossItemsFn = Callable[[Mapping[str, Any]], Iterable[tuple[str, LossItem]]]
LossItemLabelsFn = Callable[[str, Any], Sequence[object]]


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
        item_names: Iterable[str] | None = None,
        loss_items_fn: LossItemsFn | None = None,
        window_capacity: int = 20,
    ) -> None:
        super().__init__()
        if item_names is not None and loss_items_fn is not None:
            raise ValueError("item_names and loss_items_fn are mutually exclusive.")
        _positive_int(window_capacity, "window_capacity")
        self.loss_key = loss_key
        self.item_names = tuple(item_names) if item_names is not None else None
        self.loss_items_fn = loss_items_fn
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
        self._append(self.loss_key, _tensor(outputs, self.loss_key))
        for name, item in self._loss_items(outputs):
            self._append(name, item.loss)

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

    def _loss_items(self, outputs: Mapping[str, Any]) -> Iterable[tuple[str, LossItem]]:
        if self.loss_items_fn is not None:
            return self.loss_items_fn(outputs)
        if self.item_names is not None:
            return iter_loss_items(outputs, self.item_names)
        return _available_loss_items(outputs, loss_key=self.loss_key)

    def _append(self, name: str, value: Tensor) -> None:
        scalar = value.detach().float().mean()
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
        world_size = int(getattr(trainer, "world_size", 1))
        if world_size <= 1:
            self._synchronized = True
            return

        strategy = getattr(trainer, "strategy", None)
        reduce = getattr(strategy, "reduce", None)
        if not callable(reduce):
            raise RuntimeError("Distributed loss summary requires trainer.strategy.reduce.")

        names = tuple(sorted(self._values))
        schema = _schema(
            names,
            window_capacity=self.window_capacity,
            device=_summary_device(self._values, pl_module),
        )
        minimum_schema = cast(Tensor, reduce(schema.clone(), reduce_op="min"))
        maximum_schema = cast(Tensor, reduce(schema.clone(), reduce_op="max"))
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
            minimum = cast(Tensor, reduce(local_count.clone(), reduce_op="min"))
            maximum = cast(Tensor, reduce(local_count.clone(), reduce_op="max"))
            if _tensor_int(minimum) != _tensor_int(maximum):
                raise RuntimeError(
                    f"Loss summary metric {name!r} has different observation counts across ranks."
                )
            values.first = _reduced_window(values.first, values.capacity, reduce)
            values.last = _reduced_window(values.last, values.capacity, reduce)

        self._synchronized = True


class LossItemLoggerCallback(pl.Callback):
    def __init__(
        self,
        labels_fn: LossItemLabelsFn,
        *,
        template: str = "{objective}_{key}/{label}",
        label_name: str = "label",
        item_names: Iterable[str] | None = None,
        loss_items_fn: LossItemsFn | None = None,
    ) -> None:
        super().__init__()
        if item_names is not None and loss_items_fn is not None:
            raise ValueError("item_names and loss_items_fn are mutually exclusive.")
        self.labels_fn = labels_fn
        self.template = template
        self.label_name = label_name
        self.item_names = tuple(item_names) if item_names is not None else None
        self.loss_items_fn = loss_items_fn

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, batch_idx
        if not isinstance(outputs, Mapping):
            raise TypeError("LossItemLoggerCallback requires mapping training outputs.")

        for objective, item in self._loss_items(outputs):
            labels = list(self.labels_fn(objective, batch))
            if item.loss.numel() != len(labels):
                raise ValueError(
                    f"{objective} loss rows must align with logged label rows."
                )
            self._log_item(pl_module, objective, item, labels)

    def _loss_items(self, outputs: Mapping[str, Any]) -> Iterable[tuple[str, LossItem]]:
        if self.loss_items_fn is not None:
            return self.loss_items_fn(outputs)
        if self.item_names is not None:
            return iter_loss_items(outputs, self.item_names)
        return _available_loss_items(outputs, loss_key="loss")

    def _log_item(
        self,
        pl_module: pl.LightningModule,
        objective: str,
        item: LossItem,
        labels: Sequence[object],
    ) -> None:
        for label in dict.fromkeys(labels):
            mask = torch.tensor(
                [value == label for value in labels],
                device=item.loss.device,
                dtype=torch.bool,
            )
            mean_item = item.mean(mask)
            pl_module.log(
                self._tag(objective=objective, key="loss", label=label),
                mean_item.loss,
            )
            if mean_item.details is None:
                continue
            for key, scalar in mean_item.details.items():
                pl_module.log(
                    self._tag(objective=objective, key=key, label=label),
                    scalar,
                )

    def _tag(self, *, objective: str, key: str, label: object) -> str:
        fields = {
            "objective": objective,
            "key": key,
            "label": label,
            self.label_name: label,
        }
        return self.template.format(**fields)


class LossTimeBucketLoggerCallback(pl.Callback):
    def __init__(
        self,
        *,
        item_name: str,
        detail_key: str,
        histogram_tag: str | None = None,
        scalar_template: str,
        every_n_steps: int | None = 100,
        bucket_count: int = 10,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        if bucket_count < 1:
            raise ValueError("bucket_count must be positive")
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")
        self.item_name = item_name
        self.detail_key = detail_key
        self.histogram_tag = histogram_tag
        self.scalar_template = scalar_template
        self.every_n_steps = every_n_steps
        self.bucket_count = bucket_count
        self.t_min = t_min
        self.t_max = t_max

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch_idx
        if not self.should_log(trainer, pl_module, batch):
            return
        if not isinstance(outputs, Mapping):
            raise TypeError("LossTimeBucketLoggerCallback requires mapping training outputs.")

        item = outputs.get(self.item_name)
        if item is None:
            return
        if not isinstance(item, LossItem):
            raise TypeError(f"{self.item_name!r} output must be a LossItem.")
        if item.details is None:
            raise TypeError(f"{self.item_name!r} loss item requires details.")
        values = item.details.get(self.detail_key)
        if not isinstance(values, Tensor):
            raise TypeError(
                f"{self.item_name!r} details must contain tensor {self.detail_key!r}."
            )

        self._log_histogram(trainer, values)
        self._log_bucket_loss(trainer, item.loss, values)

    def should_log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
    ) -> bool:
        del pl_module, batch
        if self.every_n_steps is None:
            return True
        return int(trainer.global_step) % self.every_n_steps == 0

    def _log_histogram(self, trainer: pl.Trainer, values: Tensor) -> None:
        if self.histogram_tag is None:
            return
        if not _is_global_zero(trainer):
            return
        writer = experiment.histogram(trainer)
        if writer is None:
            return
        writer.add_histogram(
            self.histogram_tag,
            values.detach().float().cpu(),
            int(trainer.global_step),
        )

    def _log_bucket_loss(
        self,
        trainer: pl.Trainer,
        loss: Tensor,
        values: Tensor,
    ) -> None:
        writer = experiment.scalar(trainer)
        if writer is None:
            return

        bucketed = time_bucketed_mean(
            loss.detach(),
            values.detach(),
            bucket_count=self.bucket_count,
            t_min=self.t_min,
            t_max=self.t_max,
        )
        bucketed = _sync_bucketed(trainer, bucketed)
        if not _is_global_zero(trainer):
            return

        mean = bucketed.mean.detach().cpu()
        count = bucketed.count.detach().cpu()
        edges = bucketed.edges.detach().cpu()
        for index in range(self.bucket_count):
            if count[index] <= 0:
                continue
            writer.add_scalar(
                self._bucket_tag(edges[index], edges[index + 1]),
                float(mean[index]),
                int(trainer.global_step),
            )

    def _bucket_tag(self, lower: Tensor, upper: Tensor) -> str:
        return self.scalar_template.format(
            item=self.item_name,
            key=self.detail_key,
            lower=float(lower),
            upper=float(upper),
        )


def _available_loss_items(
    outputs: Mapping[str, Any],
    *,
    loss_key: str,
) -> Iterable[tuple[str, LossItem]]:
    for name, value in outputs.items():
        if name == loss_key:
            continue
        if isinstance(value, LossItem):
            yield name, value


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
    values: Sequence[Tensor],
    capacity: int,
    reduce: Callable[..., object],
) -> deque[Tensor]:
    reduced = cast(Tensor, reduce(torch.stack(tuple(values)), reduce_op="mean"))
    return deque((value.detach() for value in reduced.unbind(0)), maxlen=capacity)


def _tensor_floats(values: Sequence[Tensor]) -> list[float]:
    return [float(value) for value in torch.stack(tuple(values)).detach().cpu().tolist()]


def _tensor_int(value: Tensor) -> int:
    return int(value.detach().cpu().item())


def _tensor(outputs: Mapping[str, Any], name: str) -> Tensor:
    value = outputs[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name!r} output must be a tensor.")
    return value


def _sync_bucketed(trainer: pl.Trainer, bucketed: TimeBucketedMean) -> TimeBucketedMean:
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        return bucketed
    return TimeBucketedMean(
        edges=bucketed.edges,
        total=cast(Tensor, reduce(bucketed.total, reduce_op="sum")),
        count=cast(Tensor, reduce(bucketed.count, reduce_op="sum")),
    )


def _is_global_zero(trainer: pl.Trainer) -> bool:
    return bool(getattr(trainer, "is_global_zero", True))


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "LossItemLabelsFn",
    "LossItemLoggerCallback",
    "LossItemsFn",
    "LossSummaryCallback",
    "LossTimeBucketLoggerCallback",
    "window_summary",
]
