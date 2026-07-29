from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

import torch
from lightning import pytorch as pl
from torch import Tensor

from anytrain.loss import LossItem, iter_loss_items
from anytrain.stats import TimeBucketedMean, time_bucketed_mean

from .. import experiment

LossItemsFn = Callable[[Mapping[str, Any]], Iterable[tuple[str, LossItem]]]
LossItemLabelsFn = Callable[[str, Any], Sequence[object]]


def window_summary(
    values: Sequence[float],
    window: int = 20,
) -> dict[str, float | int | None]:
    if window <= 0:
        raise ValueError("window must be positive.")
    if not values:
        return {"steps": 0}
    size = min(window, len(values))
    first = sum(values[:size]) / size
    last = sum(values[-size:]) / size
    return {
        "steps": len(values),
        "window": size,
        "first": values[0],
        "last": values[-1],
        "first_mean": first,
        "last_mean": last,
        "last_to_first": None if first == 0 else last / first,
    }


class LossSummaryCallback(pl.Callback):
    def __init__(
        self,
        *,
        loss_key: str = "loss",
        item_names: Iterable[str] | None = None,
        loss_items_fn: LossItemsFn | None = None,
    ) -> None:
        super().__init__()
        if item_names is not None and loss_items_fn is not None:
            raise ValueError("item_names and loss_items_fn are mutually exclusive.")
        self.loss_key = loss_key
        self.item_names = tuple(item_names) if item_names is not None else None
        self.loss_items_fn = loss_items_fn
        self.values: dict[str, list[float]] = {}

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

    def report(self, window: int = 20) -> dict[str, dict[str, float | int | None]]:
        return {
            name: window_summary(values, window)
            for name, values in self.values.items()
        }

    def _loss_items(self, outputs: Mapping[str, Any]) -> Iterable[tuple[str, LossItem]]:
        if self.loss_items_fn is not None:
            return self.loss_items_fn(outputs)
        if self.item_names is not None:
            return iter_loss_items(outputs, self.item_names)
        return _available_loss_items(outputs, loss_key=self.loss_key)

    def _append(self, name: str, value: Tensor) -> None:
        self.values.setdefault(name, []).append(float(value.detach().float().mean()))


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
