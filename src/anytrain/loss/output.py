from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import torch
from torch import Tensor
from typing_extensions import NotRequired


@dataclass(frozen=True)
class LossItem:
    loss: Tensor
    details: dict[str, Tensor] | None = None

    def mean(self, mask: Tensor) -> LossItem:
        loss = self.loss[mask].mean()

        details = self.details
        if details is not None:
            details = {key: value[mask].mean() for key, value in details.items()}
        return LossItem(loss, details)

    def weighted_mean(self, weight: Tensor) -> Tensor:
        if weight.shape != self.loss.shape:
            raise ValueError("loss weights must align with loss rows.")
        if weight.device != self.loss.device:
            raise ValueError("loss weights must be on the same device as loss rows.")
        if weight.is_complex():
            raise TypeError("loss weights must be real-valued observation counts.")
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError("loss weights must be finite and non-negative.")
        total = weight.sum()
        if bool(total.le(0)):
            raise ValueError("loss weights must contain a positive total.")
        safe_loss = self.loss.masked_fill(weight == 0, 0)
        return (safe_loss * weight).sum() / total


class LossOutputs(TypedDict):
    loss: Tensor
    loss_weights: NotRequired[Mapping[str, float]]


def combine_loss_outputs(
    outputs: Sequence[Mapping[str, Any]],
    item_units: Mapping[str, str],
    *,
    loss_key: str = "loss",
    loss_weights_key: str = "loss_weights",
    detail_fallback_to_mean: bool = True,
) -> dict[str, Any]:
    if not outputs:
        raise ValueError("cannot combine an empty output sequence.")

    first_loss = _loss_tensor(outputs[0], loss_key=loss_key)
    result: dict[str, Any] = {loss_key: first_loss.new_zeros(())}
    for name, unit in item_units.items():
        items = [_loss_item(output, name=name) for output in outputs if name in output]
        if not items:
            continue
        item = concatenate_loss_items(items)
        result[name] = item
        result[loss_key] = result[loss_key] + _loss_weight(
            outputs,
            name,
            weights_key=loss_weights_key,
        ) * loss_item_mean(
            item,
            unit=unit,
            fallback_to_mean=detail_fallback_to_mean,
        )
    return result


def iter_loss_items(
    outputs: Mapping[str, Any],
    names: Iterable[str],
) -> Iterator[tuple[str, LossItem]]:
    for name in names:
        value = outputs.get(name)
        if value is not None:
            yield name, _as_loss_item(value, name=name)


def concatenate_loss_items(items: Iterable[LossItem]) -> LossItem:
    values = list(items)
    if not values:
        raise ValueError("cannot concatenate an empty loss item sequence.")
    details = _cat_details([item.details for item in values])
    return LossItem(
        torch.cat([item.loss.reshape(-1) for item in values]),
        details,
    )


def loss_item_mean(
    item: LossItem,
    *,
    unit: str,
    fallback_to_mean: bool = True,
) -> Tensor:
    details = item.details
    if details is None or unit not in details:
        if fallback_to_mean:
            return item.loss.mean()
        raise ValueError(f"loss item details must contain unit {unit!r}.")
    return item.weighted_mean(details[unit].to(device=item.loss.device, dtype=item.loss.dtype))


def _loss_tensor(output: Mapping[str, Any], *, loss_key: str) -> Tensor:
    value = output[loss_key]
    if not isinstance(value, Tensor):
        raise TypeError(f"{loss_key!r} output must be a tensor.")
    return value


def _loss_item(output: Mapping[str, Any], *, name: str) -> LossItem:
    return _as_loss_item(output[name], name=name)


def _as_loss_item(value: object, *, name: str) -> LossItem:
    if not isinstance(value, LossItem):
        raise TypeError(f"{name!r} output must be a LossItem.")
    return value


def _cat_details(values: list[dict[str, Tensor] | None]) -> dict[str, Tensor] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("loss item details must be present for every shard or none.")
    keys = set(present[0])
    if any(set(value) != keys for value in present):
        raise ValueError("loss item details must use consistent keys.")
    return {
        key: torch.cat([value[key].reshape(-1) for value in present])
        for key in sorted(keys)
    }


def _loss_weight(
    outputs: Sequence[Mapping[str, Any]],
    name: str,
    *,
    weights_key: str,
) -> float:
    weights = [
        _output_loss_weights(output, weights_key=weights_key).get(name, 1.0)
        for output in outputs
        if name in output
    ]
    first = weights[0]
    if any(weight != first for weight in weights):
        raise ValueError(f"{name} loss weight changed within a joint batch.")
    return first


def _output_loss_weights(output: Mapping[str, Any], *, weights_key: str) -> Mapping[str, float]:
    value = output.get(weights_key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{weights_key!r} output must be a mapping.")
    return cast(Mapping[str, float], value)
