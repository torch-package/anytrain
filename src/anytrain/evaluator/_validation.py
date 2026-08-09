from __future__ import annotations

import torch

MetricValue = float | torch.Tensor
MetricDict = dict[str, MetricValue]


def validate_metric_dict(metrics: object, *, separator: str) -> MetricDict:
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict of string keys to metric values.")
    if not metrics:
        raise ValueError("metrics must contain at least one value.")
    return {
        validate_metric_name(name, separator=separator): validate_metric_value(
            value,
            name=str(name),
        )
        for name, value in metrics.items()
    }


def validate_metric_value(value: object, *, name: str) -> MetricValue:
    if isinstance(value, bool):
        raise TypeError(f"Metric value {name!r} must be a float or 0-d tensor.")
    if isinstance(value, float):
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"Metric value {name!r} must be a 0-d tensor.")
        return value
    raise TypeError(f"Metric value {name!r} must be a float or 0-d tensor.")


def validate_metric_name(name: object, *, separator: str) -> str:
    if not isinstance(name, str):
        raise TypeError("metric key must be a string.")
    if not name:
        raise ValueError("metric key must not be empty.")
    if separator in name:
        raise ValueError(f"metric key must not contain separator {separator!r}.")
    return name
