from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from ._validation import (
    MetricDict,
    validate_metric_dict,
)


class EvaluatorABC(nn.Module, ABC):
    metric_key_separator = "/"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> MetricDict:
        return self._validate_metric_dict(self.evaluate(*args, **kwargs))

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> MetricDict:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement stateful update().")

    def compute(self) -> MetricDict:
        raise NotImplementedError(f"{type(self).__name__} does not implement stateful compute().")

    def reset(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement stateful reset().")

    def _validate_metric_dict(self, metrics: object) -> MetricDict:
        return validate_metric_dict(metrics, separator=self.metric_key_separator)
