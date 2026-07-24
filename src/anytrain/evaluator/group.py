from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from torch import nn

from ._validation import validate_metric_dict
from .abc import EvaluatorABC, MetricDict


class EvaluatorGroup(nn.Module):
    metric_key_separator = EvaluatorABC.metric_key_separator

    def __init__(
        self,
        evaluators: Mapping[str, nn.Module],
    ) -> None:
        super().__init__()
        self.evaluators = nn.ModuleDict(self._validate_evaluators(evaluators))

    def forward(self, *args: Any, **kwargs: Any) -> MetricDict:
        output: MetricDict = {}
        for name, evaluator in self.evaluators.items():
            metrics = evaluator(*args, **kwargs)
            self._merge_metric_output(output, name, metrics)
        return output

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._require_stateful_lifecycle()
        for evaluator in self.evaluators.values():
            _method(evaluator, "update")(*args, **kwargs)

    def compute(self) -> MetricDict:
        self._require_stateful_lifecycle()
        output: MetricDict = {}
        for name, evaluator in self.evaluators.items():
            metrics = _method(evaluator, "compute")()
            self._merge_metric_output(output, name, metrics)
        return output

    def reset(self) -> None:
        self._require_stateful_lifecycle()
        for evaluator in self.evaluators.values():
            _method(evaluator, "reset")()

    def _require_stateful_lifecycle(self) -> None:
        for name, evaluator in self.evaluators.items():
            missing = []
            for method in ("update", "compute", "reset"):
                implementation = getattr(type(evaluator), method, None)
                if implementation is None or implementation is getattr(EvaluatorABC, method):
                    missing.append(method)
            if missing:
                methods = ", ".join(f"{method}()" for method in missing)
                raise NotImplementedError(
                    f"Evaluator {name!r} does not implement the complete stateful lifecycle: "
                    f"{methods}."
                )

    def _validate_evaluators(
        self,
        evaluators: Mapping[str, nn.Module],
    ) -> dict[str, nn.Module]:
        if not isinstance(evaluators, Mapping):
            raise TypeError("evaluators must be a mapping of names to evaluators.")

        if not evaluators:
            raise ValueError("evaluators must contain at least one evaluator.")

        validated: dict[str, nn.Module] = {}
        for raw_name, evaluator in evaluators.items():
            name = self._validate_name(raw_name, "evaluator name")
            if not isinstance(evaluator, nn.Module):
                raise TypeError(f"Evaluator {name!r} must be a torch.nn.Module.")
            validated[name] = evaluator
        return validated

    def _merge_metric_output(
        self,
        output: MetricDict,
        metric_name: str,
        metrics: object,
    ) -> None:
        validated = self._validate_metric_dict(metrics)
        for key, value in validated.items():
            name = self.metric_key_separator.join([metric_name, key])
            if name in output:
                raise ValueError(f"Duplicate metric key {name!r}.")
            output[name] = value

    def _validate_metric_dict(self, metrics: object) -> MetricDict:
        return validate_metric_dict(metrics, separator=self.metric_key_separator)

    def _validate_name(self, name: str, label: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"{label} must be a string.")
        if not name:
            raise ValueError(f"{label} must not be empty.")
        if self.metric_key_separator in name:
            raise ValueError(f"{label} must not contain separator {self.metric_key_separator!r}.")
        return name


def _method(module: nn.Module, name: str) -> Callable[..., object]:
    return cast(Callable[..., object], getattr(module, name))
