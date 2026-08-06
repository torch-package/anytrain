from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast

from torch import Tensor, nn

from anytrain import observation


class _FixedWeights(Protocol):
    loss_names: list[str]
    weights: Tensor


class _UncertaintyWeights(Protocol):
    loss_names: list[str]

    @property
    def weight(self) -> Tensor: ...


_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def loss(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(module_type, (_LOSS,))
    return module_type


def loss_with_diagnostics(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(module_type, (_LOSS, _DIAGNOSTICS))
    return module_type


def fixed_weights(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(
        module_type,
        (
            observation.StateObservation(
                "weights",
                _collect_fixed_weights,
                recommended=True,
            ),
        ),
    )
    return module_type


def uncertainty_weights(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(
        module_type,
        (
            observation.StateObservation(
                "weights",
                _collect_uncertainty_weights,
                recommended=True,
            ),
        ),
    )
    return module_type


def recommend(module: nn.Module) -> None:
    observation.registry.recommend(module)


def emit_diagnostics(module: nn.Module, values: Mapping[str, Tensor]) -> None:
    observation.emit(
        module,
        "diagnostics",
        {name: observation.Curve(value.detach()) for name, value in values.items()},
    )


def _collect_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> observation.Curve:
    del module, inputs, kwargs
    if not isinstance(output, Tensor):
        raise TypeError("loss observation requires a Tensor module output.")
    return observation.Curve(output)


def _collect_fixed_weights(module: nn.Module) -> dict[str, observation.Observation]:
    balancer = cast(_FixedWeights, module)
    return {
        name: observation.Curve(balancer.weights[index].detach())
        for index, name in enumerate(balancer.loss_names)
    }


def _collect_uncertainty_weights(module: nn.Module) -> dict[str, observation.Observation]:
    balancer = cast(_UncertaintyWeights, module)
    weights = balancer.weight.detach()
    return {
        name: observation.Curve(weights[index])
        for index, name in enumerate(balancer.loss_names)
    }


_LOSS = observation.OutputObservation(
    "loss",
    _collect_loss,
    recommended=True,
)
_DIAGNOSTICS = observation.ForwardEvent("diagnostics")


__all__ = [
    "emit_diagnostics",
    "fixed_weights",
    "loss",
    "loss_with_diagnostics",
    "recommend",
    "uncertainty_weights",
]
