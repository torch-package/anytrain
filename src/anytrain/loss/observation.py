from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast

import torch
from torch import Tensor, nn

from anytrain import observation


class _Weights(Protocol):
    loss_names: tuple[str, ...]

    @property
    def weights(self) -> Tensor: ...


class _InterpolatedWeights(_Weights, Protocol):
    fixed: _Weights
    dynamic: _Weights
    blend: Tensor


_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def loss(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(module_type, (_LOSS,))
    return module_type


def loss_with_diagnostics(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(module_type, (_LOSS, _DIAGNOSTICS))
    return module_type


def fixed_balancer(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(
        module_type,
        (
            observation.StateObservation(
                "weights",
                _collect_weights,
                recommended=True,
            ),
            observation.StateObservation("fixed", _collect_weights),
        ),
    )
    return module_type


def dynamic_balancer(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(
        module_type,
        (
            observation.StateObservation(
                "weights",
                _collect_weights,
                recommended=True,
            ),
            observation.StateObservation("dynamic", _collect_weights),
        ),
    )
    return module_type


def interpolated_balancer(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(
        module_type,
        (
            observation.StateObservation(
                "weights",
                _collect_weights,
                recommended=True,
                covers_descendants=True,
            ),
            observation.StateObservation("fixed", _collect_fixed),
            observation.StateObservation("dynamic", _collect_dynamic),
            observation.StateObservation("blend", _collect_blend),
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


def _collect_weights(module: nn.Module) -> dict[str, observation.Observation]:
    return _weight_curves(cast(_Weights, module))


def _collect_fixed(module: nn.Module) -> dict[str, observation.Observation]:
    return _weight_curves(cast(_InterpolatedWeights, module).fixed)


def _collect_dynamic(module: nn.Module) -> dict[str, observation.Observation]:
    return _weight_curves(cast(_InterpolatedWeights, module).dynamic)


def _weight_curves(balancer: _Weights) -> dict[str, observation.Observation]:
    with torch.no_grad():
        weights = balancer.weights.detach()
    return {
        name: observation.Curve(weights[index])
        for index, name in enumerate(balancer.loss_names)
    }


def _collect_blend(module: nn.Module) -> observation.Curve:
    balancer = cast(_InterpolatedWeights, module)
    return observation.Curve(balancer.blend.detach())


_LOSS = observation.OutputObservation(
    "loss",
    _collect_loss,
    reduction=observation.Reduction.Mean,
    recommended=True,
)
_DIAGNOSTICS = observation.ForwardEvent(
    "diagnostics",
    reduction=observation.Reduction.Mean,
)


__all__ = [
    "dynamic_balancer",
    "emit_diagnostics",
    "fixed_balancer",
    "interpolated_balancer",
    "loss",
    "loss_with_diagnostics",
    "recommend",
]
