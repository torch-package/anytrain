from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast
from weakref import WeakKeyDictionary, WeakSet

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from ._types import (
    BackwardObservation,
    ForwardObservation,
    ObservationProvider,
    ObservationSpec,
    PollObservation,
)


class RecommendationWarning(UserWarning):
    """A module exposes recommended observations without an attached observer."""


@dataclass
class _Runtime:
    claims: int = 0
    hint: RemovableHandle | None = None


class _RecommendationHook:
    def __call__(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        recommendation_hint(module, inputs)


_SPECS_ATTRIBUTE = "_anytrain_observation_specs"
_SPEC_TYPES = (ForwardObservation, BackwardObservation, PollObservation)
_RECOMMENDATION_TYPES = (ForwardObservation, PollObservation)
_runtimes: WeakKeyDictionary[nn.Module, _Runtime] = WeakKeyDictionary()
_warned_types: WeakSet[type[nn.Module]] = WeakSet()


def register(module: nn.Module, values: tuple[ObservationSpec, ...]) -> None:
    """Bind immutable observation declarations to one module instance."""

    if not isinstance(module, nn.Module):
        raise TypeError("observation.register module must be nn.Module.")
    if _SPECS_ATTRIBUTE in module.__dict__:
        raise RuntimeError("module observations were already registered.")
    declared = _validated_specs(values)
    module.__dict__[_SPECS_ATTRIBUTE] = declared
    _install_recommendation(module, declared)


def recommend(module: nn.Module) -> None:
    """Install the one-shot hint for a module that implements ObservationProvider."""

    values = provided_specs(module)
    if not any(
        isinstance(spec, _RECOMMENDATION_TYPES) and spec.recommended
        for spec in values
    ):
        raise ValueError(
            "observation.recommend requires a supported recommended provided spec."
        )
    _install_recommendation(module, values)


def specs(module: nn.Module) -> tuple[ObservationSpec, ...]:
    value = module.__dict__.get(_SPECS_ATTRIBUTE)
    if value is not None:
        return cast(tuple[ObservationSpec, ...], value)
    return provided_specs(module)


def claim(
    module: nn.Module,
    declared: tuple[ObservationSpec, ...] | None = None,
) -> None:
    values = specs(module) if declared is None else declared
    if not values:
        raise RuntimeError("cannot claim observations from an unregistered module.")
    runtime = _runtime(module)
    runtime.claims += 1
    _consume_hint(module, runtime)


def release(module: nn.Module) -> None:
    runtime = _runtimes.get(module)
    if runtime is None or runtime.claims <= 0:
        raise RuntimeError("cannot release an observation claim that is not active.")
    runtime.claims -= 1


def recommendation_hint(module: nn.Module, inputs: tuple[object, ...]) -> None:
    del inputs
    values = specs(module)
    supported = tuple(
        spec for spec in values if isinstance(spec, _RECOMMENDATION_TYPES)
    )
    runtime = _runtime(module)
    if runtime.claims > 0:
        _consume_hint(module, runtime)
        return
    if not module.training or not torch.is_grad_enabled():
        return
    if torch.compiler.is_compiling():
        _consume_hint(module, runtime)
        return
    _consume_hint(module, runtime)
    if not any(spec.recommended for spec in supported):
        return
    if any(spec.covers_descendants for spec in supported):
        _suppress_descendant_hints(module)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            if torch.distributed.get_rank() != 0:
                return
        except (RuntimeError, ValueError):
            # A recommendation must never make the observed module unusable. This
            # also covers tests or launchers that probe distributed paths before a
            # default process group is fully available.
            return
    module_type = type(module)
    if module_type in _warned_types:
        return
    _warned_types.add(module_type)
    warnings.warn(
        f"{module_type.__name__} exposes recommended training observations. "
        "Add anytrain.lightning.ObservationCallback() to record them, "
        "including step and optional EMA statistics.",
        RecommendationWarning,
        stacklevel=6,
    )


def provided_specs(module: nn.Module) -> tuple[ObservationSpec, ...]:
    if not isinstance(module, ObservationProvider):
        return ()
    return _validated_specs(tuple(module.observation_specs()))


def _validated_specs(values: tuple[ObservationSpec, ...]) -> tuple[ObservationSpec, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("module observation specs must not be empty.")
    normalized: set[str] = set()
    for value in result:
        if not isinstance(value, _SPEC_TYPES):
            raise TypeError("module observation specs must contain ObservationSpec objects.")
        name = value.name.strip("/")
        if name in normalized:
            raise ValueError(
                "module observation spec names must be unique after normalization."
            )
        normalized.add(name)
    return result


def _install_recommendation(
    module: nn.Module,
    values: tuple[ObservationSpec, ...],
) -> None:
    supported = tuple(
        spec for spec in values if isinstance(spec, _RECOMMENDATION_TYPES)
    )
    if not any(spec.recommended for spec in supported):
        return
    if any(spec.covers_descendants for spec in supported):
        _suppress_descendant_hints(module)
    runtime = _runtime(module)
    if runtime.hint is not None or _has_hint(module):
        return
    runtime.hint = module.register_forward_pre_hook(_RecommendationHook())


def _runtime(module: nn.Module) -> _Runtime:
    runtime = _runtimes.get(module)
    if runtime is None:
        runtime = _Runtime()
        _runtimes[module] = runtime
    return runtime


def _consume_hint(module: nn.Module, runtime: _Runtime | None) -> None:
    if runtime is not None and runtime.hint is not None:
        runtime.hint.remove()
        runtime.hint = None
    hooks = module._forward_pre_hooks
    for hook_id, hook in tuple(hooks.items()):
        if isinstance(hook, _RecommendationHook):
            hooks.pop(hook_id, None)


def _has_hint(module: nn.Module) -> bool:
    return any(
        isinstance(hook, _RecommendationHook)
        for hook in module._forward_pre_hooks.values()
    )


def _suppress_descendant_hints(module: nn.Module) -> None:
    for child in module.modules():
        if child is module:
            continue
        _consume_hint(child, _runtimes.get(child))


__all__ = [
    "RecommendationWarning",
    "recommend",
    "register",
    "specs",
]
