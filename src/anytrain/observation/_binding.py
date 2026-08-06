from __future__ import annotations

import inspect
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary, WeakSet

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from ._types import (
    BackwardObservation,
    ForwardEvent,
    ObservationResult,
    ObservationSpec,
    OutputObservation,
    StateObservation,
)


class RecommendationWarning(UserWarning):
    """A module exposes recommended observations without an attached observer."""


@dataclass
class _Runtime:
    claims: int = 0
    hint: RemovableHandle | None = None
    event_sinks: dict[str, OrderedDict[int, EventSink]] = field(default_factory=dict)


class _RecommendationHook:
    def __call__(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        recommendation_hint(module, inputs)


EventSink = Callable[[nn.Module, ObservationResult], bool]


_SPEC_TYPES = (
    OutputObservation,
    BackwardObservation,
    StateObservation,
    ForwardEvent,
)
_RECOMMENDATION_TYPES = (OutputObservation, StateObservation, ForwardEvent)
_runtimes: WeakKeyDictionary[nn.Module, _Runtime] = WeakKeyDictionary()
_warned_types: WeakSet[type[nn.Module]] = WeakSet()
_context: ContextVar[tuple[str, ...]] = ContextVar(
    "anytrain_observation_context",
    default=(),
)


class _Registry:
    """The process-wide declaration registry used by ObservationCallback."""

    def __init__(self) -> None:
        self._specs: dict[type[nn.Module], tuple[ObservationSpec, ...]] = {}

    def register(
        self,
        module_type: type[nn.Module],
        values: tuple[ObservationSpec, ...],
    ) -> None:
        """Register the complete observation contract for one concrete module type."""

        if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError("observation registry keys must be nn.Module classes.")
        if module_type in self._specs:
            raise RuntimeError(
                f"observations are already registered for {module_type.__qualname__}."
            )
        self._specs[module_type] = _validated_specs(values)

    def specs(self, module: nn.Module) -> tuple[ObservationSpec, ...]:
        """Return the declaration for the module's exact concrete type."""

        if not isinstance(module, nn.Module):
            raise TypeError("observation registry lookup requires an nn.Module.")
        return self._specs.get(type(module), ())

    def recommend(self, module: nn.Module) -> None:
        """Install the one-shot hint for a registered module instance."""

        values = self.specs(module)
        if not values:
            return
        if not any(
            isinstance(spec, _RECOMMENDATION_TYPES) and spec.recommended
            for spec in values
        ):
            raise ValueError(
                "observation registry recommend requires a registered recommended spec."
            )
        _install_recommendation(module, values)


registry = _Registry()


@contextmanager
def context(prefix: str) -> Iterator[None]:
    """Prefix every observation produced inside this dynamic call context."""

    if not isinstance(prefix, str):
        raise TypeError("observation context prefix must be a string.")
    normalized = prefix.strip("/")
    if not normalized:
        raise ValueError("observation context prefix must be non-empty.")
    parts = tuple(normalized.split("/"))
    if any(not part for part in parts):
        raise ValueError("observation context prefix must not contain empty path parts.")
    token = _context.set((*_context.get(), *parts))
    try:
        yield
    finally:
        _context.reset(token)


def context_prefix() -> tuple[str, ...]:
    """Return the active immutable observation context path."""

    return _context.get()


def emit(module: nn.Module, name: str, value: ObservationResult) -> None:
    """Publish one declared local forward value to attached observers.

    Every rank calls this function as part of the same module forward. With no
    attached observer it performs no tensor conversion and retains no value.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("observation.emit requires an nn.Module.")
    if torch.compiler.is_compiling():
        return
    runtime = _runtimes.get(module)
    if runtime is None or not any(runtime.event_sinks.values()):
        return
    spec = _event_spec(module, name)
    sinks = runtime.event_sinks.get(spec.name.strip("/"))
    if not sinks:
        return
    for sink_id, sink in tuple(sinks.items()):
        if not sink(module, value):
            sinks.pop(sink_id, None)


def subscribe_event(
    module: nn.Module,
    spec: ForwardEvent,
    sink: EventSink,
) -> RemovableHandle:
    declared = _event_spec(module, spec.name)
    if declared is not spec:
        raise RuntimeError("forward event subscription does not match registry declaration.")
    if not callable(sink):
        raise TypeError("forward event sink must be callable.")
    runtime = _runtime(module)
    sinks = runtime.event_sinks.setdefault(spec.name.strip("/"), OrderedDict())
    handle = RemovableHandle(sinks)
    sinks[handle.id] = sink
    return handle


def claim(
    module: nn.Module,
    declared: tuple[ObservationSpec, ...] | None = None,
) -> None:
    values = registry.specs(module) if declared is None else declared
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
    values = registry.specs(module)
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


def _validated_specs(values: tuple[ObservationSpec, ...]) -> tuple[ObservationSpec, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("module observation specs must not be empty.")
    normalized: set[str] = set()
    for value in result:
        if not isinstance(value, _SPEC_TYPES):
            raise TypeError("module observation specs must contain ObservationSpec objects.")
        if (
            not isinstance(value, ForwardEvent)
            and inspect.ismethod(value.collect)
            and isinstance(value.collect.__self__, nn.Module)
        ):
            raise TypeError(
                "observation collectors must not be bound to an nn.Module instance."
            )
        name = value.name.strip("/")
        if name in normalized:
            raise ValueError(
                "module observation spec names must be unique after normalization."
            )
        normalized.add(name)
    return result


def _event_spec(module: nn.Module, name: str) -> ForwardEvent:
    if not isinstance(module, nn.Module):
        raise TypeError("observation.emit requires an nn.Module.")
    if not isinstance(name, str) or not name.strip("/"):
        raise ValueError("forward event name must be a non-empty string.")
    normalized = name.strip("/")
    for spec in registry.specs(module):
        if isinstance(spec, ForwardEvent) and spec.name.strip("/") == normalized:
            return spec
    raise ValueError(
        f"{type(module).__qualname__} does not declare ForwardEvent {normalized!r}."
    )


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
    "emit",
    "registry",
]
