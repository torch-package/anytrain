from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import torch

from .layout import CompiledNode, Layout
from .types import Codes, Features, Timeline

Payload: TypeAlias = Codes | Features


@dataclass(eq=False, slots=True)
class Bound:
    """A validated, mapping-protected set of payloads for one compiled layout."""

    layout: Layout
    values: Mapping[str, Payload]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, Layout):
            raise TypeError("bound layout must be a Layout.")
        parsed = _values(self.layout, self.values)
        self.values = MappingProxyType(parsed)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in self.layout.stream_names if name not in self.values)

    @property
    def complete(self) -> bool:
        return not self.missing

    def bind(self, values: Mapping[str, Payload]) -> Bound:
        if not isinstance(values, Mapping):
            raise TypeError("bound values must be a mapping.")
        overlap = set(values) & set(self.values)
        if overlap:
            raise ValueError(f"stream payloads are already bound: {sorted(overlap)!r}.")
        return Bound(self.layout, {**self.values, **values})

    def require_complete(self) -> None:
        missing = self.missing
        if missing:
            raise ValueError(f"layout has unbound stream leaves: {list(missing)!r}.")


def bind(
    layout: Layout,
    values: Mapping[str, Payload] | None = None,
) -> Bound:
    return Bound(layout, {} if values is None else values)


def _values(layout: Layout, values: Mapping[str, Payload]) -> dict[str, Payload]:
    if not isinstance(values, Mapping):
        raise TypeError("bound values must be a mapping.")
    timelines = _stream_timelines(layout.root)
    unknown = set(values) - set(timelines)
    if unknown:
        raise ValueError(f"bound values contain unknown streams: {sorted(unknown)!r}.")

    parsed: dict[str, Payload] = {}
    for name, value in values.items():
        if not isinstance(value, (Codes, Features)):
            raise TypeError(f"bound value {name!r} must be Codes or Features.")
        if value.name != name:
            raise ValueError(f"bound value key {name!r} must match payload name {value.name!r}.")
        _validate_timeline(value.timeline, timelines[name], name)
        parsed[name] = value
    return parsed


def _stream_timelines(root: CompiledNode) -> dict[str, Timeline]:
    timelines: dict[str, Timeline] = {}
    seen: set[str] = set()

    def visit(node: CompiledNode) -> None:
        if node.name in seen:
            return
        seen.add(node.name)
        if node.kind == "stream":
            timelines[node.name] = node.timeline
        for child in node.children:
            visit(child)

    visit(root)
    return timelines


def _validate_timeline(actual: Timeline, expected: Timeline, name: str) -> None:
    if actual.size != expected.size:
        raise ValueError(f"bound stream {name!r} size does not match its layout timeline.")
    if actual.frame_rate != expected.frame_rate:
        raise ValueError(f"bound stream {name!r} frame_rate does not match its layout timeline.")
    if actual.lengths.device != expected.lengths.device:
        raise ValueError(f"bound stream {name!r} must be on the layout device.")
    if not torch.equal(actual.lengths, expected.lengths):
        raise ValueError(f"bound stream {name!r} lengths do not match its layout timeline.")


__all__ = ["Bound", "Payload", "bind"]
