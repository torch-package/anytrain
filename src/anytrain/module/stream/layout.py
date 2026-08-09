from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .expression import (
    Align,
    AlignMerge,
    AlignMethod,
    Expr,
    Flatten,
    FrameMajor,
    Stream,
    StreamMajor,
)
from .types import Timeline

CONTROL_OWNER = -1


@dataclass(eq=False, frozen=True, slots=True)
class AlignLayout:
    source: Timeline
    target: Timeline
    method: AlignMethod
    merge: AlignMerge
    indices: Tensor | None = None
    weights: Tensor | None = None
    starts: Tensor | None = None
    ends: Tensor | None = None


@dataclass(eq=False, frozen=True, slots=True)
class CompiledNode:
    name: str
    kind: str
    timeline: Timeline
    carrier_names: tuple[str, ...]
    carrier: Tensor
    source_index: Tensor
    control: Tensor
    children: tuple[CompiledNode, ...] = ()
    item_owner: Tensor | None = None
    item_index: Tensor | None = None
    alignment: AlignLayout | None = None
    operator: str | None = None


@dataclass(eq=False, frozen=True, slots=True)
class Layout:
    """Batch-compiled geometry for an expression.

    Layout contains positions and operator metadata only. It never owns payload
    values or trainable modules.
    """

    expression: Expr
    root: CompiledNode
    stream_names: tuple[str, ...]
    control_names: tuple[str, ...]

    @property
    def timeline(self) -> Timeline:
        return self.root.timeline

    @property
    def lengths(self) -> Tensor:
        return self.root.timeline.lengths

    @property
    def size(self) -> int:
        return self.root.timeline.size

    @property
    def batch_size(self) -> int:
        return self.root.timeline.batch_size

    @property
    def mask(self) -> Tensor:
        return self.root.timeline.mask

    @property
    def owner(self) -> Tensor:
        return self.root.carrier

    @property
    def source_index(self) -> Tensor:
        return self.root.source_index

    @property
    def control(self) -> Tensor:
        return self.root.control

    @property
    def control_mask(self) -> Tensor:
        return self.mask & (self.root.control >= 0)

    @property
    def carrier_names(self) -> tuple[str, ...]:
        return self.root.carrier_names

    def stream_mask(self, name: str) -> Tensor:
        try:
            owner = self.stream_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown layout stream {name!r}.") from error
        return self.mask & (self.root.carrier == owner)


def compile_layout(
    expression: Expr | str,
    timelines: Mapping[str, Timeline],
) -> Layout:
    """Compile a declarative expression against one batch of timelines."""

    root_expression = expression if not isinstance(expression, str) else Stream(expression)
    if not isinstance(root_expression, (Stream, Align, Flatten)):
        raise TypeError("expression must be Stream, Align, Flatten, or a stream name.")
    if not timelines:
        raise ValueError("timelines must contain at least one stream.")
    _validate_timelines(timelines)

    definitions: dict[str, object] = {}
    _collect_definitions(root_expression, definitions)
    stream_names: list[str] = []
    stream_ordinals: dict[str, int] = {}
    control_names: list[str] = []
    control_ordinals: dict[str, int] = {}
    compiled: dict[str, CompiledNode] = {}
    compiling: set[str] = set()

    def control_id(name: str | None) -> int:
        if name is None:
            return -1
        if name not in control_ordinals:
            control_ordinals[name] = len(control_names)
            control_names.append(name)
        return control_ordinals[name]

    def stream_id(name: str) -> int:
        if name not in stream_ordinals:
            stream_ordinals[name] = len(stream_names)
            stream_names.append(name)
        return stream_ordinals[name]

    def resolve(value: object) -> object:
        if isinstance(value, str):
            return definitions.get(value, Stream(value))
        return value

    def build(value: object) -> CompiledNode:
        value = resolve(value)
        if not isinstance(value, (Stream, Align, Flatten)):
            raise TypeError("expression references must resolve to stream expressions.")
        existing = compiled.get(value.name)
        if existing is not None:
            return existing
        if value.name in compiling:
            raise ValueError(f"expression contains a cycle at node {value.name!r}.")
        compiling.add(value.name)
        if isinstance(value, Stream):
            timeline = timelines.get(value.name)
            if timeline is None:
                raise KeyError(f"missing timeline for stream {value.name!r}.")
            owner = stream_id(value.name)
            positions = torch.arange(timeline.size, device=timeline.lengths.device)
            carrier = torch.full(
                (timeline.batch_size, timeline.size),
                owner,
                dtype=torch.long,
                device=timeline.lengths.device,
            )
            source_index = positions.unsqueeze(0).expand(timeline.batch_size, -1)
            source_index = torch.where(
                timeline.mask, source_index, torch.full_like(source_index, -1)
            )
            control = torch.full_like(source_index, -1)
            node = CompiledNode(
                name=value.name,
                kind="stream",
                timeline=timeline,
                carrier_names=(value.name,),
                carrier=carrier,
                source_index=source_index,
                control=control,
            )
        elif isinstance(value, Align):
            carrier_node = build(value.carrier)
            source_node = build(value.source)
            if carrier_node.kind == "flatten" or source_node.kind == "flatten":
                raise ValueError("Align operands must be timeline expressions, not Flatten.")
            alignment = _compile_align(
                source_node.timeline,
                carrier_node.timeline,
                value.spec.method,
                value.spec.merge,
            )
            node = CompiledNode(
                name=value.name,
                kind="align",
                timeline=carrier_node.timeline,
                carrier_names=carrier_node.carrier_names,
                carrier=carrier_node.carrier,
                source_index=carrier_node.source_index,
                control=carrier_node.control,
                children=(carrier_node, source_node),
                alignment=alignment,
                operator=value.operator,
            )
        else:
            node = _build_flatten(
                value,
                build=build,
                control_id=control_id,
            )
        compiling.remove(value.name)
        compiled[value.name] = node
        return node

    root = build(root_expression)
    if not stream_names:
        raise ValueError("expression must contain at least one stream leaf.")
    return Layout(
        expression=root_expression,
        root=root,
        stream_names=tuple(stream_names),
        control_names=tuple(control_names),
    )


def _build_flatten(
    expression: Flatten,
    *,
    build,
    control_id,
) -> CompiledNode:
    children: list[CompiledNode] = []
    child_names: set[str] = set()
    item_owner_rows: list[list[int]] = []
    item_index_rows: list[list[int]] = []
    control_rows: list[list[int]] = []
    row_lengths: list[int] = []

    for segment in expression.segments:
        items = tuple(build(item) for item in segment.items)
        for item in items:
            if item.name in child_names:
                continue
            child_names.add(item.name)
            children.append(item)
        if isinstance(segment, FrameMajor):
            _validate_frame_children(items, expression.name)

    child_by_name = {child.name: child for child in children}
    carrier_names: list[str] = []
    for child in children:
        for name in child.carrier_names:
            if name not in carrier_names:
                carrier_names.append(name)

    for segment in expression.segments:
        control_id(segment.controls.segment_start)
        if isinstance(segment, FrameMajor):
            control_id(segment.controls.frame_start)
            control_id(segment.controls.frame_end)
        control_id(segment.controls.segment_end)

    # Structural rows are intentionally compiled from lengths only. Payloads are
    # bound later and never participate in this step.
    length_values = {
        child.name: child.timeline.lengths.detach().cpu().tolist() for child in children
    }
    batch_size = children[0].timeline.batch_size
    for child in children[1:]:
        if child.timeline.batch_size != batch_size:
            raise ValueError("all flatten children must have the same batch size.")

    child_index = {child.name: index for index, child in enumerate(children)}
    for batch_index in range(batch_size):
        owner_row: list[int] = []
        source_row: list[int] = []
        control_row: list[int] = []
        for segment in expression.segments:
            items = tuple(child_by_name[_ref_name(item)] for item in segment.items)
            if isinstance(segment, StreamMajor):
                _append_control(
                    control_id(segment.controls.segment_start), owner_row, source_row, control_row
                )
                for item in items:
                    length = int(length_values[item.name][batch_index])
                    owner_row.extend([child_index[item.name]] * length)
                    source_row.extend(range(length))
                    control_row.extend([-1] * length)
                _append_control(
                    control_id(segment.controls.segment_end), owner_row, source_row, control_row
                )
            else:
                _append_control(
                    control_id(segment.controls.segment_start), owner_row, source_row, control_row
                )
                length = _frame_length(items, batch_index)
                for time_index in range(length):
                    _append_control(
                        control_id(segment.controls.frame_start), owner_row, source_row, control_row
                    )
                    for item in items:
                        owner_row.append(child_index[item.name])
                        source_row.append(time_index)
                        control_row.append(-1)
                    _append_control(
                        control_id(segment.controls.frame_end), owner_row, source_row, control_row
                    )
                _append_control(
                    control_id(segment.controls.segment_end), owner_row, source_row, control_row
                )
        item_owner_rows.append(owner_row)
        item_index_rows.append(source_row)
        control_rows.append(control_row)
        row_lengths.append(len(owner_row))

    size = max(row_lengths, default=0)
    device = children[0].timeline.lengths.device
    item_owner = _pad_rows(item_owner_rows, size=size, device=device)
    item_index = _pad_rows(item_index_rows, size=size, device=device)
    control = _pad_rows(control_rows, size=size, device=device)
    lengths = torch.tensor(row_lengths, dtype=torch.long, device=device)
    timeline = Timeline(lengths=lengths, size=size)

    carrier = torch.full((batch_size, size), -1, dtype=torch.long, device=device)
    source_index = torch.full_like(carrier, -1)
    child_active = item_owner >= 0
    batch = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(item_owner)
    for index, child in enumerate(children):
        positions = child_active & (item_owner == index)
        if not bool(positions.any()):
            continue
        child_positions = item_index[positions]
        batch_index = batch[positions]
        carrier[positions] = child.carrier[batch_index, child_positions]
        source_index[positions] = child.source_index[batch_index, child_positions]
        control[positions] = child.control[batch_index, child_positions]

    return CompiledNode(
        name=expression.name,
        kind="flatten",
        timeline=timeline,
        carrier_names=tuple(carrier_names),
        carrier=carrier,
        source_index=source_index,
        control=control,
        children=tuple(children),
        item_owner=item_owner,
        item_index=item_index,
    )


def _compile_align(
    source: Timeline,
    target: Timeline,
    method: AlignMethod,
    merge: AlignMerge,
) -> AlignLayout:
    if source.batch_size != target.batch_size:
        raise ValueError("align source and target must have the same batch size.")
    if source.lengths.device != target.lengths.device:
        raise ValueError("align timelines must be on the same device.")
    active_target = target.lengths > 0
    if target.size > 0 and source.size == 0 and bool(active_target.any()):
        raise ValueError("align source must have a positive padded size for a non-empty target.")
    if bool((source.lengths[active_target] <= 0).any()):
        raise ValueError("active align targets require a non-empty source.")

    if method is AlignMethod.IDENTITY:
        if not torch.equal(source.lengths, target.lengths):
            raise ValueError("identity alignment requires equal sample lengths.")
        positions = torch.arange(target.size, device=target.lengths.device)
        indices = positions.unsqueeze(0).expand(target.batch_size, -1)
        indices = indices.clamp_max(max(source.size - 1, 0))
        return AlignLayout(source, target, method, merge, indices=indices)
    if method is AlignMethod.NEAREST:
        return AlignLayout(source, target, method, merge, indices=_nearest_indices(source, target))
    if method is AlignMethod.LINEAR:
        indices, weights = _linear_indices(source, target)
        return AlignLayout(source, target, method, merge, indices=indices, weights=weights)
    if method is AlignMethod.POOL:
        if bool((target.lengths > source.lengths).any()):
            raise ValueError(
                "pool alignment requires target lengths no greater than source lengths."
            )
        starts, ends = _pool_windows(source, target)
        return AlignLayout(source, target, method, merge, starts=starts, ends=ends)
    return AlignLayout(source, target, method, merge)


def _nearest_indices(source: Timeline, target: Timeline) -> Tensor:
    positions = torch.arange(target.size, device=target.lengths.device, dtype=torch.long).unsqueeze(
        0
    )
    source_lengths = source.lengths.to(dtype=torch.long).unsqueeze(1)
    target_lengths = target.lengths.clamp_min(1).to(dtype=torch.long).unsqueeze(1)
    indices = ((2 * positions + 1) * source_lengths) // (2 * target_lengths)
    indices = torch.minimum(indices, (source_lengths - 1).clamp_min(0))
    return torch.where(target.mask, indices, torch.zeros_like(indices))


def _linear_indices(source: Timeline, target: Timeline) -> tuple[Tensor, Tensor]:
    positions = torch.arange(
        target.size, device=target.lengths.device, dtype=torch.float64
    ).unsqueeze(0)
    source_lengths = source.lengths.to(dtype=torch.float64).unsqueeze(1)
    target_lengths = target.lengths.clamp_min(1).to(dtype=torch.float64).unsqueeze(1)
    position = (positions + 0.5) * source_lengths / target_lengths - 0.5
    position = position.clamp_min(0).minimum((source_lengths - 1).clamp_min(0))
    left = position.floor().to(dtype=torch.long)
    alpha = position - left.to(dtype=position.dtype)
    right = (left + 1).clamp_max((source.lengths - 1).clamp_min(0).unsqueeze(1))
    indices = torch.stack((left, right), dim=2)
    weights = torch.stack((1.0 - alpha, alpha), dim=2)
    target_mask = target.mask.unsqueeze(-1)
    return (
        torch.where(target_mask, indices, torch.zeros_like(indices)),
        torch.where(target_mask, weights, torch.zeros_like(weights)),
    )


def _pool_windows(source: Timeline, target: Timeline) -> tuple[Tensor, Tensor]:
    positions = torch.arange(target.size, device=target.lengths.device, dtype=torch.long).unsqueeze(
        0
    )
    source_lengths = source.lengths.to(dtype=torch.long).unsqueeze(1)
    target_lengths = target.lengths.clamp_min(1).to(dtype=torch.long).unsqueeze(1)
    starts = (positions * source_lengths) // target_lengths
    ends = ((positions + 1) * source_lengths + target_lengths - 1) // target_lengths
    target_mask = target.mask
    return (
        torch.where(target_mask, starts, torch.zeros_like(starts)),
        torch.where(target_mask, ends, torch.zeros_like(ends)),
    )


def _validate_frame_children(children: tuple[CompiledNode, ...], name: str) -> None:
    if any(child.kind == "flatten" for child in children):
        raise ValueError(f"frame-major node {name!r} cannot contain nested Flatten items.")
    first = children[0].timeline.lengths
    if any(not torch.equal(first, child.timeline.lengths) for child in children[1:]):
        raise ValueError(f"frame-major node {name!r} requires equal per-sample lengths.")
    frame_rate = children[0].timeline.frame_rate
    if any(child.timeline.frame_rate != frame_rate for child in children[1:]):
        raise ValueError(f"frame-major node {name!r} requires equal frame rates.")


def _append_control(
    control: int,
    owner_row: list[int],
    source_row: list[int],
    control_row: list[int],
) -> None:
    if control < 0:
        return
    owner_row.append(CONTROL_OWNER)
    source_row.append(-1)
    control_row.append(control)


def _pad_rows(rows: list[list[int]], *, size: int, device: torch.device) -> Tensor:
    output = torch.full((len(rows), size), -1, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        if row:
            output[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return output


def _frame_length(children: tuple[CompiledNode, ...], batch_index: int) -> int:
    lengths = {int(child.timeline.lengths[batch_index].detach().cpu()) for child in children}
    if len(lengths) != 1:
        raise ValueError("frame-major children must have equal per-sample lengths.")
    return next(iter(lengths))


def _ref_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (Stream, Align, Flatten)):
        return value.name
    raise TypeError("expression reference must be a stream name or expression.")


def _collect_definitions(value: object, definitions: dict[str, object]) -> None:
    if isinstance(value, str):
        return
    if isinstance(value, Stream):
        _register_definition(value, definitions)
        return
    if isinstance(value, Align):
        _register_definition(value, definitions)
        _collect_definitions(value.carrier, definitions)
        _collect_definitions(value.source, definitions)
        return
    if isinstance(value, Flatten):
        _register_definition(value, definitions)
        for segment in value.segments:
            for item in segment.items:
                _collect_definitions(item, definitions)
        return
    raise TypeError("expression references must be Stream, Align, Flatten, or names.")


def _register_definition(value: object, definitions: dict[str, object]) -> None:
    name = _ref_name(value)
    existing = definitions.get(name)
    if existing is not None and existing != value:
        raise ValueError(f"expression node name {name!r} is defined more than once.")
    definitions[name] = value


def _validate_timelines(timelines: Mapping[str, Timeline]) -> None:
    first: Timeline | None = None
    for name, timeline in timelines.items():
        if not isinstance(name, str) or not name:
            raise ValueError("timeline mapping keys must be non-empty strings.")
        if not isinstance(timeline, Timeline):
            raise TypeError(f"timelines[{name!r}] must be a Timeline.")
        if first is None:
            first = timeline
        elif timeline.batch_size != first.batch_size:
            raise ValueError("all timelines must have the same batch size.")
        elif timeline.lengths.device != first.lengths.device:
            raise ValueError("all timeline lengths must be on the same device.")


__all__ = [
    "AlignLayout",
    "CONTROL_OWNER",
    "CompiledNode",
    "Layout",
    "compile_layout",
]
