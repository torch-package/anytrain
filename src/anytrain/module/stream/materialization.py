from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .binding import Bound
from .expression import AlignMerge, AlignMethod
from .layout import AlignLayout, CompiledNode, Layout
from .types import Codes, Features, Timeline


@dataclass(eq=False, frozen=True, slots=True)
class EmbeddingBatch:
    values: Tensor
    layout: Layout

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("embedding batch values must be a Tensor.")
        if not isinstance(self.layout, Layout):
            raise TypeError("embedding batch layout must be a Layout.")
        if self.values.ndim != 3:
            raise ValueError("embedding batch values must have shape [batch, time, dim].")
        if self.values.shape[:2] != (self.layout.batch_size, self.layout.size):
            raise ValueError("embedding batch values must match the root layout shape.")
        if not torch.is_floating_point(self.values):
            raise TypeError("embedding batch values must use a floating-point dtype.")
        if self.values.device != self.layout.lengths.device:
            raise ValueError("embedding batch values and layout must be on the same device.")

    @property
    def lengths(self) -> Tensor:
        return self.layout.lengths

    @property
    def mask(self) -> Tensor:
        return self.layout.mask


class AlignOperator(nn.Module, ABC):
    """A trainable or stateless evaluator for one compiled Align node."""

    @abstractmethod
    def forward(
        self,
        carrier: Features,
        source: Features,
        layout: AlignLayout,
    ) -> Features: ...


class StandardAlign(AlignOperator):
    """Apply compiled geometry, then use the declarative standard merge."""

    def forward(
        self,
        carrier: Features,
        source: Features,
        layout: AlignLayout,
    ) -> Features:
        if layout.method is AlignMethod.ATTENTION:
            raise ValueError("attention alignment requires a custom AlignOperator.")
        aligned = _apply_alignment(source, layout)
        if layout.merge is AlignMerge.SOURCE:
            values = aligned.values
        elif layout.merge is AlignMerge.ADD:
            if carrier.values.shape[-1] != aligned.values.shape[-1]:
                raise ValueError("add alignment requires equal carrier and source feature dims.")
            values = carrier.values + aligned.values
        else:
            values = torch.cat((carrier.values, aligned.values), dim=-1)
        values = values * layout.target.mask.unsqueeze(-1).to(dtype=values.dtype)
        return Features(
            name=carrier.name,
            values=values,
            lengths=layout.target.lengths,
            frame_rate=layout.target.frame_rate,
        )


class Materializer(nn.Module):
    """Evaluate a bound expression into one embedding sequence.

    Embedders and Align operators are registered here, never in the declarative
    expression or compiled layout.
    """

    def __init__(
        self,
        *,
        embedders: Mapping[str, nn.Module] | None = None,
        operators: Mapping[str, AlignOperator] | None = None,
        control_embedding: nn.Embedding | None = None,
        control_ids: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._embedder_names, self.embedders = _module_list(embedders, "embedders")
        operator_values = _operators(operators)
        operator_values.setdefault("default", StandardAlign())
        self._operator_names, modules = _module_list(operator_values, "operators")
        self.operators = modules
        if control_embedding is not None and not isinstance(control_embedding, nn.Embedding):
            raise TypeError("control_embedding must be an nn.Embedding or None.")
        self.control_embedding = control_embedding
        self.control_ids = _control_ids(control_ids)

    def forward(self, bound: Bound) -> EmbeddingBatch:
        if not isinstance(bound, Bound):
            raise TypeError("Materializer input must be Bound.")
        bound.require_complete()
        layout = bound.layout
        if bool(layout.control_mask.any()):
            control = self.control_embedding
            if control is None:
                raise ValueError("control_embedding is required by the bound layout.")
            missing = set(layout.control_names) - set(self.control_ids)
            if missing:
                raise ValueError(f"control_ids is missing layout controls: {sorted(missing)!r}.")
            largest = max(self.control_ids[name] for name in layout.control_names)
            if largest >= control.num_embeddings:
                raise ValueError("a control_ids row is outside control_embedding.")

        cache: dict[str, Features] = {}

        def evaluate(node: CompiledNode) -> Features:
            existing = cache.get(node.name)
            if existing is not None:
                return existing
            if node.kind == "stream":
                result = self._leaf(node, bound)
            elif node.kind == "align":
                carrier = evaluate(node.children[0])
                source = evaluate(node.children[1])
                if node.alignment is None or node.operator is None:
                    raise RuntimeError("compiled Align node is missing its execution contract.")
                operator = self._operator(node.operator)
                evaluated = operator(carrier, source, node.alignment)
                if not isinstance(evaluated, Features):
                    raise TypeError("AlignOperator must return Features.")
                _validate_feature_timeline(evaluated, node.timeline, node.name)
                result = Features(
                    name=node.name,
                    values=evaluated.values,
                    lengths=node.timeline.lengths,
                    frame_rate=node.timeline.frame_rate,
                )
            elif node.kind == "flatten":
                result = self._flatten(
                    node,
                    tuple(evaluate(child) for child in node.children),
                    control_names=layout.control_names,
                )
            else:
                raise RuntimeError(f"unknown compiled node kind {node.kind!r}.")
            cache[node.name] = result
            return result

        root = evaluate(layout.root)
        return EmbeddingBatch(root.values, layout)

    def _leaf(self, node: CompiledNode, bound: Bound) -> Features:
        payload = bound.values[node.name]
        if isinstance(payload, Features):
            return payload
        if not isinstance(payload, Codes):
            raise TypeError(f"bound stream {node.name!r} has an unsupported payload type.")
        embedder = self._embedder(node.name)
        values = embedder(payload.values)
        if not isinstance(values, Tensor):
            raise TypeError(f"embedder for stream {node.name!r} must return a Tensor.")
        if values.ndim != 3 or values.shape[:2] != payload.values.shape[:2]:
            raise ValueError(
                f"embedder for stream {node.name!r} must preserve [batch, time] "
                "and return [batch, time, dim]."
            )
        if not torch.is_floating_point(values):
            raise TypeError(f"embedder for stream {node.name!r} must return floating features.")
        return Features(
            name=node.name,
            values=values,
            lengths=payload.lengths,
            frame_rate=payload.frame_rate,
        )

    def _flatten(
        self,
        node: CompiledNode,
        children: tuple[Features, ...],
        *,
        control_names: tuple[str, ...],
    ) -> Features:
        if node.item_owner is None or node.item_index is None:
            raise RuntimeError("compiled Flatten node is missing item positions.")
        if not children:
            raise RuntimeError("compiled Flatten node has no children.")
        dim = children[0].values.shape[-1]
        dtype = children[0].values.dtype
        device = children[0].values.device
        for child in children[1:]:
            if child.values.shape[-1] != dim:
                raise ValueError("Flatten children must have the same embedding dimension.")
            if child.values.dtype != dtype or child.values.device != device:
                raise ValueError("Flatten children must share dtype and device.")

        values = children[0].values.new_zeros((node.timeline.batch_size, node.timeline.size, dim))
        active = node.timeline.mask
        batch = torch.arange(node.timeline.batch_size, device=device).unsqueeze(1)
        batch = batch.expand_as(node.item_owner)
        for owner, child in enumerate(children):
            positions = active & (node.item_owner == owner)
            if not bool(positions.any()):
                continue
            values[positions] = child.values[
                batch[positions],
                node.item_index[positions],
            ]

        controls = active & (node.item_owner < 0) & (node.control >= 0)
        if bool(controls.any()):
            control_embedding = self.control_embedding
            if control_embedding is None:
                raise ValueError("control_embedding is required by the Flatten node.")
            control_rows = torch.tensor(
                [self.control_ids[name] for name in control_names],
                dtype=torch.long,
                device=device,
            )
            ids = control_rows[node.control[controls]]
            control_values = control_embedding(ids)
            if control_values.shape[-1] != dim:
                raise ValueError("control embedding dim must match Flatten child features.")
            values[controls] = control_values.to(dtype=dtype)
        return Features(
            name=node.name,
            values=values,
            lengths=node.timeline.lengths,
            frame_rate=node.timeline.frame_rate,
        )

    def _embedder(self, name: str) -> nn.Module:
        try:
            index = self._embedder_names.index(name)
        except ValueError as error:
            raise KeyError(f"no embedder is registered for discrete stream {name!r}.") from error
        return self.embedders[index]

    def _operator(self, name: str) -> AlignOperator:
        try:
            index = self._operator_names.index(name)
        except ValueError as error:
            raise KeyError(f"no AlignOperator is registered as {name!r}.") from error
        operator = self.operators[index]
        if not isinstance(operator, AlignOperator):
            raise TypeError(f"registered operator {name!r} must be an AlignOperator.")
        return operator


def _apply_alignment(source: Features, layout: AlignLayout) -> Features:
    _validate_feature_timeline(source, layout.source, source.name)
    target = layout.target
    if target.size == 0 or source.values.shape[1] == 0:
        values = source.values.new_zeros(
            (source.values.shape[0], target.size, source.values.shape[2])
        )
    elif layout.method in {AlignMethod.IDENTITY, AlignMethod.NEAREST}:
        if layout.indices is None:
            raise RuntimeError("hard Align layout is missing indices.")
        batch = torch.arange(source.values.shape[0], device=source.values.device).unsqueeze(1)
        values = source.values[batch, layout.indices]
    elif layout.method is AlignMethod.LINEAR:
        if layout.indices is None or layout.weights is None:
            raise RuntimeError("linear Align layout is missing indices or weights.")
        batch = torch.arange(source.values.shape[0], device=source.values.device).view(-1, 1, 1)
        picked = source.values[batch, layout.indices]
        weights = layout.weights.to(dtype=source.values.dtype).unsqueeze(-1)
        values = (picked * weights).sum(dim=2)
    elif layout.method is AlignMethod.POOL:
        if layout.starts is None or layout.ends is None:
            raise RuntimeError("pool Align layout is missing windows.")
        prefix = torch.cat(
            (
                source.values.new_zeros((source.values.shape[0], 1, source.values.shape[2])),
                source.values.cumsum(dim=1),
            ),
            dim=1,
        )
        batch = torch.arange(source.values.shape[0], device=source.values.device).unsqueeze(1)
        sums = prefix[batch, layout.ends] - prefix[batch, layout.starts]
        counts = (layout.ends - layout.starts).clamp_min(1).to(source.values.dtype)
        values = sums / counts.unsqueeze(-1)
    elif layout.method is AlignMethod.BROADCAST:
        valid = source.timeline.mask.unsqueeze(-1)
        counts = source.lengths.clamp_min(1).to(source.values.dtype)
        pooled = (source.values * valid).sum(dim=1) / counts.unsqueeze(-1)
        values = pooled.unsqueeze(1).expand(-1, target.size, -1).clone()
    else:
        raise ValueError("attention alignment must be evaluated by a custom AlignOperator.")
    values = values * target.mask.unsqueeze(-1).to(dtype=values.dtype)
    return Features(
        name=source.name,
        values=values,
        lengths=target.lengths,
        frame_rate=target.frame_rate,
    )


def _validate_feature_timeline(features: Features, timeline: Timeline, name: str) -> None:
    if features.values.shape[:2] != (timeline.batch_size, timeline.size):
        raise ValueError(f"features from {name!r} do not match the compiled timeline shape.")
    if features.frame_rate != timeline.frame_rate:
        raise ValueError(f"features from {name!r} do not match the compiled frame_rate.")
    if not torch.equal(features.lengths, timeline.lengths):
        raise ValueError(f"features from {name!r} do not match the compiled lengths.")


def _module_list(
    values: Mapping[str, nn.Module] | None,
    field: str,
) -> tuple[tuple[str, ...], nn.ModuleList]:
    if values is None:
        return (), nn.ModuleList()
    if not isinstance(values, Mapping):
        raise TypeError(f"{field} must be a mapping of names to modules.")
    names: list[str] = []
    modules: list[nn.Module] = []
    for name, module in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field} names must be non-empty strings.")
        if not isinstance(module, nn.Module):
            raise TypeError(f"{field}[{name!r}] must be an nn.Module.")
        names.append(name)
        modules.append(module)
    return tuple(names), nn.ModuleList(modules)


def _operators(
    values: Mapping[str, AlignOperator] | None,
) -> dict[str, AlignOperator]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("operators must be a mapping of names to AlignOperator.")
    parsed: dict[str, AlignOperator] = {}
    for name, operator in values.items():
        if not isinstance(operator, AlignOperator):
            raise TypeError(f"operators[{name!r}] must be an AlignOperator.")
        parsed[name] = operator
    return parsed


def _control_ids(values: Mapping[str, int] | None) -> dict[str, int]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("control_ids must be a mapping of names to integers.")
    parsed: dict[str, int] = {}
    for name, index in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("control_ids names must be non-empty strings.")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"control_ids[{name!r}] must be an integer.")
        if index < 0:
            raise ValueError(f"control_ids[{name!r}] must be a non-negative integer.")
        parsed[name] = index
    return parsed


__all__ = [
    "AlignOperator",
    "EmbeddingBatch",
    "Materializer",
    "StandardAlign",
]
