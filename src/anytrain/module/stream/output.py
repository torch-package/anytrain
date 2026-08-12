from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .binding import bind
from .layout import Layout
from .types import Codes


@dataclass(eq=False, frozen=True, slots=True)
class TargetBatch:
    """Local labels plus the namespace routing needed to interpret them."""

    values: Tensor
    predict_mask: Tensor
    layout: Layout
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor) or not isinstance(self.predict_mask, Tensor):
            raise TypeError("target values and predict_mask must be Tensors.")
        if not isinstance(self.layout, Layout):
            raise TypeError("target layout must be a Layout.")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("target ignore_index must be an integer.")
        shape = (self.layout.batch_size, self.layout.size)
        if self.values.shape != shape or self.predict_mask.shape != shape:
            raise ValueError("target values and predict_mask must match the root layout shape.")
        if self.values.dtype != torch.long:
            raise TypeError("target values must use torch.int64.")
        if self.predict_mask.dtype != torch.bool:
            raise TypeError("target predict_mask must use torch.bool.")
        if self.values.device != self.layout.lengths.device:
            raise ValueError("target values and layout must be on the same device.")
        if self.predict_mask.device != self.values.device:
            raise ValueError("target values and predict_mask must be on the same device.")

    @property
    def owner(self) -> Tensor:
        return self.layout.owner

    @property
    def source_index(self) -> Tensor:
        return self.layout.source_index

    @property
    def control(self) -> Tensor:
        return self.layout.control

    @property
    def lengths(self) -> Tensor:
        return self.layout.lengths

    @property
    def mask(self) -> Tensor:
        return self.layout.mask


def bind_targets(
    layout: Layout,
    codes: Mapping[str, Codes],
    *,
    predicted_controls: Collection[str] = (),
    ignore_index: int = -100,
) -> TargetBatch:
    """Bind scalar carrier codes to the root sequence layout.

    Values remain local to their owner stream. At control positions they are
    local to ``layout.control_names``. Consumers must use owner/control routing
    or explicitly materialize a global id space.
    """

    if not isinstance(layout, Layout):
        raise TypeError("target layout must be a Layout.")
    if not isinstance(codes, Mapping):
        raise TypeError("target codes must be a mapping.")
    if isinstance(predicted_controls, (str, bytes)) or not isinstance(
        predicted_controls,
        Collection,
    ):
        raise TypeError("predicted_controls must be a collection of control names.")
    if any(not isinstance(name, str) for name in predicted_controls):
        raise TypeError("predicted_controls must contain only strings.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer.")

    required = set(layout.carrier_names)
    missing = required - set(codes)
    if missing:
        raise ValueError(f"target codes are missing carrier streams: {sorted(missing)!r}.")
    extra = set(codes) - required
    if extra:
        raise ValueError(f"target codes contain non-carrier streams: {sorted(extra)!r}.")
    bound = bind(layout, codes)
    selected_controls = frozenset(predicted_controls)
    unknown_controls = selected_controls - set(layout.control_names)
    if unknown_controls:
        raise ValueError(f"unknown predicted controls: {sorted(unknown_controls)!r}.")
    values = torch.full(
        (layout.batch_size, layout.size),
        ignore_index,
        dtype=torch.long,
        device=layout.lengths.device,
    )
    predict_mask = torch.zeros_like(values, dtype=torch.bool)
    batch = torch.arange(layout.batch_size, device=values.device).unsqueeze(1)
    batch = batch.expand_as(values)
    for owner, name in enumerate(layout.stream_names):
        positions = layout.mask & (layout.owner == owner)
        if not bool(positions.any()):
            continue
        payload = bound.values.get(name)
        if payload is None:
            raise RuntimeError(f"carrier stream {name!r} was not bound.")
        if payload.values.ndim != 2:
            raise ValueError(
                f"target stream {name!r} must contain scalar codes with shape [batch, time]."
            )
        values[positions] = payload.values[
            batch[positions],
            layout.source_index[positions],
        ].to(dtype=torch.long)
        predict_mask[positions] = True

    for control_id, name in enumerate(layout.control_names):
        if name not in selected_controls:
            continue
        positions = layout.mask & (layout.control == control_id)
        values[positions] = control_id
        predict_mask[positions] = True
    return TargetBatch(values, predict_mask, layout, ignore_index)


__all__ = ["TargetBatch", "bind_targets"]
