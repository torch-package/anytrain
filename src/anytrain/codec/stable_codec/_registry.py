from __future__ import annotations

from torch import device as TorchDevice

from anytrain.codec._registry import CodecEntry
from anytrain.codec.protocol import CodecCapability


def _load_frame(*, device: str | TorchDevice | None = None):
    from .codec import StableCodec

    return StableCodec.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="stable_codec",
    capabilities=frozenset({CodecCapability.FRAME_CODES}),
    frame_loader=_load_frame,
)

__all__ = ["ENTRY"]
