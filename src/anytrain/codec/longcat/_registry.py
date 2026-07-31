from __future__ import annotations

from torch import device as TorchDevice

from anytrain.codec._registry import CodecEntry
from anytrain.codec.protocol import CodecCapability


def _load_codec(*, device: str | TorchDevice | None = None):
    from .codec import LongCat

    return LongCat.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="longcat",
    capabilities=frozenset(
        {
            CodecCapability.FRAME_CODES,
            CodecCapability.SEMANTIC_ACOUSTIC,
        }
    ),
    frame_loader=_load_codec,
    semantic_acoustic_loader=_load_codec,
)

__all__ = ["ENTRY"]
