from __future__ import annotations

from torch import device as TorchDevice

from anytrain.codec._registry import CodecEntry
from anytrain.codec.protocol import CodecCapability


def _load_semantic_acoustic(*, device: str | TorchDevice | None = None):
    from .codec import BiCodec

    return BiCodec.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="bicodec",
    capabilities=frozenset({CodecCapability.SEMANTIC_ACOUSTIC}),
    semantic_acoustic_loader=_load_semantic_acoustic,
)

__all__ = ["ENTRY"]
