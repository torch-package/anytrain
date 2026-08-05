from __future__ import annotations

from torch import device as TorchDevice

from anytrain.codec._registry import CodecEntry
from anytrain.codec.protocol import CodecCapability

DEFAULT_BICODEC_REVISION = "642071559bfc6346c2359d19dcb6be3f9dd8a05d"


def _load_semantic_global(*, device: str | TorchDevice | None = None):
    from .codec import BiCodec

    return BiCodec.from_pretrained(device=device, revision=DEFAULT_BICODEC_REVISION)


ENTRY = CodecEntry(
    name="bicodec",
    capabilities=frozenset({CodecCapability.SEMANTIC_GLOBAL}),
    semantic_global_loader=_load_semantic_global,
)

__all__ = ["DEFAULT_BICODEC_REVISION", "ENTRY"]
