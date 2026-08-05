from __future__ import annotations

from typing import cast

import torch

from ._registry import CodecEntry
from .bicodec._registry import ENTRY as BICODEC_ENTRY
from .dac._registry import ENTRY as DAC_ENTRY
from .longcat._registry import ENTRY as LONGCAT_ENTRY
from .protocol import (
    CodecCapability,
    FrameCodec,
    SemanticAcousticCodec,
    SemanticGlobalCodec,
)
from .stable_codec._registry import ENTRY as STABLE_CODEC_ENTRY
from .unicodec._registry import ENTRY as UNICODEC_ENTRY

_ENTRIES: dict[str, CodecEntry] = {
    entry.name: entry
    for entry in (
        BICODEC_ENTRY,
        DAC_ENTRY,
        LONGCAT_ENTRY,
        STABLE_CODEC_ENTRY,
        UNICODEC_ENTRY,
    )
}


def capabilities(name: str) -> frozenset[CodecCapability]:
    """Return statically declared capabilities without loading a codec model."""
    return _entry(name).capabilities


def _entry(name: str) -> CodecEntry:
    try:
        return _ENTRIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENTRIES))
        raise ValueError(f"Unknown codec {name!r}. Available codecs: {available}.") from exc


def _require_capability(name: str, capability: CodecCapability) -> CodecEntry:
    entry = _entry(name)
    if capability not in entry.capabilities:
        raise ValueError(
            f"Codec {name!r} does not provide the {capability.value!r} capability."
        )
    return entry


def load_semantic_acoustic(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> SemanticAcousticCodec:
    """Load a default codec implementing the semantic-acoustic contract."""
    entry = _require_capability(name, CodecCapability.SEMANTIC_ACOUSTIC)
    if entry.semantic_acoustic_loader is None:
        raise RuntimeError(f"Codec {name!r} has no semantic-acoustic loader.")
    return cast(SemanticAcousticCodec, entry.semantic_acoustic_loader(device=device))


def load_semantic_global(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> SemanticGlobalCodec:
    """Load a default codec implementing the semantic-global contract."""
    entry = _require_capability(name, CodecCapability.SEMANTIC_GLOBAL)
    if entry.semantic_global_loader is None:
        raise RuntimeError(f"Codec {name!r} has no semantic-global loader.")
    return cast(SemanticGlobalCodec, entry.semantic_global_loader(device=device))


def load_frame(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> FrameCodec:
    """Load a default codec implementing the complete frame-code contract."""
    entry = _require_capability(name, CodecCapability.FRAME_CODES)
    if entry.frame_loader is None:
        raise RuntimeError(f"Codec {name!r} has no frame-code loader.")
    return cast(FrameCodec, entry.frame_loader(device=device))


__all__ = [
    "capabilities",
    "load_frame",
    "load_semantic_acoustic",
    "load_semantic_global",
]
