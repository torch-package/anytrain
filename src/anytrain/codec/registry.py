from __future__ import annotations

import torch

from .protocol import CodecCapability, FrameCodec, SemanticAcousticCodec

_CAPABILITIES: dict[str, frozenset[CodecCapability]] = {
    "longcat": frozenset(
        {
            CodecCapability.FRAME_CODES,
            CodecCapability.SEMANTIC_ACOUSTIC,
        }
    ),
    "bicodec": frozenset(
        {
            CodecCapability.SEMANTIC_ACOUSTIC,
        }
    ),
    "stable_codec": frozenset({CodecCapability.FRAME_CODES}),
    "dac": frozenset({CodecCapability.FRAME_CODES}),
    "unicodec": frozenset({CodecCapability.FRAME_CODES}),
}


def capabilities(name: str) -> frozenset[CodecCapability]:
    """Return statically declared capabilities without loading a codec model."""
    try:
        return _CAPABILITIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_CAPABILITIES))
        raise ValueError(f"Unknown codec {name!r}. Available codecs: {available}.") from exc


def load_semantic_acoustic(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> SemanticAcousticCodec:
    """Load a default codec implementing the semantic-acoustic contract."""
    supported = capabilities(name)
    if CodecCapability.SEMANTIC_ACOUSTIC not in supported:
        raise ValueError(
            f"Codec {name!r} does not provide the "
            f"{CodecCapability.SEMANTIC_ACOUSTIC.value!r} capability."
        )

    if name == "longcat":
        from .longcat import LongCat

        return LongCat.from_pretrained(device=device)

    if name == "bicodec":
        from .bicodec import BiCodec

        return BiCodec.from_pretrained(device=device)

    raise RuntimeError(f"Codec {name!r} has no semantic-acoustic loader.")


def load_frame(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> FrameCodec:
    """Load a default codec implementing the complete frame-code contract."""
    supported = capabilities(name)
    if CodecCapability.FRAME_CODES not in supported:
        raise ValueError(
            f"Codec {name!r} does not provide the "
            f"{CodecCapability.FRAME_CODES.value!r} capability."
        )

    if name == "longcat":
        from .longcat import LongCat

        return LongCat.from_pretrained(device=device)

    if name == "unicodec":
        from .unicodec import UniCodec

        return UniCodec.from_pretrained(device=device)

    if name == "stable_codec":
        from .stable_codec import StableCodec

        return StableCodec.from_pretrained(device=device)

    if name == "dac":
        from .dac import DAC

        return DAC.from_pretrained(device=device)

    raise RuntimeError(f"Codec {name!r} has no frame-code loader.")


__all__ = ["capabilities", "load_frame", "load_semantic_acoustic"]
