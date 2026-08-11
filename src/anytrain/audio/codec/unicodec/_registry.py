from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._detokenizer import represent_frame_detokenizer
from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec._tokenizer import represent_frame_tokenizer
from anytrain.audio.codec.protocol import AudioCapability
from anytrain.audio.codec.spec import (
    AudioBackendIdentity,
    AudioCodeSpec,
    AudioCodeStreamSpec,
)

from .assets import DEFAULT_CONFIG_NAME, DEFAULT_HF_REPO_ID


def _load_frame(*, device: str | TorchDevice | None = None):
    from .codec import UniCodec

    return UniCodec.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="unicodec",
    loader=_load_frame,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        sample_rate=24_000,
        streams=(AudioCodeStreamSpec("rvq_0", 16_384, frame_rate=75.0),),
    ),
    identity=AudioBackendIdentity(
        preset="unicodec",
        artifact=f"{DEFAULT_HF_REPO_ID}/{DEFAULT_CONFIG_NAME}",
    ),
    tokenizer=represent_frame_tokenizer,
    detokenizer=represent_frame_detokenizer,
)

__all__ = ["ENTRY"]
