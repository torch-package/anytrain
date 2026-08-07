from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AudioCapability, CodecCapability
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec

from .assets import DEFAULT_CONFIG_NAME, DEFAULT_HF_REPO_ID


def _load_frame(*, device: str | TorchDevice | None = None):
    from .codec import UniCodec

    return UniCodec.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="unicodec",
    capabilities=frozenset({CodecCapability.FRAME_CODES}),
    loader=_load_frame,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_schema=AudioCodeSchema.FRAME,
    audio_spec=AudioCodeSpec(
        names=("frame_0",),
        vocab_sizes=(16_384,),
        sample_rate=24_000,
        frame_rates=(75.0,),
    ),
    identity=AudioBackendIdentity(
        preset="unicodec",
        artifact=f"{DEFAULT_HF_REPO_ID}/{DEFAULT_CONFIG_NAME}",
    ),
)

__all__ = ["ENTRY"]
