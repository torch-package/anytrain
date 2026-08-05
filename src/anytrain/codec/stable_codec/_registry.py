from __future__ import annotations

from torch import device as TorchDevice

from anytrain.codec._registry import CodecEntry
from anytrain.codec.protocol import AudioCapability, CodecCapability
from anytrain.codec.spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec

_DEFAULT_ARTIFACT = "stabilityai/stable-codec-speech-16k"


def _load_frame(*, device: str | TorchDevice | None = None):
    from .codec import StableCodec

    return StableCodec.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="stable_codec",
    capabilities=frozenset({CodecCapability.FRAME_CODES}),
    loader=_load_frame,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        view="stable",
        schema=AudioCodeSchema.FRAME,
        sample_rate=16_000,
        frame_rate=25.0,
        frame_codebook_sizes=(46_656,),
    ),
    identity=AudioBackendIdentity(
        preset="stable_codec",
        artifact=_DEFAULT_ARTIFACT,
    ),
)

__all__ = ["ENTRY"]
