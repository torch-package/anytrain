from __future__ import annotations

from torch import device as TorchDevice

from anytrain.codec._registry import CodecEntry
from anytrain.codec.protocol import AudioCapability, CodecCapability
from anytrain.codec.spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec

from .assets import (
    DEFAULT_MODEL_BITRATE,
    DEFAULT_MODEL_TYPE,
    LATEST_TAGS,
    MODEL_URLS,
)

_DEFAULT_TAG = LATEST_TAGS[(DEFAULT_MODEL_TYPE, DEFAULT_MODEL_BITRATE)]


def _load_frame(*, device: str | TorchDevice | None = None):
    from .codec import DAC

    return DAC.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="dac",
    capabilities=frozenset({CodecCapability.FRAME_CODES}),
    loader=_load_frame,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        view="dac",
        schema=AudioCodeSchema.FRAME,
        sample_rate=44_100,
        frame_rate=44_100 / 512,
        frame_codebook_sizes=(1024,) * 9,
    ),
    identity=AudioBackendIdentity(
        preset="dac",
        artifact=MODEL_URLS[(DEFAULT_MODEL_TYPE, DEFAULT_MODEL_BITRATE, _DEFAULT_TAG)],
        revision=_DEFAULT_TAG,
    ),
)

__all__ = ["ENTRY"]
