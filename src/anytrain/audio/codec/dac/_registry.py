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
    loader=_load_frame,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        sample_rate=44_100,
        streams=tuple(
            AudioCodeStreamSpec(
                f"rvq_{index}",
                1024,
                frame_rate=44_100 / 512,
            )
            for index in range(9)
        ),
    ),
    identity=AudioBackendIdentity(
        preset="dac",
        artifact=MODEL_URLS[(DEFAULT_MODEL_TYPE, DEFAULT_MODEL_BITRATE, _DEFAULT_TAG)],
        revision=_DEFAULT_TAG,
    ),
    tokenizer=represent_frame_tokenizer,
    detokenizer=represent_frame_detokenizer,
)

__all__ = ["ENTRY"]
