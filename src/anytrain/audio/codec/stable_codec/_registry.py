from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._detokenizer import represent_frame_detokenizer
from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec._tokenizer import represent_frame_tokenizer
from anytrain.audio.codec.protocol import AudioCapability
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec

_DEFAULT_ARTIFACT = "stabilityai/stable-codec-speech-16k"


def _load_frame(*, device: str | TorchDevice | None = None):
    from .codec import StableCodec

    return StableCodec.from_pretrained(device=device)


ENTRY = CodecEntry(
    name="stable_codec",
    loader=_load_frame,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        names=("rvq_0",),
        vocab_sizes=(46_656,),
        sample_rate=16_000,
        frame_rates=(25.0,),
    ),
    identity=AudioBackendIdentity(
        preset="stable_codec",
        artifact=_DEFAULT_ARTIFACT,
    ),
    tokenizer=represent_frame_tokenizer,
    detokenizer=represent_frame_detokenizer,
)

__all__ = ["ENTRY"]
