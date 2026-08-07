from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AcousticLayout, AudioCapability, CodecCapability
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec

from .assets import DEFAULT_HF_REPO_ID, DEFAULT_LONGCAT_REVISION


def _load_codec(*, device: str | TorchDevice | None = None):
    from .codec import LongCat

    return LongCat.from_pretrained(device=device, revision=DEFAULT_LONGCAT_REVISION)


ENTRY = CodecEntry(
    name="longcat",
    capabilities=frozenset(
        {
            CodecCapability.FRAME_CODES,
            CodecCapability.SEMANTIC_ACOUSTIC,
        }
    ),
    loader=_load_codec,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        view="longcat",
        schema=AudioCodeSchema.SEMANTIC_ACOUSTIC,
        sample_rate=16_000,
        frame_rate=16_000 / 960,
        frame_codebook_sizes=(8192, 8100, 8100, 8100),
        semantic_codebook_sizes=(8192,),
        acoustic_codebook_sizes=(8100, 8100, 8100),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    ),
    identity=AudioBackendIdentity(
        preset="longcat",
        artifact=DEFAULT_HF_REPO_ID,
        revision=DEFAULT_LONGCAT_REVISION,
    ),
)

__all__ = ["DEFAULT_LONGCAT_REVISION", "ENTRY"]
