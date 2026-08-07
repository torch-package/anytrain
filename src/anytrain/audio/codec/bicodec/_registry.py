from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AudioCapability, CodecCapability
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec

from .assets import DEFAULT_HF_REPO_ID

DEFAULT_BICODEC_REVISION = "642071559bfc6346c2359d19dcb6be3f9dd8a05d"


def _load_semantic_global(*, device: str | TorchDevice | None = None):
    from .codec import BiCodec

    return BiCodec.from_pretrained(device=device, revision=DEFAULT_BICODEC_REVISION)


ENTRY = CodecEntry(
    name="bicodec",
    capabilities=frozenset({CodecCapability.SEMANTIC_GLOBAL}),
    loader=_load_semantic_global,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        view="bicodec",
        schema=AudioCodeSchema.SEMANTIC_GLOBAL,
        sample_rate=16_000,
        frame_rate=50.0,
        semantic_codebook_sizes=(8192,),
        global_codebook_sizes=(4096,),
        global_unit_length=32,
    ),
    identity=AudioBackendIdentity(
        preset="bicodec",
        artifact=DEFAULT_HF_REPO_ID,
        revision=DEFAULT_BICODEC_REVISION,
    ),
)

__all__ = ["DEFAULT_BICODEC_REVISION", "ENTRY"]
