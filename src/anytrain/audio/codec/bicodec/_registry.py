from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AudioCapability
from anytrain.audio.codec.spec import (
    AudioBackendIdentity,
    AudioCodeSpec,
    AudioCodeStreamSpec,
)

from .assets import DEFAULT_HF_REPO_ID

DEFAULT_BICODEC_REVISION = "642071559bfc6346c2359d19dcb6be3f9dd8a05d"


def _load_semantic_global(*, device: str | TorchDevice | None = None):
    from .codec import BiCodec

    return BiCodec.from_pretrained(device=device, revision=DEFAULT_BICODEC_REVISION)


def _represent_tokenizer(backend, code_spec, identity):
    from .tokenizer import represent_tokenizer

    return represent_tokenizer(backend, code_spec, identity)


def _represent_detokenizer(backend, code_spec, identity):
    from .detokenizer import represent_detokenizer

    return represent_detokenizer(backend, code_spec, identity)


ENTRY = CodecEntry(
    name="bicodec",
    loader=_load_semantic_global,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        sample_rate=16_000,
        streams=(
            AudioCodeStreamSpec("semantic", 8192, frame_rate=50.0),
            AudioCodeStreamSpec("global", 4096, fixed_units=32),
        ),
    ),
    identity=AudioBackendIdentity(
        preset="bicodec",
        artifact=DEFAULT_HF_REPO_ID,
        revision=DEFAULT_BICODEC_REVISION,
    ),
    tokenizer=_represent_tokenizer,
    detokenizer=_represent_detokenizer,
)

__all__ = ["DEFAULT_BICODEC_REVISION", "ENTRY"]
