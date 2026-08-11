from __future__ import annotations

from torch import device as TorchDevice

from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AudioCapability
from anytrain.audio.codec.spec import (
    AudioBackendIdentity,
    AudioCodeSpec,
    AudioCodeStreamSpec,
)

from .assets import DEFAULT_HF_REPO_ID, DEFAULT_LONGCAT_REVISION


def _load_codec(*, device: str | TorchDevice | None = None):
    from .codec import LongCat

    return LongCat.from_pretrained(device=device, revision=DEFAULT_LONGCAT_REVISION)


def _represent_tokenizer(backend, code_spec, identity):
    from .tokenizer import represent_tokenizer

    return represent_tokenizer(backend, code_spec, identity)


def _represent_detokenizer(backend, code_spec, identity):
    from .detokenizer import represent_detokenizer

    return represent_detokenizer(backend, code_spec, identity)


ENTRY = CodecEntry(
    name="longcat",
    loader=_load_codec,
    audio_capabilities=frozenset(
        {
            AudioCapability.TOKENIZER,
            AudioCapability.DETOKENIZER,
        }
    ),
    audio_spec=AudioCodeSpec(
        sample_rate=16_000,
        streams=(
            AudioCodeStreamSpec("semantic", 8192, frame_rate=16_000 / 960),
            *(
                AudioCodeStreamSpec(
                    f"rvq_{index}",
                    8100,
                    frame_rate=16_000 / 960,
                )
                for index in range(3)
            ),
        ),
    ),
    identity=AudioBackendIdentity(
        preset="longcat",
        artifact=DEFAULT_HF_REPO_ID,
        revision=DEFAULT_LONGCAT_REVISION,
    ),
    tokenizer=_represent_tokenizer,
    detokenizer=_represent_detokenizer,
)

__all__ = ["DEFAULT_LONGCAT_REVISION", "ENTRY"]
