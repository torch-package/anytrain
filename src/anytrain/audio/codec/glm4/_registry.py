from __future__ import annotations

import os

from torch import device as TorchDevice

from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AudioCapability
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec

DEFAULT_GLM4_ARTIFACT = "zai-org/glm-4-voice-tokenizer"
DEFAULT_GLM4_REVISION = "a5f2404e63c84e92f5238908e1706316324ebafa"
GLM4_SOURCE_ROOT_ENV = "GLM4_VOICE_SOURCE_ROOT"
GLM4_RUNTIME_GUIDANCE = (
    f"Set {GLM4_SOURCE_ROOT_ENV} to the root of a GLM-4-Voice checkout or fork "
    "that provides speech_tokenizer (or pass source_root directly to "
    "GLM4Tokenizer.from_pretrained). The source Git commit is not pinned; "
    "required files, import origin, APIs, and model/preprocess contracts are "
    "validated instead. The validated runtime requires "
    "transformers==4.44.1, which conflicts with transformers>=4.46 in the "
    "BiCodec extra and transformers>=4.51.0 in the module extra; use an "
    "independent GLM producer unless a joint environment has passed exact "
    "token-ID parity."
)


def _load_tokenizer(*, device: str | TorchDevice | None = None):
    source_root = os.environ.get(GLM4_SOURCE_ROOT_ENV)
    if not source_root:
        raise RuntimeError(
            f"GLM-4 online tokenization requires an explicit source_root. {GLM4_RUNTIME_GUIDANCE}"
        )
    from .codec import GLM4Tokenizer

    return GLM4Tokenizer.from_pretrained(
        source_root=source_root,
        revision=DEFAULT_GLM4_REVISION,
        device=device,
    )


ENTRY = CodecEntry(
    name="glm4",
    capabilities=frozenset(),
    loader=_load_tokenizer,
    audio_capabilities=frozenset({AudioCapability.TOKENIZER}),
    audio_schema=AudioCodeSchema.FRAME,
    audio_spec=AudioCodeSpec(
        names=("frame_0",),
        vocab_sizes=(16_384,),
        sample_rate=16_000,
        frame_rates=(12.5,),
    ),
    identity=AudioBackendIdentity(
        preset="glm4",
        artifact=DEFAULT_GLM4_ARTIFACT,
        revision=DEFAULT_GLM4_REVISION,
    ),
)

__all__ = [
    "DEFAULT_GLM4_ARTIFACT",
    "DEFAULT_GLM4_REVISION",
    "ENTRY",
    "GLM4_RUNTIME_GUIDANCE",
    "GLM4_SOURCE_ROOT_ENV",
]
