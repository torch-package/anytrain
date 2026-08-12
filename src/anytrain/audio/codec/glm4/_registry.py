from __future__ import annotations

import os
from typing import TYPE_CHECKING

from torch import device as TorchDevice

from anytrain.audio.codec._planning import chunked_pooled_planning_estimator
from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.protocol import AudioCapability
from anytrain.audio.codec.spec import (
    AudioBackendIdentity,
    AudioCodeSpec,
    AudioCodeStreamSpec,
)

if TYPE_CHECKING:
    from anytrain.audio._tokenizer import AudioTokenizer
    from anytrain.audio.codec.protocol import CodecBackend

DEFAULT_GLM4_ARTIFACT = "zai-org/glm-4-voice-tokenizer"
DEFAULT_GLM4_REVISION = "a5f2404e63c84e92f5238908e1706316324ebafa"
GLM4_IDENTITY_REVISION = f"{DEFAULT_GLM4_REVISION}+anytrain-batched-chunked-v1"
GLM4_SOURCE_ROOT_ENV = "GLM4_VOICE_SOURCE_ROOT"
SAMPLE_RATE = 16_000
FEATURE_HOP_LENGTH = 160
STFT_WINDOW_LENGTH = 400
CONVOLUTION_STRIDE = 2
POOLING_KERNEL_SIZE = 4
FRAME_RATE = SAMPLE_RATE / (FEATURE_HOP_LENGTH * CONVOLUTION_STRIDE * POOLING_KERNEL_SIZE)
CODEBOOK_SIZE = 16_384
FEATURE_DIM = 1_280
MAX_SOURCE_POSITIONS = 1_500
_PAD_MULTIPLE = FEATURE_HOP_LENGTH * CONVOLUTION_STRIDE * POOLING_KERNEL_SIZE
_MAXIMUM_MODEL_FRAMES = MAX_SOURCE_POSITIONS * CONVOLUTION_STRIDE * FEATURE_HOP_LENGTH
_MAXIMUM_PADDED_FRAMES = (_MAXIMUM_MODEL_FRAMES // _PAD_MULTIPLE) * _PAD_MULTIPLE
_MAXIMUM_CHUNK_FRAMES = (
    (_MAXIMUM_PADDED_FRAMES - STFT_WINDOW_LENGTH // 2) // _PAD_MULTIPLE
) * _PAD_MULTIPLE
_PLANNING_VERSION = "glm4-whisper-vq-pooling-v1"
_IDENTITY = AudioBackendIdentity(
    preset="glm4",
    artifact=DEFAULT_GLM4_ARTIFACT,
    revision=GLM4_IDENTITY_REVISION,
)
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


def _represent_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> AudioTokenizer:
    from .tokenizer import represent_tokenizer

    return represent_tokenizer(backend, code_spec, identity)


ENTRY = CodecEntry(
    name="glm4",
    loader=_load_tokenizer,
    audio_capabilities=frozenset({AudioCapability.TOKENIZER}),
    audio_spec=AudioCodeSpec(
        sample_rate=16_000,
        streams=(AudioCodeStreamSpec("rvq_0", 16_384, frame_rate=12.5),),
    ),
    tokenizer=_represent_tokenizer,
    identity=_IDENTITY,
    planning_estimator=chunked_pooled_planning_estimator(
        backend_identity=_IDENTITY,
        version=_PLANNING_VERSION,
        sample_rate=SAMPLE_RATE,
        feature_hop_length=FEATURE_HOP_LENGTH,
        convolution_stride=CONVOLUTION_STRIDE,
        pooling=POOLING_KERNEL_SIZE,
        maximum_chunk_frames=_MAXIMUM_CHUNK_FRAMES,
    ),
)

__all__ = [
    "CODEBOOK_SIZE",
    "CONVOLUTION_STRIDE",
    "DEFAULT_GLM4_ARTIFACT",
    "DEFAULT_GLM4_REVISION",
    "ENTRY",
    "FEATURE_DIM",
    "FEATURE_HOP_LENGTH",
    "FRAME_RATE",
    "GLM4_IDENTITY_REVISION",
    "GLM4_RUNTIME_GUIDANCE",
    "GLM4_SOURCE_ROOT_ENV",
    "MAX_SOURCE_POSITIONS",
    "POOLING_KERNEL_SIZE",
    "SAMPLE_RATE",
    "STFT_WINDOW_LENGTH",
]
