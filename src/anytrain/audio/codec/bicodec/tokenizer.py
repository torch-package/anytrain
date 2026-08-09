from __future__ import annotations

from typing import cast

from torch import Tensor

from anytrain.audio.codec._tokenizer import BackendTokenizer
from anytrain.audio.codec.protocol import CodecBackend
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec

from .codec import BiCodec


class BiCodecTokenizer(BackendTokenizer):
    @property
    def codec(self) -> BiCodec:
        return cast(BiCodec, self.backend)

    def _tokenize(self, audio: Tensor, sample_rate: int) -> dict[str, Tensor]:
        semantic, global_codes = self.codec.encode_codes(audio, sample_rate)
        return self._named(semantic[..., None], global_codes[..., None])


def represent_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> BiCodecTokenizer:
    return BiCodecTokenizer(backend, code_spec, identity)


__all__ = ["BiCodecTokenizer", "represent_tokenizer"]
