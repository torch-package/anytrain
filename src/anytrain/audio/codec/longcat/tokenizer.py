from __future__ import annotations

from typing import cast

from torch import Tensor

from anytrain.audio.codec._tokenizer import BackendTokenizer
from anytrain.audio.codec.protocol import CodecBackend
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec

from .codec import LongCat


class LongCatTokenizer(BackendTokenizer):
    @property
    def codec(self) -> LongCat:
        return cast(LongCat, self.backend)

    def _tokenize(self, audio: Tensor, sample_rate: int) -> dict[str, Tensor]:
        codes = self.codec.encode(audio, sample_rate)
        return self._named(codes)


def represent_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> LongCatTokenizer:
    return LongCatTokenizer(backend, code_spec, identity)


__all__ = ["LongCatTokenizer", "represent_tokenizer"]
