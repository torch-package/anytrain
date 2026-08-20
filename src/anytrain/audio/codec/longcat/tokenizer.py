from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from anytrain.audio._tokenizer import PaddedAudioCodes
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

    @torch.no_grad()
    def tokenize_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> PaddedAudioCodes:
        """Encode one padded batch and crop every LongCat stream equally."""

        codes, frame_lengths = self.codec.encode_padded(audio, sample_rate, lengths)
        named = self._named(codes)
        return self._padded(
            codes=named,
            lengths={stream.name: frame_lengths for stream in self.code_spec.streams},
        )


def represent_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> LongCatTokenizer:
    return LongCatTokenizer(backend, code_spec, identity)


__all__ = ["LongCatTokenizer", "represent_tokenizer"]
