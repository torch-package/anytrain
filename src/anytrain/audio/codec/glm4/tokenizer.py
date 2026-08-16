"""Padded-batch tokenizer representation for the GLM-4 backend."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from anytrain.audio._tokenizer import PaddedAudioCodes
from anytrain.audio.codec._tokenizer import BackendTokenizer
from anytrain.audio.codec.protocol import CodecBackend
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec

from .codec import GLM4Tokenizer


class GLM4AudioTokenizer(BackendTokenizer):
    """Expose equal-length and padded GLM-4 waveform batches."""

    @property
    def tokenizer(self) -> GLM4Tokenizer:
        return cast(GLM4Tokenizer, self.backend)

    def _tokenize(self, audio: Tensor, sample_rate: int) -> dict[str, Tensor]:
        return self._named(self.tokenizer.encode(audio, sample_rate))

    @torch.no_grad()
    def tokenize_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> PaddedAudioCodes:
        codes, code_lengths = self.tokenizer.encode_padded(
            audio,
            sample_rate,
            lengths,
        )
        return self._padded(
            codes=self._named(codes),
            lengths={"rvq_0": code_lengths},
        )


def represent_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> GLM4AudioTokenizer:
    return GLM4AudioTokenizer(backend, code_spec, identity)


__all__ = ["GLM4AudioTokenizer", "represent_tokenizer"]
