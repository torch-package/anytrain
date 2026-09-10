from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from anytrain.audio._tokenizer import PaddedAudioCodes
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

    @torch.no_grad()
    def tokenize_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> PaddedAudioCodes:
        """Encode one padded waveform batch and report per-stream valid lengths."""

        if audio.dim() != 3:
            raise ValueError("BiCodec padded tokenization expects [batch, channel, time].")
        if not isinstance(lengths, Tensor) or lengths.ndim != 1:
            raise ValueError("BiCodec padded waveform lengths must have shape [batch].")
        if lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("BiCodec padded waveform lengths must contain signed integers.")
        if lengths.numel() != audio.size(0):
            raise ValueError("BiCodec padded waveform lengths must align with the batch.")
        if lengths.numel() == 0 or bool((lengths <= 0).any()):
            raise ValueError("BiCodec padded waveform lengths must be positive.")
        if bool((lengths > audio.size(-1)).any()):
            raise ValueError("BiCodec padded waveform lengths exceed the waveform extent.")

        semantic, global_codes, semantic_lengths = self.codec.encode_codes_padded(
            audio,
            sample_rate,
            lengths,
        )
        codes = self._named(semantic[..., None], global_codes[..., None])
        if bool((semantic_lengths <= 0).any()) or bool(
            (semantic_lengths > codes["semantic"].size(1)).any()
        ):
            raise ValueError("BiCodec semantic valid lengths exceed the encoded stream extent.")
        return self._padded(
            codes=codes,
            lengths={
                "semantic": semantic_lengths.to(device=lengths.device, dtype=torch.long),
                "global": torch.full(
                    (audio.size(0),),
                    codes["global"].size(1),
                    dtype=torch.long,
                    device=lengths.device,
                ),
            },
        )


def represent_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> BiCodecTokenizer:
    return BiCodecTokenizer(backend, code_spec, identity)


__all__ = ["BiCodecTokenizer", "represent_tokenizer"]
