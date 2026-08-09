from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from anytrain.audio.codec.spec import AudioCodeSpec

AudioCodes = Mapping[str, Tensor]
_SIGNED_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


class AudioTokenizer(ABC):
    """Waveform to complete named code-stream inventory."""

    @property
    @abstractmethod
    def code_spec(self) -> AudioCodeSpec: ...

    @abstractmethod
    def _tokenize(self, audio: Tensor, sample_rate: int) -> AudioCodes: ...

    @torch.no_grad()
    def tokenize(self, audio: Tensor, sample_rate: int) -> dict[str, Tensor]:
        codes = self._tokenize(audio, sample_rate)
        if not isinstance(codes, Mapping):
            raise TypeError("tokenizer output must map stream names to tensors.")
        expected = set(self.code_spec.names)
        actual = set(codes)
        if actual != expected:
            missing = tuple(name for name in self.code_spec.names if name not in actual)
            unexpected = tuple(name for name in codes if name not in expected)
            raise ValueError(
                "tokenizer output does not match code_spec: "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )
        ordered: dict[str, Tensor] = {}
        batch_size: int | None = None
        for index, stream in enumerate(self.code_spec.names):
            value = codes[stream]
            if not isinstance(value, Tensor) or value.dim() != 2:
                raise ValueError(f"tokenizer output {stream!r} must have shape [batch, unit].")
            if value.dtype not in _SIGNED_INTEGER_DTYPES:
                raise TypeError(
                    f"tokenizer output {stream!r} must contain signed integer IDs."
                )
            if value.numel() == 0:
                raise ValueError(f"tokenizer output {stream!r} must not be empty.")
            if batch_size is None:
                batch_size = value.size(0)
            elif value.size(0) != batch_size:
                raise ValueError("tokenizer output streams must align on batch.")
            if bool((value < 0).any()) or bool(
                (value >= self.code_spec.vocab_sizes[index]).any()
            ):
                raise ValueError(f"tokenizer output {stream!r} contains an invalid code ID.")
            ordered[stream] = value
        return ordered


__all__ = ["AudioCodes", "AudioTokenizer"]
