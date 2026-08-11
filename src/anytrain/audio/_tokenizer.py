from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class PaddedAudioCodes:
    """Named padded token streams and each stream's valid units per sample."""

    codes: Mapping[str, Tensor]
    lengths: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.codes, Mapping) or not isinstance(self.lengths, Mapping):
            raise TypeError("padded audio codes and lengths must be mappings.")
        if set(self.codes) != set(self.lengths):
            raise ValueError("padded audio code streams and lengths must match.")
        if not self.codes:
            raise ValueError("padded audio codes must contain at least one stream.")
        batch_size: int | None = None
        for name, lengths in self.lengths.items():
            if not isinstance(name, str) or not name:
                raise ValueError("padded audio stream names must be non-empty strings.")
            codes = self.codes[name]
            if not isinstance(codes, Tensor) or codes.ndim != 2:
                raise ValueError(
                    f"padded audio codes for {name!r} must have shape [batch, unit]."
                )
            if codes.dtype not in _SIGNED_INTEGER_DTYPES:
                raise TypeError(
                    f"padded audio codes for {name!r} must contain signed integer IDs."
                )
            if codes.numel() == 0:
                raise ValueError(f"padded audio codes for {name!r} must not be empty.")
            if not isinstance(lengths, Tensor) or lengths.ndim != 1:
                raise ValueError(f"padded audio lengths for {name!r} must have shape [batch].")
            if lengths.dtype not in _SIGNED_INTEGER_DTYPES:
                raise TypeError(
                    f"padded audio lengths for {name!r} must contain signed integers."
                )
            if codes.size(0) != lengths.size(0):
                raise ValueError(
                    f"padded audio codes and lengths for {name!r} must align on batch."
                )
            if batch_size is None:
                batch_size = lengths.size(0)
            elif lengths.size(0) != batch_size:
                raise ValueError("padded audio streams must align on batch.")
            if bool(((lengths <= 0) | (lengths > codes.size(1))).any()):
                raise ValueError(
                    f"padded audio lengths for {name!r} must be positive and not exceed "
                    "the code extent."
                )


@runtime_checkable
class PaddedAudioTokenizer(Protocol):
    """Optional capability for one padded variable-length waveform batch."""

    @property
    def code_spec(self) -> AudioCodeSpec: ...

    def tokenize_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> PaddedAudioCodes: ...


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
        expected = {stream.name for stream in self.code_spec.streams}
        actual = set(codes)
        if actual != expected:
            missing = tuple(
                stream.name
                for stream in self.code_spec.streams
                if stream.name not in actual
            )
            unexpected = tuple(name for name in codes if name not in expected)
            raise ValueError(
                "tokenizer output does not match code_spec: "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )
        ordered: dict[str, Tensor] = {}
        batch_size: int | None = None
        for stream in self.code_spec.streams:
            value = codes[stream.name]
            if not isinstance(value, Tensor) or value.dim() != 2:
                raise ValueError(
                    f"tokenizer output {stream.name!r} must have shape [batch, unit]."
                )
            if value.dtype not in _SIGNED_INTEGER_DTYPES:
                raise TypeError(
                    f"tokenizer output {stream.name!r} must contain signed integer IDs."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"tokenizer output {stream.name!r} must not be empty."
                )
            if batch_size is None:
                batch_size = value.size(0)
            elif value.size(0) != batch_size:
                raise ValueError("tokenizer output streams must align on batch.")
            if bool((value < 0).any()) or bool(
                (value >= stream.vocab_size).any()
            ):
                raise ValueError(
                    f"tokenizer output {stream.name!r} contains an invalid code ID."
                )
            ordered[stream.name] = value
        return ordered


__all__ = [
    "AudioCodes",
    "AudioTokenizer",
    "PaddedAudioCodes",
    "PaddedAudioTokenizer",
]
