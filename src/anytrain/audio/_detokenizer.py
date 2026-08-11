from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from anytrain.audio.codec.spec import AudioCodeSpec


class AudioDetokenizer(ABC):
    """Named code-stream to waveform decoder with a shared convenience path."""

    @property
    @abstractmethod
    def code_spec(self) -> AudioCodeSpec: ...

    @abstractmethod
    def codes_to_features(self, stream: str, codes: Tensor) -> Tensor: ...

    @abstractmethod
    def decode_features(self, features: Mapping[str, Tensor | None]) -> Tensor: ...

    @torch.no_grad()
    def detokenize(self, codes: Mapping[str, Tensor | None]) -> Tensor:
        streams = self._ordered(codes, value_name="codes")
        features: dict[str, Tensor | None] = {}
        for stream in self.code_spec.streams:
            stream_codes = streams[stream.name]
            features[stream.name] = (
                None
                if stream_codes is None
                else self.codes_to_features(stream.name, stream_codes)
            )
        return self.decode_features(features)

    def _ordered(
        self,
        values: Mapping[str, Tensor | None],
        *,
        value_name: str,
    ) -> dict[str, Tensor | None]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{value_name} must be a mapping from stream names to tensors.")
        expected = {stream.name for stream in self.code_spec.streams}
        actual = set(values)
        if actual != expected:
            missing = tuple(
                stream.name
                for stream in self.code_spec.streams
                if stream.name not in actual
            )
            unexpected = tuple(name for name in values if name not in expected)
            raise ValueError(
                f"{value_name} do not match code_spec: "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )
        ordered: dict[str, Tensor | None] = {}
        for stream in self.code_spec.streams:
            value = values[stream.name]
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(
                    f"{value_name}[{stream.name!r}] must be a Tensor or None."
                )
            ordered[stream.name] = value
        return ordered


__all__ = ["AudioDetokenizer"]
