from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor

from anytrain.audio.codec._detokenizer import BackendDetokenizer
from anytrain.audio.codec.protocol import CodecBackend
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec

from .codec import LongCat


class LongCatDetokenizer(BackendDetokenizer):
    @property
    def codec(self) -> LongCat:
        return cast(LongCat, self.backend)

    @torch.no_grad()
    def codes_to_features(self, stream: str, codes: Tensor) -> Tensor:
        values, index = self._stream_codes(stream, codes)
        if index == 0:
            features = self.codec.semantic_codes_to_features(values)
        else:
            features = self.codec.acoustic_codebook_codes_to_features(index - 1, values)
        return _features(features, stream=stream, shape=values.shape)

    @torch.no_grad()
    def decode_features(self, features: Mapping[str, Tensor | None]) -> Tensor:
        streams = self._ordered(features, value_name="features")
        semantic_name = self.code_spec.names[0]
        semantic_value = streams[semantic_name]
        if semantic_value is None:
            raise ValueError("LongCat cannot infer waveform duration without semantic features.")
        semantic = _features(semantic_value, stream=semantic_name)
        acoustic_streams: list[Tensor] = []
        for name in self.code_spec.names[1:]:
            value = streams[name]
            if value is not None:
                acoustic_streams.append(_features(value, stream=name, shape=semantic.shape[:2]))
        if not acoustic_streams:
            zero = semantic.new_zeros(
                semantic.size(0),
                semantic.size(1),
                self.codec.acoustic_feature_dim,
            )
            return cast(Tensor, self.codec.decode_acoustic_features(semantic, zero))
        if len(acoustic_streams) == 1:
            return cast(
                Tensor,
                self.codec.decode_acoustic_features(semantic, acoustic_streams[0]),
            )
        return cast(
            Tensor,
            self.codec.decode_feature_streams(semantic, tuple(acoustic_streams)),
        )


def represent_detokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> LongCatDetokenizer:
    return LongCatDetokenizer(backend, code_spec, identity)


def _features(
    value: object,
    *,
    stream: str,
    shape: tuple[int, ...] | torch.Size | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"{stream} features must have shape [batch, unit, feature].")
    if shape is not None and value.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"{stream} features must align with its code units.")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"{stream} features must be floating point tensors.")
    return value


__all__ = ["LongCatDetokenizer", "represent_detokenizer"]
