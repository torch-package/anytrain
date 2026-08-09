from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor

from anytrain.audio.codec._detokenizer import BackendDetokenizer
from anytrain.audio.codec.protocol import CodecBackend
from anytrain.audio.codec.spec import AudioBackendIdentity, AudioCodeSpec

from .codec import BiCodec


class BiCodecDetokenizer(BackendDetokenizer):
    @property
    def codec(self) -> BiCodec:
        return cast(BiCodec, self.backend)

    @torch.no_grad()
    def codes_to_features(self, stream: str, codes: Tensor) -> Tensor:
        values, index = self._stream_codes(stream, codes)
        if index == 0:
            convert = getattr(self.codec, "semantic_codes_to_features", None)
            if not callable(convert):
                raise TypeError("BiCodec backend must provide semantic_codes_to_features().")
            features = convert(values)
        else:
            features = self.codec.global_codes_to_features(values[..., None])
        return _features(features, stream=stream, batch_size=values.size(0))

    @torch.no_grad()
    def decode_features(self, features: Mapping[str, Tensor | None]) -> Tensor:
        streams = self._ordered(features, value_name="features")
        semantic_name, global_name = self.code_spec.names
        semantic_value = streams[semantic_name]
        if semantic_value is None:
            raise ValueError("BiCodec cannot infer waveform duration without semantic features.")
        semantic = _features(semantic_value, stream=semantic_name)
        global_value = streams[global_name]
        global_features = (
            semantic.new_zeros(
                semantic.size(0),
                self.codec.global_unit_length,
                self.codec.global_feature_dim,
            )
            if global_value is None
            else _features(
                global_value,
                stream=global_name,
                batch_size=semantic.size(0),
            )
        )
        decode = getattr(self.codec, "decode_feature_streams", None)
        if not callable(decode):
            raise TypeError("BiCodec backend must provide decode_feature_streams().")
        return cast(Tensor, decode(semantic, global_features))


def represent_detokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> BiCodecDetokenizer:
    return BiCodecDetokenizer(backend, code_spec, identity)


def _features(
    value: object,
    *,
    stream: str,
    batch_size: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"{stream} features must have shape [batch, unit, feature].")
    if batch_size is not None and value.size(0) != batch_size:
        raise ValueError("BiCodec feature streams must align on batch.")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"{stream} features must be floating point tensors.")
    return value


__all__ = ["BiCodecDetokenizer", "represent_detokenizer"]
