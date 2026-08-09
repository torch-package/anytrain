from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from torch import Tensor

from anytrain.audio._detokenizer import AudioDetokenizer

from ._module import integer_tensor

if TYPE_CHECKING:
    from .protocol import CodecBackend
    from .spec import AudioBackendIdentity, AudioCodeSpec


class BackendDetokenizer(AudioDetokenizer):
    """Preset metadata plus a shared backend reference for concrete adapters."""

    def __init__(
        self,
        backend: CodecBackend,
        code_spec: AudioCodeSpec,
        identity: AudioBackendIdentity,
    ) -> None:
        self._backend = backend
        self._code_spec = code_spec
        self._identity = identity

    @property
    def backend(self) -> CodecBackend:
        return self._backend

    @property
    def code_spec(self) -> AudioCodeSpec:
        return self._code_spec

    @property
    def identity(self) -> AudioBackendIdentity:
        return self._identity

    def _stream_codes(self, stream: str, codes: Tensor) -> tuple[Tensor, int]:
        try:
            index = self.code_spec.names.index(stream)
        except ValueError as exc:
            raise ValueError(
                f"unknown code stream {stream!r}; available streams are {self.code_spec.names!r}."
            ) from exc
        values = integer_tensor(codes, f"{stream} codes")
        if values.dim() != 2:
            raise ValueError(f"{stream} codes must have shape [batch, unit].")
        if values.numel() == 0:
            raise ValueError(f"{stream} codes must not be empty.")
        if bool((values < 0).any()) or bool((values >= self.code_spec.vocab_sizes[index]).any()):
            raise ValueError(f"{stream} codes contain an ID outside the codebook.")
        return values, index


class FrameBackendDetokenizer(BackendDetokenizer):
    def codes_to_features(self, stream: str, codes: Tensor) -> Tensor:
        values, index = self._stream_codes(stream, codes)
        convert = getattr(self.backend, "codebook_codes_to_features", None)
        if not callable(convert):
            raise TypeError(
                f"Audio backend {self.identity.preset!r} must provide codebook_codes_to_features()."
            )
        features = convert(index, values)
        return _features(features, stream=stream, shape=values.shape)

    def decode_features(self, features: Mapping[str, Tensor | None]) -> Tensor:
        streams = self._ordered(features, value_name="features")
        missing = tuple(name for name in self.code_spec.names if streams[name] is None)
        if missing:
            raise ValueError(f"frame detokenizer cannot infer omitted feature streams {missing!r}.")
        values: list[Tensor] = []
        for name in self.code_spec.names:
            value = streams[name]
            if value is None:
                raise AssertionError("omitted frame streams were checked above.")
            values.append(_features(value, stream=name))
        decode = getattr(self.backend, "decode_codebook_features", None)
        if not callable(decode):
            raise TypeError(
                f"Audio backend {self.identity.preset!r} must provide decode_codebook_features()."
            )
        return cast(Tensor, decode(tuple(values)))


def represent_frame_detokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> FrameBackendDetokenizer:
    return FrameBackendDetokenizer(backend, code_spec, identity)


def _features(
    value: Any,
    *,
    stream: str,
    shape: tuple[int, ...] | Any | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"{stream} features must have shape [batch, unit, feature].")
    if shape is not None and value.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"{stream} features must align with its code units.")
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{stream} features must be floating point tensors.")
    return value


__all__ = [
    "BackendDetokenizer",
    "FrameBackendDetokenizer",
    "represent_frame_detokenizer",
]
