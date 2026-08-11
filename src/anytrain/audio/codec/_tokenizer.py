from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

from anytrain.audio._tokenizer import AudioTokenizer

if TYPE_CHECKING:
    from .protocol import CodecBackend
    from .spec import AudioBackendIdentity, AudioCodeSpec


class BackendTokenizer(AudioTokenizer):
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

    def _named(self, *groups: Tensor) -> dict[str, Tensor]:
        if not groups:
            raise ValueError("audio tokenizer must return at least one code group.")
        batch_size: int | None = None
        streams: tuple[Tensor, ...] = ()
        for group in groups:
            if not isinstance(group, Tensor) or group.dim() != 3:
                raise ValueError(
                    "audio tokenizer code groups must have shape [batch, unit, codebook]."
                )
            if batch_size is None:
                batch_size = group.size(0)
            elif group.size(0) != batch_size:
                raise ValueError("audio tokenizer code groups must align on batch.")
            streams += group.unbind(dim=-1)
        if len(streams) != len(self.code_spec.streams):
            raise ValueError("audio tokenizer output must contain every code_spec stream.")
        return {
            stream.name: streams[index]
            for index, stream in enumerate(self.code_spec.streams)
        }


class FrameBackendTokenizer(BackendTokenizer):
    def _tokenize(self, audio: Tensor, sample_rate: int) -> dict[str, Tensor]:
        encode = getattr(self.backend, "encode", None)
        if not callable(encode):
            raise TypeError(
                f"Audio backend {self.identity.preset!r} must provide encode()."
            )
        return self._named(cast(Tensor, encode(audio, sample_rate)))


def represent_frame_tokenizer(
    backend: CodecBackend,
    code_spec: AudioCodeSpec,
    identity: AudioBackendIdentity,
) -> FrameBackendTokenizer:
    return FrameBackendTokenizer(backend, code_spec, identity)


__all__ = [
    "BackendTokenizer",
    "FrameBackendTokenizer",
    "represent_frame_tokenizer",
]
