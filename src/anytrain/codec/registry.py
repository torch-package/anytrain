from __future__ import annotations

from threading import RLock
from typing import cast

import torch
from torch import Tensor

from ._registry import CodecEntry
from .bicodec._registry import ENTRY as BICODEC_ENTRY
from .dac._registry import ENTRY as DAC_ENTRY
from .glm4._registry import ENTRY as GLM4_ENTRY
from .longcat._registry import ENTRY as LONGCAT_ENTRY
from .protocol import (
    AudioCapability,
    AudioCodes,
    AudioDetokenizer,
    AudioTokenizer,
    CodecBackend,
    CodecCapability,
    FrameCodec,
    FrameDetokenizerBackend,
    FrameTokenizerBackend,
    SemanticAcousticCodec,
    SemanticAcousticCodes,
    SemanticGlobalCodec,
    SemanticGlobalCodes,
)
from .spec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec
from .stable_codec._registry import ENTRY as STABLE_CODEC_ENTRY
from .unicodec._registry import ENTRY as UNICODEC_ENTRY

_ENTRIES: dict[str, CodecEntry] = {
    entry.name: entry
    for entry in (
        BICODEC_ENTRY,
        DAC_ENTRY,
        GLM4_ENTRY,
        LONGCAT_ENTRY,
        STABLE_CODEC_ENTRY,
        UNICODEC_ENTRY,
    )
}
_BACKENDS: dict[tuple[AudioBackendIdentity, str], CodecBackend] = {}
_BACKEND_LOCK = RLock()


def capabilities(name: str) -> frozenset[CodecCapability]:
    """Return statically declared capabilities without loading a codec model."""
    return _entry(name).capabilities


def audio_capabilities(name: str) -> frozenset[AudioCapability]:
    """Return declared waveform conversion directions without loading a model."""

    return _entry(name).audio_capabilities


def audio_spec(name: str) -> AudioCodeSpec:
    """Return the static code schema for a registered default preset."""

    return _audio_spec(_entry(name))


def audio_backend_identity(name: str) -> AudioBackendIdentity:
    """Return the resolved resource identity used for backend instance sharing."""

    return _entry(name).resolved_identity


def can_load_audio_tokenizer(name: str) -> bool:
    """Whether anytrain implements an online loader for the declared tokenizer."""

    entry = _ENTRIES.get(name)
    return (
        entry is not None
        and AudioCapability.TOKENIZER in entry.audio_capabilities
        and entry.loader is not None
    )


def can_load_audio_detokenizer(name: str) -> bool:
    """Whether anytrain implements an online loader for the declared detokenizer."""

    entry = _ENTRIES.get(name)
    return (
        entry is not None
        and AudioCapability.DETOKENIZER in entry.audio_capabilities
        and entry.loader is not None
    )


def clear_audio_backend_cache() -> None:
    """Drop registry-owned backend references, primarily for explicit lifecycle control."""

    with _BACKEND_LOCK:
        _BACKENDS.clear()


def _entry(name: str) -> CodecEntry:
    try:
        return _ENTRIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENTRIES))
        raise ValueError(f"Unknown codec {name!r}. Available codecs: {available}.") from exc


def _require_capability(name: str, capability: CodecCapability) -> CodecEntry:
    entry = _entry(name)
    if capability not in entry.capabilities:
        raise ValueError(f"Codec {name!r} does not provide the {capability.value!r} capability.")
    return entry


def _require_audio_capability(name: str, capability: AudioCapability) -> CodecEntry:
    entry = _entry(name)
    if capability not in entry.audio_capabilities:
        raise ValueError(
            f"Audio backend {name!r} does not provide the {capability.value!r} capability."
        )
    return entry


def _audio_spec(entry: CodecEntry) -> AudioCodeSpec:
    spec = entry.audio_spec
    if spec is None:
        raise RuntimeError(f"Audio backend {entry.name!r} has no static code spec.")
    return spec


def _device_key(device: str | torch.device | None) -> str:
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", torch.cuda.current_device())
    return str(resolved)


def _load_backend(
    entry: CodecEntry,
    *,
    device: str | torch.device | None,
) -> CodecBackend:
    loader = entry.loader
    if loader is None:
        raise NotImplementedError(
            f"Audio backend {entry.name!r} declares static capabilities but anytrain "
            "does not implement an online loader for this preset."
        )
    key = (entry.resolved_identity, _device_key(device))
    with _BACKEND_LOCK:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = cast(CodecBackend, loader(device=device))
            _BACKENDS[key] = backend
        return backend


class _AudioTokenizerView:
    def __init__(self, entry: CodecEntry, backend: CodecBackend) -> None:
        self._entry = entry
        self._backend = backend

    @property
    def backend(self) -> CodecBackend:
        return self._backend

    @property
    def identity(self) -> AudioBackendIdentity:
        return self._entry.resolved_identity

    @property
    def spec(self) -> AudioCodeSpec:
        return _audio_spec(self._entry)

    def tokenize(self, audio: Tensor, sample_rate: int) -> AudioCodes:
        if self.spec.schema is AudioCodeSchema.FRAME:
            return cast(FrameTokenizerBackend, self.backend).encode(audio, sample_rate)
        if self.spec.schema is AudioCodeSchema.SEMANTIC_ACOUSTIC:
            return cast(SemanticAcousticCodec, self.backend).tokenize(audio, sample_rate)
        return cast(SemanticGlobalCodec, self.backend).tokenize(audio, sample_rate)


class _AudioDetokenizerView:
    def __init__(self, entry: CodecEntry, backend: CodecBackend) -> None:
        self._entry = entry
        self._backend = backend

    @property
    def backend(self) -> CodecBackend:
        return self._backend

    @property
    def identity(self) -> AudioBackendIdentity:
        return self._entry.resolved_identity

    @property
    def spec(self) -> AudioCodeSpec:
        return _audio_spec(self._entry)

    def detokenize(self, codes: AudioCodes) -> Tensor:
        if self.spec.schema is AudioCodeSchema.FRAME:
            return cast(FrameDetokenizerBackend, self.backend).decode(cast(Tensor, codes))
        if self.spec.schema is AudioCodeSchema.SEMANTIC_ACOUSTIC:
            return cast(SemanticAcousticCodec, self.backend).detokenize(
                cast(SemanticAcousticCodes, codes)
            )
        return cast(SemanticGlobalCodec, self.backend).detokenize(cast(SemanticGlobalCodes, codes))


def load_audio_tokenizer(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> AudioTokenizer[AudioCodes]:
    """Load a tokenizer view over the preset's shared backend instance."""

    entry = _require_audio_capability(name, AudioCapability.TOKENIZER)
    return _AudioTokenizerView(entry, _load_backend(entry, device=device))


def load_audio_detokenizer(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> AudioDetokenizer[AudioCodes]:
    """Load a detokenizer view over the preset's shared backend instance."""

    entry = _require_audio_capability(name, AudioCapability.DETOKENIZER)
    return _AudioDetokenizerView(entry, _load_backend(entry, device=device))


def load_semantic_acoustic(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> SemanticAcousticCodec:
    """Load a default codec implementing the semantic-acoustic contract."""
    entry = _require_capability(name, CodecCapability.SEMANTIC_ACOUSTIC)
    return cast(SemanticAcousticCodec, _load_backend(entry, device=device))


def load_semantic_global(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> SemanticGlobalCodec:
    """Load a default codec implementing the semantic-global contract."""
    entry = _require_capability(name, CodecCapability.SEMANTIC_GLOBAL)
    return cast(SemanticGlobalCodec, _load_backend(entry, device=device))


def load_frame(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> FrameCodec:
    """Load a default codec implementing the complete frame-code contract."""
    entry = _require_capability(name, CodecCapability.FRAME_CODES)
    return cast(FrameCodec, _load_backend(entry, device=device))


__all__ = [
    "audio_backend_identity",
    "audio_capabilities",
    "audio_spec",
    "can_load_audio_detokenizer",
    "can_load_audio_tokenizer",
    "capabilities",
    "clear_audio_backend_cache",
    "load_audio_detokenizer",
    "load_audio_tokenizer",
    "load_frame",
    "load_semantic_acoustic",
    "load_semantic_global",
]
