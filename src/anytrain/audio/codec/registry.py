from __future__ import annotations

from threading import RLock
from typing import cast

import torch

from anytrain.audio._detokenizer import AudioDetokenizer
from anytrain.audio._tokenizer import AudioTokenizer

from ._registry import CodecEntry
from .bicodec._registry import ENTRY as BICODEC_ENTRY
from .dac._registry import ENTRY as DAC_ENTRY
from .glm4._registry import ENTRY as GLM4_ENTRY
from .longcat._registry import ENTRY as LONGCAT_ENTRY
from .planning import AudioPlanningEstimator
from .protocol import AudioCapability, CodecBackend
from .spec import AudioBackendIdentity, AudioCodeSpec
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


def audio_capabilities(name: str) -> frozenset[AudioCapability]:
    return _entry(name).audio_capabilities


def audio_spec(name: str) -> AudioCodeSpec:
    return _audio_spec(_entry(name))


def audio_backend_identity(name: str) -> AudioBackendIdentity:
    return _entry(name).resolved_identity


def audio_planning_estimator(name: str) -> AudioPlanningEstimator:
    entry = _entry(name)
    estimator = entry.planning_estimator
    if estimator is None:
        raise ValueError(
            f"Audio backend {name!r} does not provide a static planning estimator."
        )
    return estimator


def can_load_audio_tokenizer(name: str) -> bool:
    _validate_name(name)
    entry = _ENTRIES.get(name)
    return bool(
        entry is not None
        and AudioCapability.TOKENIZER in entry.audio_capabilities
        and entry.loader is not None
        and entry.tokenizer is not None
    )


def can_load_audio_detokenizer(name: str) -> bool:
    _validate_name(name)
    entry = _ENTRIES.get(name)
    return bool(
        entry is not None
        and AudioCapability.DETOKENIZER in entry.audio_capabilities
        and entry.loader is not None
        and entry.detokenizer is not None
    )


def clear_audio_backend_cache() -> None:
    with _BACKEND_LOCK:
        _BACKENDS.clear()


def _entry(name: str) -> CodecEntry:
    _validate_name(name)
    try:
        return _ENTRIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENTRIES))
        raise ValueError(f"Unknown codec {name!r}. Available codecs: {available}.") from exc


def _validate_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError("codec name must be a string.")
    if not name:
        raise ValueError("codec name must not be empty.")


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
    resolved = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else "cpu"
        if device is None
        else device
    )
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", torch.cuda.current_device())
    return str(resolved)


def _load_backend(entry: CodecEntry, *, device: str | torch.device | None) -> CodecBackend:
    loader = entry.loader
    if loader is None:
        raise NotImplementedError(f"Audio backend {entry.name!r} has no online loader.")
    key = (entry.resolved_identity, _device_key(device))
    with _BACKEND_LOCK:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = cast(CodecBackend, loader(device=device))
            _BACKENDS[key] = backend
        return backend


def load_audio_backend(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> CodecBackend:
    """Load and cache the registered backend without selecting a waveform view."""
    return _load_backend(_entry(name), device=device)


def load_audio_tokenizer(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> AudioTokenizer:
    entry = _require_audio_capability(name, AudioCapability.TOKENIZER)
    represent = entry.tokenizer
    if represent is None:
        raise RuntimeError(f"Audio backend {entry.name!r} has no tokenizer representation.")
    return represent(_load_backend(entry, device=device), _audio_spec(entry), entry.resolved_identity)


def load_audio_detokenizer(
    name: str,
    *,
    device: str | torch.device | None = None,
) -> AudioDetokenizer:
    entry = _require_audio_capability(name, AudioCapability.DETOKENIZER)
    represent = entry.detokenizer
    if represent is None:
        raise RuntimeError(f"Audio backend {entry.name!r} has no detokenizer representation.")
    return represent(_load_backend(entry, device=device), _audio_spec(entry), entry.resolved_identity)


__all__ = [
    "audio_backend_identity",
    "audio_capabilities",
    "audio_planning_estimator",
    "audio_spec",
    "can_load_audio_detokenizer",
    "can_load_audio_tokenizer",
    "clear_audio_backend_cache",
    "load_audio_backend",
    "load_audio_detokenizer",
    "load_audio_tokenizer",
]
