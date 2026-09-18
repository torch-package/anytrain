"""Temporary reference files shared by TTS adapters.

Adapters retain their waveform normalization and legacy argument policies.
Files stay alive until the complete generation operation leaves the context.
"""
from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Self

from .protocol import AudioReference, EncodedAudioReference, WaveformReference


class ReferenceFiles:
    def __init__(
        self, *, prefix: str,
        save_waveform: Callable[[WaveformReference, Path], None],
        invalid: str = "reference audio must be a path, WaveformReference, or EncodedAudioReference.",
    ) -> None:
        self.prefix = prefix
        self.save_waveform = save_waveform
        self.invalid = invalid
        self._temporary: TemporaryDirectory[str] | None = None

    def path(self, value: AudioReference, *, index: int | None = None) -> str:
        if isinstance(value, (str, PathLike)):
            return str(value)
        if not isinstance(value, (WaveformReference, EncodedAudioReference)):
            raise TypeError(self.invalid)
        if self._temporary is None:
            self._temporary = TemporaryDirectory(prefix=self.prefix)
        root = Path(self._temporary.name)
        stem = "reference" if index is None else f"reference-{index}"
        if isinstance(value, WaveformReference):
            path = root / f"{stem}.wav"
            self.save_waveform(value, path)
        else:
            path = root / f"{stem}{value.suffix}"
            path.write_bytes(value.data)
        return str(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
