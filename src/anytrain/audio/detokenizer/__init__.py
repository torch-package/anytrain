"""Codes → waveform capability views and loaders.

Concrete backends live under ``anytrain.audio.codec``; this package only exposes
the detokenizer direction of the shared audio protocol.
"""

from anytrain.audio.codec import (
    AudioDetokenizer,
    FrameDetokenizerBackend,
    can_load_audio_detokenizer,
    load_audio_detokenizer,
)

__all__ = [
    "AudioDetokenizer",
    "FrameDetokenizerBackend",
    "can_load_audio_detokenizer",
    "load_audio_detokenizer",
]
