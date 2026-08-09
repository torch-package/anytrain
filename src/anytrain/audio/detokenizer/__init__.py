"""Named code streams to features to waveform detokenizers.

Concrete backends live under ``anytrain.audio.codec``; this package only exposes
the detokenizer direction and its shared convenience implementation.
"""

from anytrain.audio._detokenizer import AudioDetokenizer
from anytrain.audio.codec import (
    AudioCodeSpec,
    audio_spec,
    can_load_audio_detokenizer,
    load_audio_detokenizer,
)

__all__ = [
    "AudioDetokenizer",
    "AudioCodeSpec",
    "audio_spec",
    "can_load_audio_detokenizer",
    "load_audio_detokenizer",
]
