"""CosyVoice 2/3 inference backend."""

from .tts import (
    DEFAULT_MODEL_V2,
    DEFAULT_MODEL_V3,
    CosyVoiceLoadKwargs,
    CosyVoiceTTS,
    CosyVoiceTTSConfig,
    CosyVoiceVersion,
)

__all__ = [
    "DEFAULT_MODEL_V2",
    "DEFAULT_MODEL_V3",
    "CosyVoiceLoadKwargs",
    "CosyVoiceTTS",
    "CosyVoiceTTSConfig",
    "CosyVoiceVersion",
]
