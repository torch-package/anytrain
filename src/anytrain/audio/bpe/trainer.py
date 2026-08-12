from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ._deps import tokenizers_version

TRAINER_IDENTITY_VERSION = "codec-bpe-trainer-v1"
DEFAULT_VOCAB_SIZE = 30_000
DEFAULT_MIN_FREQUENCY = 0
DEFAULT_MAX_TOKEN_LENGTH: int | None = None
DEFAULT_MAX_FRAMES: int | None = 1_000_000_000


@dataclass(frozen=True)
class CodecBPETrainerSpec:
    """Output-affecting CodecBPE trainer contract for reproducible artifacts."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    max_token_length: int | None = DEFAULT_MAX_TOKEN_LENGTH
    max_frames: int | None = DEFAULT_MAX_FRAMES
    tokenizers_version: str = field(default_factory=tokenizers_version)

    def __post_init__(self) -> None:
        _integer(self.vocab_size, name="vocab_size", minimum=1)
        _integer(self.min_frequency, name="min_frequency", minimum=0)
        if self.max_token_length is not None:
            _integer(self.max_token_length, name="max_token_length", minimum=1)
        if self.max_frames is not None:
            _integer(self.max_frames, name="max_frames", minimum=1)
        if not isinstance(self.tokenizers_version, str):
            raise TypeError("tokenizers_version must be a string.")
        if not self.tokenizers_version:
            raise ValueError("tokenizers_version must not be empty.")

    @property
    def identity(self) -> str:
        payload = {
            "implementation": "anytrain.audio.bpe.CodecBPE",
            "parameters": {
                "max_frames": self.max_frames,
                "max_token_length": self.max_token_length,
                "min_frequency": self.min_frequency,
                "vocab_size": self.vocab_size,
            },
            "tokenizers": {
                "model": "BPE",
                "trainer": "BpeTrainer",
                "version": self.tokenizers_version,
            },
            "version": TRAINER_IDENTITY_VERSION,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{TRAINER_IDENTITY_VERSION}:{digest}"


def _integer(value: object, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


__all__ = ["CodecBPETrainerSpec"]
