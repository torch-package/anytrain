"""Exact prompt-token planning for Qwen3-TTS CustomVoice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor

from anytrain._semantic import semantic_id
from anytrain.tts.protocol import validate_text

from .tts import DEFAULT_MODEL

CUSTOM_VOICE_PROMPT_PROTOCOL = "qwen3-tts-customvoice-assistant-v1"
_IDENTITY_PREFIX = "qwen-customvoice-prompt-v1"


@dataclass(kw_only=True)
class QwenCustomVoicePromptEstimator:
    """Count the exact input tokens used by Qwen3-TTS CustomVoice."""

    revision: str
    model: str = DEFAULT_MODEL
    prompt_protocol: str = CUSTOM_VOICE_PROMPT_PROTOCOL
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    _processor: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _logical_model(self.model)
        _commit(self.revision)
        if self.prompt_protocol != CUSTOM_VOICE_PROMPT_PROTOCOL:
            raise ValueError(
                f"unsupported Qwen CustomVoice prompt protocol {self.prompt_protocol!r}."
            )
        if self.cache_dir is not None and not isinstance(self.cache_dir, (str, Path)):
            raise TypeError("cache_dir must be a string, Path, or None.")
        if isinstance(self.cache_dir, str) and not self.cache_dir.strip():
            raise ValueError("cache_dir must not be empty when set.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be a boolean.")

    @property
    def processor_loaded(self) -> bool:
        return self._processor is not None

    @property
    def identity(self) -> str:
        return semantic_id(
            _IDENTITY_PREFIX,
            {
                "model": self.model,
                "prompt_protocol": self.prompt_protocol,
                "revision": self.revision,
            },
        )

    def estimate(self, text: str) -> int:
        processor = self._load_processor()
        if not callable(processor):
            raise TypeError("Qwen CustomVoice processor must be callable.")
        encoded = processor(
            text=_assistant_prompt(validate_text(text)),
            return_tensors="pt",
            padding=True,
        )
        return _token_count(encoded)

    def _load_processor(self) -> Any:
        if self._processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "Qwen CustomVoice prompt estimation requires transformers; "
                    "install `anytrain[qwen-tts]`."
                ) from exc

            options: dict[str, object] = {
                "fix_mistral_regex": True,
                "local_files_only": self.local_files_only,
                "revision": self.revision,
            }
            if self.cache_dir is not None:
                options["cache_dir"] = str(self.cache_dir)
            self._processor = AutoProcessor.from_pretrained(self.model, **options)
        return self._processor

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = dict(self.__dict__)
        state["_processor"] = None
        return state


def _assistant_prompt(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _token_count(encoded: object) -> int:
    if not isinstance(encoded, Mapping):
        raise TypeError("Qwen CustomVoice processor output must be a mapping.")
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, Tensor):
        raise TypeError("Qwen CustomVoice processor input_ids must be a Tensor.")
    if input_ids.ndim == 1:
        count = int(input_ids.shape[0])
    elif input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("Qwen CustomVoice processor input_ids must contain exactly one text.")
        count = int(input_ids.shape[1])
    else:
        raise ValueError("Qwen CustomVoice processor input_ids must have rank 1 or rank 2.")
    if count <= 0:
        raise ValueError("Qwen CustomVoice processor input_ids must not be empty.")
    return count


def _logical_model(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model must be a string.")
    if not value.strip():
        raise ValueError("model must not be empty.")
    if value != value.strip():
        raise ValueError("model must not contain surrounding whitespace.")
    if Path(value).is_absolute():
        raise ValueError("model must be a logical artifact id, not an absolute path.")
    return value


def _commit(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("revision must be a string.")
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("revision must be a lowercase 40-character commit hash.")
    return value


__all__ = ["CUSTOM_VOICE_PROMPT_PROTOCOL", "QwenCustomVoicePromptEstimator"]
