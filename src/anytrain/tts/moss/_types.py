from __future__ import annotations

from typing import Any, Protocol


class AttentionModel(Protocol):
    def _set_attention_implementation(self, value: str) -> None: ...


class DeviceMovable(Protocol):
    def to(self, device: object) -> DeviceMovable: ...


class AudioTokenizerProcessor(Protocol):
    audio_tokenizer: DeviceMovable


class Processor(Protocol):
    model_config: Any

    def __call__(self, value: object, **kwargs: object) -> object: ...

    def build_user_message(self, **kwargs: object) -> object: ...

    def decode(self, value: object) -> object: ...
