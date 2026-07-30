from __future__ import annotations

from collections.abc import Mapping
from enum import auto
from typing import Literal, NamedTuple, TypedDict

from anytrain._compat import StrEnum


class ModelType(StrEnum):
    DEEPSEEK = auto()
    GLM = auto()
    OPENROUTER = auto()


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ImageOutput(NamedTuple):
    url: str | None
    b64_json: str | None
    revised_prompt: str | None


class ImageGeneration(NamedTuple):
    model: str
    prompt: str
    size: str
    data: tuple[ImageOutput, ...]
    raw: Mapping[str, object]

    @property
    def url(self) -> str | None:
        return self.data[0].url

    @property
    def b64_json(self) -> str | None:
        return self.data[0].b64_json
