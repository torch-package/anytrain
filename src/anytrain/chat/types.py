from __future__ import annotations

from collections.abc import Mapping
from enum import auto
from typing import Literal, NamedTuple, TypedDict

from anytrain._compat import StrEnum


class ModelType(StrEnum):
    DEEPSEEK = auto()
    GLM = auto()
    OPENROUTER = auto()
    LLM_PROXY = auto()


class ChatTextPart(TypedDict):
    type: Literal["text"]
    text: str


class ChatImageUrl(TypedDict):
    url: str


class ChatImageUrlPart(TypedDict):
    type: Literal["image_url"]
    image_url: ChatImageUrl


ChatContentPart = ChatTextPart | ChatImageUrlPart
ChatContent = str | list[ChatContentPart]


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: ChatContent


class ChatResponse(NamedTuple):
    text: str
    model: str | None
    usage: Mapping[str, object] | None
    request_id: str | None
    finish_reason: str | None
    raw: object


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
