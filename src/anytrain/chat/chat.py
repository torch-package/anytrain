from __future__ import annotations

import asyncio
import base64
import mimetypes
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from threading import RLock
from typing import TypeVar

from ._openrouter import resolve_models
from ._provider import (
    GLM_IMAGE_MODEL,
    GLM_IMAGE_SIZE,
    GLM_IMAGE_TIMEOUT,
    async_request_detail,
    create_async_client,
    create_client,
    generate_image,
    initial_messages,
    request_detail,
)
from .config import config_from_env
from .types import (
    ChatContent,
    ChatContentPart,
    ChatMessage,
    ChatResponse,
    ImageGeneration,
    ModelType,
)

_ChatT = TypeVar("_ChatT", bound="Chat")
ImageInput = str | PathLike[str]


class Chat:
    def __init__(self, model_type: ModelType | str = ModelType.DEEPSEEK):
        self.model_type = ModelType(model_type)
        config = config_from_env(self.model_type)
        if self.model_type == ModelType.OPENROUTER:
            config = resolve_models(config)
        self._config = config
        self._client = create_client(self.model_type, self._config)
        self._async_client: object | None = None
        self._async_lock: asyncio.Lock | None = None
        self._async_lock_loop: asyncio.AbstractEventLoop | None = None
        self._messages = initial_messages(self.model_type)
        self._lock = RLock()
        self.last_response: ChatResponse | None = None

    @classmethod
    async def async_create(
        cls: type[_ChatT],
        model_type: ModelType | str = ModelType.DEEPSEEK,
    ) -> _ChatT:
        return await asyncio.to_thread(cls, model_type)

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def models(self) -> tuple[str, ...]:
        return self._config.models

    def refresh(self) -> None:
        with self._lock:
            self._messages = initial_messages(self.model_type)

    def __call__(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> str:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        with self._lock:
            return self._request_detail(prompt, images=images, refresh=refresh).text

    async def async_call(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> str:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        async with self._async_text_lock():
            return (
                await self._request_detail_async(prompt, images=images, refresh=refresh)
            ).text

    async def acall(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> str:
        return await self.async_call(prompt, images=images, refresh=refresh)

    def request(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> ChatResponse:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        with self._lock:
            return self._request_detail(prompt, images=images, refresh=refresh)

    async def async_request(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> ChatResponse:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        async with self._async_text_lock():
            return await self._request_detail_async(prompt, images=images, refresh=refresh)

    @property
    def last_usage(self) -> object:
        return None if self.last_response is None else self.last_response.usage

    @property
    def last_model(self) -> str | None:
        return None if self.last_response is None else self.last_response.model

    @property
    def last_request_id(self) -> str | None:
        return None if self.last_response is None else self.last_response.request_id

    @property
    def last_finish_reason(self) -> str | None:
        return None if self.last_response is None else self.last_response.finish_reason

    def image(
        self,
        prompt: str,
        *,
        size: str = GLM_IMAGE_SIZE,
        model: str = GLM_IMAGE_MODEL,
        timeout: float | None = GLM_IMAGE_TIMEOUT,
    ) -> ImageGeneration:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        if size == "":
            raise ValueError("size must not be empty.")
        if model == "":
            raise ValueError("model must not be empty.")
        return generate_image(
            self.model_type,
            self._config,
            prompt=prompt,
            model=model,
            size=size,
            timeout=timeout,
        )

    async def async_image(
        self,
        prompt: str,
        *,
        size: str = GLM_IMAGE_SIZE,
        model: str = GLM_IMAGE_MODEL,
        timeout: float | None = GLM_IMAGE_TIMEOUT,
    ) -> ImageGeneration:
        return await asyncio.to_thread(
            self.image,
            prompt,
            size=size,
            model=model,
            timeout=timeout,
        )

    def _request_text(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None,
        refresh: bool,
    ) -> str:
        return self._request_detail(prompt, images=images, refresh=refresh).text

    def _request_detail(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None,
        refresh: bool,
    ) -> ChatResponse:
        content = _content_for_prompt(self.model_type, prompt, images=images)
        messages = self._messages_for_prompt(content, refresh=refresh)
        response = request_detail(
            self.model_type,
            self._client,
            self._config.model,
            messages,
            models=self._config.models,
        )
        self._record_response(messages, response)
        return response

    async def _request_text_async(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None,
        refresh: bool,
    ) -> str:
        return (await self._request_detail_async(prompt, images=images, refresh=refresh)).text

    async def _request_detail_async(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None,
        refresh: bool,
    ) -> ChatResponse:
        content = _content_for_prompt(self.model_type, prompt, images=images)
        messages = self._messages_for_prompt(content, refresh=refresh)
        response = await async_request_detail(
            self.model_type,
            self._async_provider_client(),
            self._config.model,
            messages,
            models=self._config.models,
        )
        self._record_response(messages, response)
        return response

    def _messages_for_prompt(self, content: ChatContent, *, refresh: bool) -> list[ChatMessage]:
        if refresh:
            self._messages = initial_messages(self.model_type)
        return [
            *self._messages,
            {"role": "user", "content": content},
        ]

    def _record_response(self, messages: list[ChatMessage], response: ChatResponse) -> None:
        self.last_response = response
        self._messages = [*messages, {"role": "assistant", "content": response.text}]

    def _async_provider_client(self) -> object:
        if self._async_client is None:
            if self.model_type == ModelType.GLM:
                self._async_client = self._client
            else:
                self._async_client = create_async_client(self.model_type, self._config)
        return self._async_client

    def _async_text_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock


def _content_for_prompt(
    model_type: ModelType,
    prompt: str,
    *,
    images: ImageInput | Iterable[ImageInput] | None,
) -> ChatContent:
    if images is None:
        return prompt
    image_inputs = _normalize_images(images)
    if len(image_inputs) == 0:
        raise ValueError("images must not be empty.")
    if model_type == ModelType.DEEPSEEK:
        raise NotImplementedError("DeepSeek image inputs are not supported.")

    parts: list[ChatContentPart] = [{"type": "text", "text": prompt}]
    parts.extend(
        {"type": "image_url", "image_url": {"url": _image_url(image)}}
        for image in image_inputs
    )
    return parts


def _normalize_images(images: ImageInput | Iterable[ImageInput]) -> list[ImageInput]:
    if isinstance(images, (str, PathLike)):
        return [images]
    return list(images)


def _image_url(image: ImageInput) -> str:
    value = str(image)
    if value == "":
        raise ValueError("images must not contain empty values.")
    lower = value.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return value

    path = Path(image)
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
