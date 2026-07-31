from __future__ import annotations

import asyncio
from threading import RLock
from typing import TypeVar

from ._openrouter import resolve_models
from ._provider import (
    GLM_IMAGE_MODEL,
    GLM_IMAGE_SIZE,
    GLM_IMAGE_TIMEOUT,
    async_request,
    create_async_client,
    create_client,
    generate_image,
    initial_messages,
    request,
)
from .config import config_from_env
from .types import ChatMessage, ImageGeneration, ModelType

_ChatT = TypeVar("_ChatT", bound="Chat")


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

    def __call__(self, prompt: str, *, refresh: bool = False) -> str:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        with self._lock:
            return self._request_text(prompt, refresh=refresh)

    async def async_call(self, prompt: str, *, refresh: bool = False) -> str:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        async with self._async_text_lock():
            return await self._request_text_async(prompt, refresh=refresh)

    async def acall(self, prompt: str, *, refresh: bool = False) -> str:
        return await self.async_call(prompt, refresh=refresh)

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

    def _request_text(self, prompt: str, *, refresh: bool) -> str:
        messages = self._messages_for_prompt(prompt, refresh=refresh)
        response = request(
            self.model_type,
            self._client,
            self._config.model,
            messages,
            models=self._config.models,
        )
        self._record_response(messages, response)
        return response

    async def _request_text_async(self, prompt: str, *, refresh: bool) -> str:
        messages = self._messages_for_prompt(prompt, refresh=refresh)
        response = await async_request(
            self.model_type,
            self._async_provider_client(),
            self._config.model,
            messages,
            models=self._config.models,
        )
        self._record_response(messages, response)
        return response

    def _messages_for_prompt(self, prompt: str, *, refresh: bool) -> list[ChatMessage]:
        if refresh:
            self._messages = initial_messages(self.model_type)
        return [
            *self._messages,
            {"role": "user", "content": prompt},
        ]

    def _record_response(self, messages: list[ChatMessage], response: str) -> None:
        self._messages = [*messages, {"role": "assistant", "content": response}]

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
