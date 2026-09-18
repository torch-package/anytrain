from __future__ import annotations

import asyncio
import base64
import inspect
import math
import mimetypes
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from numbers import Real
from os import PathLike
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Self, TypeVar

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
        self._async_client_loop: asyncio.AbstractEventLoop | None = None
        self._messages = initial_messages(self.model_type)
        self._lock = Lock()
        self._closed = False
        self._sync_client_closed = False
        self.last_response: ChatResponse | None = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Chat is closed.")

    def _close_sync_client(self) -> None:
        if not self._sync_client_closed:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._sync_client_closed = True

    def close(self) -> None:
        """Release a synchronous session; use aclose after async text requests."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._lock.acquire()
        else:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("Chat has an in-flight request; await chat.aclose().")
        try:
            if self._async_client is not None and self._async_client is not self._client:
                raise RuntimeError("Chat has an async client; await chat.aclose() on its event loop.")
            self._closed = True
            self._close_sync_client()
            self._async_client = None
        finally:
            self._lock.release()

    async def aclose(self) -> None:
        """Wait for in-flight requests, then release both provider clients."""
        async with self._async_text_lock():
            if self._async_client is not None:
                loop = asyncio.get_running_loop()
                if self._async_client_loop is not None and self._async_client_loop is not loop:
                    raise RuntimeError("Chat must be closed on the event loop that used its async client.")
            self._closed = True
            cleanup = asyncio.create_task(self._close_clients())
            cancelled = False
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    # Keep the session lock until SDK cleanup completes, even
                    # if the context's owning task is cancelled repeatedly.
                    if cleanup.cancelled():
                        raise
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

    async def _close_clients(self) -> None:
        try:
            if self._async_client is not None and self._async_client is not self._client:
                close = getattr(self._async_client, "close", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                self._async_client = None
        finally:
            await asyncio.to_thread(self._close_sync_client)
        self._async_client = None

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        self._require_open()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        await self.aclose()

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
            self._require_open()
            self._messages = initial_messages(self.model_type)

    def refresh_models(self) -> None:
        """Refresh OpenRouter live catalog selection without changing messages."""
        if self.model_type != ModelType.OPENROUTER:
            return
        with self._lock:
            self._require_open()
            self._config = resolve_models(self._config, refresh=True)

    async def async_refresh_models(self) -> None:
        await asyncio.to_thread(self.refresh_models)

    def __call__(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> str:
        prompt = _validate_prompt(prompt)
        _validate_refresh(refresh)
        with self._lock:
            return self._request_detail(prompt, images=images, refresh=refresh).text

    async def async_call(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> str:
        prompt = _validate_prompt(prompt)
        _validate_refresh(refresh)
        async with self._async_text_lock():
            self._ensure_async_text_loop()
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
        prompt = _validate_prompt(prompt)
        _validate_refresh(refresh)
        with self._lock:
            return self._request_detail(prompt, images=images, refresh=refresh)

    async def async_request(
        self,
        prompt: str,
        *,
        images: ImageInput | Iterable[ImageInput] | None = None,
        refresh: bool = False,
    ) -> ChatResponse:
        prompt = _validate_prompt(prompt)
        _validate_refresh(refresh)
        async with self._async_text_lock():
            self._ensure_async_text_loop()
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
        prompt = _validate_prompt(prompt)
        size = _non_empty_string(size, "size")
        model = _non_empty_string(model, "model")
        timeout = _validate_timeout(timeout)
        self._require_open()
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
        self._require_open()
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
        self._require_open()
        content = _content_for_prompt(self.model_type, prompt, images=images)
        messages = self._messages_for_prompt(content, refresh=refresh)
        request = async_request_detail(
            self.model_type, self._async_provider_client(), self._config.model,
            messages, models=self._config.models,
        )
        if self.model_type == ModelType.GLM:
            # Cancelling to_thread does not stop the SDK call. Hold the session
            # lock until it returns, so another request or close cannot race it.
            pending = asyncio.create_task(request)
            cancelled = False
            while True:
                try:
                    response = await asyncio.shield(pending)
                    break
                except asyncio.CancelledError:
                    if pending.cancelled():
                        raise
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
        else:
            response = await request
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
        self._ensure_async_text_loop()
        if self._async_client is None:
            if self.model_type == ModelType.GLM:
                self._async_client = self._client
            else:
                self._async_client = create_async_client(self.model_type, self._config)
        return self._async_client

    def _ensure_async_text_loop(self) -> None:
        self._require_open()
        loop = asyncio.get_running_loop()
        if self._async_client_loop is None:
            self._async_client_loop = loop
        elif self._async_client_loop is not loop:
            raise RuntimeError(
                "Chat async text requests must use the event loop that first used this "
                "instance. Create a separate Chat instance for another event loop."
            )

    @asynccontextmanager
    async def _async_text_lock(self) -> AsyncIterator[None]:
        # The synchronous API uses the same lock. Wait for it in a worker so
        # another text request never blocks this event loop while in flight.
        acquire = asyncio.create_task(asyncio.to_thread(self._lock.acquire))
        cancelled = False
        while True:
            try:
                await asyncio.shield(acquire)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            self._lock.release()
            raise asyncio.CancelledError
        try:
            yield
        finally:
            self._lock.release()


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
    try:
        values = list(images)
    except TypeError as exc:
        raise TypeError("images must be an image path/URL or an iterable of them.") from exc
    for index, value in enumerate(values):
        if not isinstance(value, (str, PathLike)):
            raise TypeError(f"images[{index}] must be a string or path-like object.")
    return values


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


def _validate_prompt(prompt: object) -> str:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")
    if not prompt.strip():
        raise ValueError("prompt must not be empty.")
    return prompt


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _validate_refresh(refresh: object) -> None:
    if not isinstance(refresh, bool):
        raise TypeError("refresh must be a bool.")


def _validate_timeout(timeout: object | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, Real):
        raise TypeError("timeout must be a real number or None.")
    resolved = float(timeout)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("timeout must be finite and positive when set.")
    return resolved
