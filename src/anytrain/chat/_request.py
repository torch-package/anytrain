from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from ._constants import (
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_SYSTEM_PROMPT,
    GLM_MAX_TOKENS,
    GLM_TEMPERATURE,
)
from ._response import (
    chat_response,
    response_finish_reason,
    response_model,
    response_text,
)
from .config import OPENROUTER_MAX_FALLBACK_MODELS
from .types import ChatMessage, ChatResponse, ModelType

LOGGER = logging.getLogger("anytrain.chat")


def initial_messages(model_type: ModelType) -> list[ChatMessage]:
    if model_type == ModelType.DEEPSEEK:
        return [{"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT}]
    if model_type in {ModelType.GLM, ModelType.OPENROUTER}:
        return []
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def request(
    model_type: ModelType,
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
    *,
    models: Sequence[str] | None = None,
) -> str:
    return request_detail(
        model_type,
        client,
        model,
        messages,
        models=models,
    ).text


def request_detail(
    model_type: ModelType,
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
    *,
    models: Sequence[str] | None = None,
) -> ChatResponse:
    if model_type == ModelType.GLM:
        return _request_glm_detail(client, model, messages)
    if model_type == ModelType.DEEPSEEK:
        return _request_deepseek_detail(client, model, messages)
    if model_type == ModelType.OPENROUTER:
        return _request_openrouter_detail(
            client,
            models if models is not None else (model,),
            messages,
        )
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


async def async_request(
    model_type: ModelType,
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
    *,
    models: Sequence[str] | None = None,
) -> str:
    return (
        await async_request_detail(
            model_type,
            client,
            model,
            messages,
            models=models,
        )
    ).text


async def async_request_detail(
    model_type: ModelType,
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
    *,
    models: Sequence[str] | None = None,
) -> ChatResponse:
    if model_type == ModelType.GLM:
        return await asyncio.to_thread(
            request_detail,
            model_type,
            client,
            model,
            list(messages),
            models=models,
        )
    if model_type == ModelType.DEEPSEEK:
        return await _async_request_deepseek_detail(client, model, messages)
    if model_type == ModelType.OPENROUTER:
        return await _async_request_openrouter_detail(
            client,
            models if models is not None else (model,),
            messages,
        )
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _request_deepseek_detail(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> ChatResponse:
    text, response = _request_with_logging(
        "DeepSeek",
        model,
        messages,
        lambda: cast(Any, client).chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            reasoning_effort=DEEPSEEK_REASONING_EFFORT,
            extra_body={"thinking": {"type": "enabled"}},
        ),
    )
    return chat_response(text, response, fallback_model=model)


async def _async_request_deepseek_detail(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> ChatResponse:
    text, response = await _async_request_with_logging(
        "DeepSeek",
        model,
        messages,
        lambda: cast(Any, client).chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            reasoning_effort=DEEPSEEK_REASONING_EFFORT,
            extra_body={"thinking": {"type": "enabled"}},
        ),
    )
    return chat_response(text, response, fallback_model=model)


def _request_glm_detail(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> ChatResponse:
    text, response = _request_with_logging(
        "GLM",
        model,
        messages,
        lambda: cast(Any, client).chat.completions.create(
            model=model,
            messages=messages,
            thinking={"type": "enabled"},
            max_tokens=GLM_MAX_TOKENS,
            temperature=GLM_TEMPERATURE,
        ),
    )
    return chat_response(text, response, fallback_model=model)


def _request_openrouter_detail(
    client: object,
    models: Sequence[str],
    messages: Sequence[ChatMessage],
) -> ChatResponse:
    if len(models) == 0:
        raise ValueError("OpenRouter models must not be empty.")
    completions = cast(Any, client).chat.completions
    fallbacks = list(models[1 : 1 + OPENROUTER_MAX_FALLBACK_MODELS])
    if len(fallbacks) == 0:
        text, response = _request_with_logging(
            "OpenRouter",
            models[0],
            messages,
            lambda: completions.create(
                model=models[0],
                messages=messages,
                stream=False,
            ),
        )
    else:
        text, response = _request_with_logging(
            "OpenRouter",
            models[0],
            messages,
            lambda: completions.create(
                model=models[0],
                messages=messages,
                stream=False,
                extra_body={"models": fallbacks},
            ),
        )
    used_model = response_model(response)
    LOGGER.info(
        "openrouter using model=%s finish_reason=%s candidates=%s",
        used_model if used_model is not None else models[0],
        response_finish_reason(response),
        list(models),
    )
    return chat_response(text, response, fallback_model=models[0])


async def _async_request_openrouter_detail(
    client: object,
    models: Sequence[str],
    messages: Sequence[ChatMessage],
) -> ChatResponse:
    if len(models) == 0:
        raise ValueError("OpenRouter models must not be empty.")
    completions = cast(Any, client).chat.completions
    fallbacks = list(models[1 : 1 + OPENROUTER_MAX_FALLBACK_MODELS])
    if len(fallbacks) == 0:
        text, response = await _async_request_with_logging(
            "OpenRouter",
            models[0],
            messages,
            lambda: completions.create(
                model=models[0],
                messages=messages,
                stream=False,
            ),
        )
    else:
        text, response = await _async_request_with_logging(
            "OpenRouter",
            models[0],
            messages,
            lambda: completions.create(
                model=models[0],
                messages=messages,
                stream=False,
                extra_body={"models": fallbacks},
            ),
        )
    used_model = response_model(response)
    LOGGER.info(
        "openrouter using model=%s finish_reason=%s candidates=%s",
        used_model if used_model is not None else models[0],
        response_finish_reason(response),
        list(models),
    )
    return chat_response(text, response, fallback_model=models[0])


def _request_with_logging(
    provider: str,
    model: str,
    messages: Sequence[ChatMessage],
    create_response: Callable[[], object],
) -> tuple[str, object]:
    start = time.perf_counter()
    LOGGER.info(
        "chat request started provider=%s model=%s messages=%d started_at=%s",
        provider,
        model,
        len(messages),
        _utc_now(),
    )
    try:
        response = create_response()
        text = response_text(response, provider=provider)
    except Exception:
        LOGGER.warning(
            "chat request failed provider=%s model=%s elapsed_s=%.3f stopped_at=%s",
            provider,
            model,
            time.perf_counter() - start,
            _utc_now(),
            exc_info=True,
        )
        raise
    _log_request_stopped(provider, model, response=response, text=text, start=start)
    return text, response


async def _async_request_with_logging(
    provider: str,
    model: str,
    messages: Sequence[ChatMessage],
    create_response: Callable[[], Awaitable[object]],
) -> tuple[str, object]:
    start = time.perf_counter()
    LOGGER.info(
        "chat request started provider=%s model=%s messages=%d started_at=%s",
        provider,
        model,
        len(messages),
        _utc_now(),
    )
    try:
        response = await create_response()
        text = response_text(response, provider=provider)
    except asyncio.CancelledError:
        LOGGER.warning(
            "chat request cancelled provider=%s model=%s elapsed_s=%.3f stopped_at=%s",
            provider,
            model,
            time.perf_counter() - start,
            _utc_now(),
        )
        raise
    except Exception:
        LOGGER.warning(
            "chat request failed provider=%s model=%s elapsed_s=%.3f stopped_at=%s",
            provider,
            model,
            time.perf_counter() - start,
            _utc_now(),
            exc_info=True,
        )
        raise
    _log_request_stopped(provider, model, response=response, text=text, start=start)
    return text, response


def _log_request_stopped(
    provider: str,
    model: str,
    *,
    response: object,
    text: str,
    start: float,
) -> None:
    used_model = response_model(response)
    LOGGER.info(
        "chat request stopped provider=%s model=%s used_model=%s finish_reason=%s "
        "elapsed_s=%.3f stopped_at=%s response_chars=%d",
        provider,
        model,
        used_model if used_model is not None else model,
        response_finish_reason(response),
        time.perf_counter() - start,
        _utc_now(),
        len(text),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
