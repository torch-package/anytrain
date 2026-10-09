from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True)
class _CompletionSpec:
    provider: str
    model: str
    options: tuple[tuple[str, object], ...]
    candidates: tuple[str, ...] = ()

    def kwargs(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": messages,
            **dict(self.options),
        }


def initial_messages(model_type: ModelType) -> list[ChatMessage]:
    if model_type == ModelType.DEEPSEEK:
        return [{"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT}]
    if model_type in {ModelType.GLM, ModelType.OPENROUTER, ModelType.LLM_PROXY}:
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
    spec = _completion_spec(
        model_type,
        model,
        models=models,
    )
    text, response = _request_completion(client, spec, messages)
    return _chat_response(text, response, spec)


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
    spec = _completion_spec(
        model_type,
        model,
        models=models,
    )
    text, response = await _async_request_completion(client, spec, messages)
    return _chat_response(text, response, spec)


def _completion_spec(
    model_type: ModelType,
    model: str,
    *,
    models: Sequence[str] | None,
) -> _CompletionSpec:
    if model_type == ModelType.DEEPSEEK:
        return _CompletionSpec(
            provider="DeepSeek",
            model=model,
            options=(
                ("stream", False),
                ("reasoning_effort", DEEPSEEK_REASONING_EFFORT),
                ("extra_body", {"thinking": {"type": "enabled"}}),
            ),
        )
    if model_type == ModelType.GLM:
        return _CompletionSpec(
            provider="GLM",
            model=model,
            options=(
                ("thinking", {"type": "enabled"}),
                ("max_tokens", GLM_MAX_TOKENS),
                ("temperature", GLM_TEMPERATURE),
            ),
        )
    if model_type == ModelType.OPENROUTER:
        return _openrouter_completion_spec(
            models if models is not None else (model,),
        )
    if model_type == ModelType.LLM_PROXY:
        return _CompletionSpec(
            provider="LLMProxy",
            model=model,
            options=(("stream", False),),
        )
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _openrouter_completion_spec(models: Sequence[str]) -> _CompletionSpec:
    if len(models) == 0:
        raise ValueError("OpenRouter models must not be empty.")
    fallbacks = list(models[1 : 1 + OPENROUTER_MAX_FALLBACK_MODELS])
    options: tuple[tuple[str, object], ...] = (("stream", False),)
    if len(fallbacks) > 0:
        options = (*options, ("extra_body", {"models": fallbacks}))
    return _CompletionSpec(
        provider="OpenRouter",
        model=models[0],
        options=options,
        candidates=tuple(models),
    )


def _request_completion(
    client: object,
    spec: _CompletionSpec,
    messages: Sequence[ChatMessage],
) -> tuple[str, object]:
    return _request_with_logging(
        spec.provider,
        spec.model,
        messages,
        lambda: cast(Any, client).chat.completions.create(
            **spec.kwargs(messages),
        ),
    )


async def _async_request_completion(
    client: object,
    spec: _CompletionSpec,
    messages: Sequence[ChatMessage],
) -> tuple[str, object]:
    return await _async_request_with_logging(
        spec.provider,
        spec.model,
        messages,
        lambda: cast(Any, client).chat.completions.create(
            **spec.kwargs(messages),
        ),
    )


def _log_openrouter_response(response: object, spec: _CompletionSpec) -> None:
    if len(spec.candidates) == 0:
        return
    used_model = response_model(response)
    LOGGER.info(
        "openrouter using model=%s finish_reason=%s candidates=%s",
        used_model if used_model is not None else spec.model,
        response_finish_reason(response),
        list(spec.candidates),
    )


def _chat_response(text: str, response: object, spec: _CompletionSpec) -> ChatResponse:
    _log_openrouter_response(response, spec)
    return chat_response(text, response, fallback_model=spec.model)


def _request_with_logging(
    provider: str,
    model: str,
    messages: Sequence[ChatMessage],
    create_response: Callable[[], object],
) -> tuple[str, object]:
    start = _log_request_started(provider, model, messages)
    try:
        response = create_response()
        text = response_text(response, provider=provider)
    except Exception:
        _log_request_failed(provider, model, start)
        raise
    _log_request_stopped(provider, model, response=response, text=text, start=start)
    return text, response


async def _async_request_with_logging(
    provider: str,
    model: str,
    messages: Sequence[ChatMessage],
    create_response: Callable[[], Awaitable[object]],
) -> tuple[str, object]:
    start = _log_request_started(provider, model, messages)
    try:
        response = await create_response()
        text = response_text(response, provider=provider)
    except asyncio.CancelledError:
        _log_request_cancelled(provider, model, start)
        raise
    except Exception:
        _log_request_failed(provider, model, start)
        raise
    _log_request_stopped(provider, model, response=response, text=text, start=start)
    return text, response


def _log_request_started(
    provider: str,
    model: str,
    messages: Sequence[ChatMessage],
) -> float:
    start = time.perf_counter()
    LOGGER.info(
        "chat request started provider=%s model=%s messages=%d started_at=%s",
        provider,
        model,
        len(messages),
        _utc_now(),
    )
    return start


def _log_request_failed(provider: str, model: str, start: float) -> None:
    LOGGER.warning(
        "chat request failed provider=%s model=%s elapsed_s=%.3f stopped_at=%s",
        provider,
        model,
        time.perf_counter() - start,
        _utc_now(),
        exc_info=True,
    )


def _log_request_cancelled(provider: str, model: str, start: float) -> None:
    LOGGER.warning(
        "chat request cancelled provider=%s model=%s elapsed_s=%.3f stopped_at=%s",
        provider,
        model,
        time.perf_counter() - start,
        _utc_now(),
    )


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
    return datetime.now(UTC).isoformat(timespec="milliseconds")
