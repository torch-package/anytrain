from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from .config import OPENROUTER_MAX_FALLBACK_MODELS, ChatConfig
from .types import ChatMessage, ChatResponse, ImageGeneration, ImageOutput, ModelType

DEEPSEEK_REASONING_EFFORT = "high"
DEEPSEEK_SYSTEM_PROMPT = "You are a helpful assistant"
GLM_MAX_TOKENS = 65536
GLM_TEMPERATURE = 1.0
GLM_IMAGE_MODEL = "glm-image"
GLM_IMAGE_SIZE = "1280x1280"
GLM_IMAGE_TIMEOUT = 120.0
GLM_IMAGE_GENERATIONS_PATH = "/images/generations"
INSTALL_HINT = (
    'Install chat dependencies with `python -m pip install "openai>=1.0" "requests>=2.31" "zai-sdk>=0.2.3"`.'
)
LOGGER = logging.getLogger("anytrain.chat")


def create_client(model_type: ModelType, config: ChatConfig) -> object:
    if model_type == ModelType.GLM:
        return _create_glm_client(config)
    if model_type in {ModelType.DEEPSEEK, ModelType.OPENROUTER}:
        return _create_openai_client(config, provider=_provider_name(model_type))
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def create_async_client(model_type: ModelType, config: ChatConfig) -> object:
    if model_type == ModelType.GLM:
        return _create_glm_client(config)
    if model_type in {ModelType.DEEPSEEK, ModelType.OPENROUTER}:
        return _create_async_openai_client(config, provider=_provider_name(model_type))
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _create_openai_client(config: ChatConfig, *, provider: str) -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` {provider} backend requires `openai`. {INSTALL_HINT}"
        ) from exc

    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _create_async_openai_client(config: ChatConfig, *, provider: str) -> object:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` {provider} async backend requires `openai`. {INSTALL_HINT}"
        ) from exc

    return AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


def _create_glm_client(config: ChatConfig) -> object:
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` GLM backend requires `zai-sdk`. {INSTALL_HINT}"
        ) from exc

    return ZhipuAiClient(api_key=config.api_key, base_url=config.base_url)


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


def _request_deepseek(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    return _request_deepseek_detail(client, model, messages).text


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
    return _chat_response(text, response, fallback_model=model)


async def _async_request_deepseek(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    return (await _async_request_deepseek_detail(client, model, messages)).text


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
    return _chat_response(text, response, fallback_model=model)


def _request_glm(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    return _request_glm_detail(client, model, messages).text


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
    return _chat_response(text, response, fallback_model=model)


def _request_openrouter(
    client: object,
    models: Sequence[str],
    messages: Sequence[ChatMessage],
) -> str:
    return _request_openrouter_detail(client, models, messages).text


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
    used_model = _response_model(response)
    LOGGER.info(
        "openrouter using model=%s finish_reason=%s candidates=%s",
        used_model if used_model is not None else models[0],
        _response_finish_reason(response),
        list(models),
    )
    return _chat_response(text, response, fallback_model=models[0])


async def _async_request_openrouter(
    client: object,
    models: Sequence[str],
    messages: Sequence[ChatMessage],
) -> str:
    return (await _async_request_openrouter_detail(client, models, messages)).text


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
    used_model = _response_model(response)
    LOGGER.info(
        "openrouter using model=%s finish_reason=%s candidates=%s",
        used_model if used_model is not None else models[0],
        _response_finish_reason(response),
        list(models),
    )
    return _chat_response(text, response, fallback_model=models[0])


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
        text = _response_text(response, provider=provider)
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
        text = _response_text(response, provider=provider)
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
    used_model = _response_model(response)
    LOGGER.info(
        "chat request stopped provider=%s model=%s used_model=%s finish_reason=%s "
        "elapsed_s=%.3f stopped_at=%s response_chars=%d",
        provider,
        model,
        used_model if used_model is not None else model,
        _response_finish_reason(response),
        time.perf_counter() - start,
        _utc_now(),
        len(text),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def generate_image(
    model_type: ModelType,
    config: ChatConfig,
    *,
    prompt: str,
    model: str,
    size: str,
    timeout: float | None,
) -> ImageGeneration:
    if model_type == ModelType.GLM:
        return _generate_glm_image(
            config,
            prompt=prompt,
            model=model,
            size=size,
            timeout=timeout,
        )
    if model_type == ModelType.DEEPSEEK:
        raise NotImplementedError("DeepSeek image generation is not supported.")
    if model_type == ModelType.OPENROUTER:
        raise NotImplementedError("OpenRouter image generation is not supported.")
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _generate_glm_image(
    config: ChatConfig,
    *,
    prompt: str,
    model: str,
    size: str,
    timeout: float | None,
) -> ImageGeneration:
    requests = _load_requests()
    response = requests.post(
        url=_glm_image_url(config.base_url),
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
        },
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("GLM image response must be a JSON object.")
    data = _image_outputs(payload)
    return ImageGeneration(
        model=model,
        prompt=prompt,
        size=size,
        data=data,
        raw=dict(payload),
    )


def _load_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` GLM image generation requires `requests`. {INSTALL_HINT}"
        ) from exc
    return requests


def _glm_image_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{GLM_IMAGE_GENERATIONS_PATH}"


def _image_outputs(payload: Mapping[str, object]) -> tuple[ImageOutput, ...]:
    value = payload.get("data")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("GLM image response must contain a data sequence.")
    if len(value) == 0:
        raise ValueError("GLM image response data must not be empty.")
    return tuple(_image_output(item) for item in value)


def _image_output(item: object) -> ImageOutput:
    if not isinstance(item, Mapping):
        raise TypeError("GLM image response data items must be JSON objects.")
    url = _optional_str(item, "url")
    b64_json = _optional_str(item, "b64_json")
    revised_prompt = _optional_str(item, "revised_prompt")
    if url is None and b64_json is None:
        raise TypeError("GLM image response data item must contain url or b64_json.")
    return ImageOutput(
        url=url,
        b64_json=b64_json,
        revised_prompt=revised_prompt,
    )


def _optional_str(payload: Mapping[object, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"GLM image response field {key!r} must be a string.")
    return value


def _chat_response(
    text: str,
    response: object,
    *,
    fallback_model: str | None = None,
) -> ChatResponse:
    model = _response_model(response)
    return ChatResponse(
        text=text,
        model=model if model is not None else fallback_model,
        usage=_response_usage(response),
        request_id=_response_request_id(response),
        finish_reason=_response_finish_reason(response),
        raw=response,
    )


def _response_model(response: object) -> str | None:
    model = getattr(response, "model", None)
    if isinstance(model, str):
        return model
    if isinstance(response, Mapping):
        value = response.get("model")
        if isinstance(value, str):
            return value
    return None


def _response_finish_reason(response: object) -> str | None:
    choice = _response_choice(response)
    if choice is None:
        return None
    finish_reason = getattr(choice, "finish_reason", None)
    if isinstance(finish_reason, str):
        return finish_reason
    if isinstance(choice, Mapping):
        value = choice.get("finish_reason")
        if isinstance(value, str):
            return value
    return None


def _response_text(response: object, *, provider: str) -> str:
    choice = _response_choice(response)
    if choice is None:
        raise TypeError(f"{provider} response must contain choices[0].message.")
    message = _choice_message(choice)

    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    raise TypeError(f"{provider} response message must be a string or contain string content.")


def _response_choice(response: object) -> object | None:
    choices = getattr(response, "choices", None)
    if isinstance(response, Mapping):
        choices = response.get("choices")
    if isinstance(choices, str) or not isinstance(choices, Sequence):
        return None
    try:
        return choices[0]
    except (IndexError, TypeError):
        return None


def _choice_message(choice: object) -> object:
    message = getattr(choice, "message", None)
    if isinstance(choice, Mapping):
        message = choice.get("message")
    return message


def _response_usage(response: object) -> Mapping[str, object] | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    return _usage_mapping(usage)


def _usage_mapping(usage: object) -> Mapping[str, object] | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return {str(key): value for key, value in usage.items()}

    for method_name in ("model_dump", "dict"):
        method = getattr(usage, method_name, None)
        if callable(method):
            try:
                payload = method()
            except TypeError:
                continue
            if isinstance(payload, Mapping):
                return {str(key): value for key, value in payload.items()}

    attrs: dict[str, object] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cost",
        "total_cost",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            attrs[key] = value
    if attrs:
        return attrs

    raw_attrs = getattr(usage, "__dict__", None)
    if isinstance(raw_attrs, Mapping):
        return {str(key): value for key, value in raw_attrs.items() if not str(key).startswith("_")}
    return None


def _response_request_id(response: object) -> str | None:
    for key in ("_request_id", "request_id", "id"):
        value = getattr(response, key, None)
        if isinstance(value, str):
            return value
    if isinstance(response, Mapping):
        for key in ("_request_id", "request_id", "id"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    return None


def _provider_name(model_type: ModelType) -> str:
    if model_type == ModelType.DEEPSEEK:
        return "DeepSeek"
    if model_type == ModelType.OPENROUTER:
        return "OpenRouter"
    raise AssertionError(f"Unhandled openai-compatible model_type: {model_type!r}")
