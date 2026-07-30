from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .config import OPENROUTER_MAX_FALLBACK_MODELS, ChatConfig
from .types import ChatMessage, ImageGeneration, ImageOutput, ModelType

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


def _create_openai_client(config: ChatConfig, *, provider: str) -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` {provider} backend requires `openai`. {INSTALL_HINT}"
        ) from exc

    return OpenAI(api_key=config.api_key, base_url=config.base_url)


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
    if model_type == ModelType.GLM:
        return _request_glm(client, model, messages)
    if model_type == ModelType.DEEPSEEK:
        return _request_deepseek(client, model, messages)
    if model_type == ModelType.OPENROUTER:
        return _request_openrouter(client, models if models is not None else (model,), messages)
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _request_deepseek(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        reasoning_effort=DEEPSEEK_REASONING_EFFORT,
        extra_body={"thinking": {"type": "enabled"}},
    )
    return _response_text(response, provider="DeepSeek")


def _request_glm(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        thinking={"type": "enabled"},
        max_tokens=GLM_MAX_TOKENS,
        temperature=GLM_TEMPERATURE,
    )
    return _response_text(response, provider="GLM")


def _request_openrouter(
    client: object,
    models: Sequence[str],
    messages: Sequence[ChatMessage],
) -> str:
    if len(models) == 0:
        raise ValueError("OpenRouter models must not be empty.")
    completions = cast(Any, client).chat.completions
    fallbacks = list(models[1 : 1 + OPENROUTER_MAX_FALLBACK_MODELS])
    if len(fallbacks) == 0:
        response = completions.create(
            model=models[0],
            messages=messages,
            stream=False,
        )
    else:
        response = completions.create(
            model=models[0],
            messages=messages,
            stream=False,
            extra_body={"models": fallbacks},
        )
    used_model = _response_model(response)
    LOGGER.info(
        "openrouter using model=%s candidates=%s",
        used_model if used_model is not None else models[0],
        list(models),
    )
    return _response_text(response, provider="OpenRouter")


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


def _response_model(response: object) -> str | None:
    model = getattr(response, "model", None)
    if isinstance(model, str):
        return model
    if isinstance(response, Mapping):
        value = response.get("model")
        if isinstance(value, str):
            return value
    return None


def _response_text(response: object, *, provider: str) -> str:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError(f"{provider} response must contain choices[0].message.") from exc

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


def _provider_name(model_type: ModelType) -> str:
    if model_type == ModelType.DEEPSEEK:
        return "DeepSeek"
    if model_type == ModelType.OPENROUTER:
        return "OpenRouter"
    raise AssertionError(f"Unhandled openai-compatible model_type: {model_type!r}")
