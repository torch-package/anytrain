from __future__ import annotations

from ._constants import INSTALL_HINT
from .config import ChatConfig
from .types import ModelType


_OPENAI_COMPATIBLE = frozenset(
    {ModelType.DEEPSEEK, ModelType.OPENROUTER, ModelType.LLM_PROXY}
)


def create_client(model_type: ModelType, config: ChatConfig) -> object:
    if model_type == ModelType.GLM:
        return _create_glm_client(config)
    if model_type in _OPENAI_COMPATIBLE:
        return _create_openai_client(config, provider=_provider_name(model_type))
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def create_async_client(model_type: ModelType, config: ChatConfig) -> object:
    if model_type == ModelType.GLM:
        return _create_glm_client(config)
    if model_type in _OPENAI_COMPATIBLE:
        return _create_async_openai_client(config, provider=_provider_name(model_type))
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _create_openai_client(config: ChatConfig, *, provider: str) -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised without the extra.
        raise ImportError(
            f"`anytrain.chat` {provider} backend requires `openai`. {INSTALL_HINT}"
        ) from exc

    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _create_async_openai_client(config: ChatConfig, *, provider: str) -> object:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - exercised without the extra.
        raise ImportError(
            f"`anytrain.chat` {provider} async backend requires `openai`. {INSTALL_HINT}"
        ) from exc

    return AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


def _create_glm_client(config: ChatConfig) -> object:
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:  # pragma: no cover - exercised without the extra.
        raise ImportError(
            f"`anytrain.chat` GLM backend requires `zai-sdk`. {INSTALL_HINT}"
        ) from exc

    return ZhipuAiClient(api_key=config.api_key, base_url=config.base_url)


def _provider_name(model_type: ModelType) -> str:
    if model_type == ModelType.DEEPSEEK:
        return "DeepSeek"
    if model_type == ModelType.OPENROUTER:
        return "OpenRouter"
    if model_type == ModelType.LLM_PROXY:
        return "LLMProxy"
    raise AssertionError(f"Unhandled openai-compatible model_type: {model_type!r}")
