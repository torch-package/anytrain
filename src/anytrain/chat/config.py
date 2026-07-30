from __future__ import annotations

import os
from typing import NamedTuple

from .types import ModelType

DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
GLM_BASE_URL_ENV = "GLM_BASE_URL"
GLM_MODEL_ENV = "GLM_MODEL"
GLM_API_KEY_ENV = "GLM_API_KEY"
OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
OPENROUTER_MODELS_ENV = "OPENROUTER_MODELS"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CANDIDATE_LIMIT_ENV = "OPENROUTER_CANDIDATE_LIMIT"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_CANDIDATE_LIMIT = 4
OPENROUTER_MAX_FALLBACK_MODELS = 3


class ChatConfig:
    __slots__ = ("allowlist", "api_key", "base_url", "candidate_limit", "model", "models")

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        models: tuple[str, ...] | None = None,
        allowlist: tuple[str, ...] | None = None,
        candidate_limit: int = OPENROUTER_DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.models = (model,) if models is None else models
        self.allowlist = allowlist
        self.candidate_limit = candidate_limit

    def __repr__(self) -> str:
        return (
            f"ChatConfig(base_url={self.base_url!r}, model={self.model!r}, "
            f"models={self.models!r}, allowlist={self.allowlist!r}, "
            f"candidate_limit={self.candidate_limit!r}, api_key=<hidden>)"
        )


class _EnvNames(NamedTuple):
    base_url: str
    model: str
    api_key: str


ENV_NAMES: dict[ModelType, _EnvNames] = {
    ModelType.DEEPSEEK: _EnvNames(
        base_url=DEEPSEEK_BASE_URL_ENV,
        model=DEEPSEEK_MODEL_ENV,
        api_key=DEEPSEEK_API_KEY_ENV,
    ),
    ModelType.GLM: _EnvNames(
        base_url=GLM_BASE_URL_ENV,
        model=GLM_MODEL_ENV,
        api_key=GLM_API_KEY_ENV,
    ),
}


def config_from_env(model_type: ModelType | str) -> ChatConfig:
    resolved = ModelType(model_type)
    if resolved == ModelType.OPENROUTER:
        return _openrouter_config_from_env()
    names = ENV_NAMES[resolved]
    base_url = _required_env(names.base_url)
    model = _required_env(names.model)
    api_key = _required_env(names.api_key)
    return ChatConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        models=(model,),
    )


def _openrouter_config_from_env() -> ChatConfig:
    return ChatConfig(
        base_url=_openrouter_base_url(),
        model="",
        api_key=_required_env(OPENROUTER_API_KEY_ENV),
        models=(),
        allowlist=_openrouter_allowlist(),
        candidate_limit=_openrouter_candidate_limit(),
    )


def _openrouter_base_url() -> str:
    value = os.environ.get(OPENROUTER_BASE_URL_ENV)
    if value is None:
        return OPENROUTER_DEFAULT_BASE_URL
    if value == "":
        raise ValueError(f"{OPENROUTER_BASE_URL_ENV} must not be empty.")
    return value


def _openrouter_allowlist() -> tuple[str, ...] | None:
    value = os.environ.get(OPENROUTER_MODELS_ENV)
    if value is None:
        return None
    if value == "":
        raise ValueError(f"{OPENROUTER_MODELS_ENV} must not be empty.")
    return _openrouter_models(value)


def _openrouter_candidate_limit() -> int:
    value = os.environ.get(OPENROUTER_CANDIDATE_LIMIT_ENV)
    if value is None:
        return OPENROUTER_DEFAULT_CANDIDATE_LIMIT
    if value == "":
        raise ValueError(f"{OPENROUTER_CANDIDATE_LIMIT_ENV} must not be empty.")
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{OPENROUTER_CANDIDATE_LIMIT_ENV} must be an integer >= 1."
        ) from exc
    if limit < 1:
        raise ValueError(f"{OPENROUTER_CANDIDATE_LIMIT_ENV} must be an integer >= 1.")
    return limit


def _openrouter_models(value: str) -> tuple[str, ...]:
    models: list[str] = []
    for part in value.split(","):
        model = part.strip()
        if model == "":
            raise ValueError(f"{OPENROUTER_MODELS_ENV} must not contain empty model names.")
        models.append(model)
    if len(models) == 0:
        raise ValueError(f"{OPENROUTER_MODELS_ENV} must contain at least one model.")
    return tuple(models)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"{name} must be set for anytrain.chat.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")
    return value
