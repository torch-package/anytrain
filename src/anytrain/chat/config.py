from __future__ import annotations

import math
import os
from typing import NamedTuple
from urllib.parse import urlparse

from .types import ModelType

DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
GLM_BASE_URL_ENV = "GLM_BASE_URL"
GLM_MODEL_ENV = "GLM_MODEL"
GLM_API_KEY_ENV = "GLM_API_KEY"
OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV = "OPENROUTER_ALLOW_CUSTOM_BASE_URL"
OPENROUTER_MODELS_ENV = "OPENROUTER_MODELS"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CANDIDATE_LIMIT_ENV = "OPENROUTER_CANDIDATE_LIMIT"
OPENROUTER_MODE_ENV = "OPENROUTER_MODE"
OPENROUTER_CATALOG_TTL_ENV = "OPENROUTER_CATALOG_TTL"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_CANDIDATE_LIMIT = 4
OPENROUTER_DEFAULT_CATALOG_TTL = 300.0
OPENROUTER_MAX_FALLBACK_MODELS = 3


class ChatConfig:
    __slots__ = (
        "allowlist",
        "api_key",
        "base_url",
        "candidate_limit",
        "catalog_ttl",
        "mode",
        "model",
        "models",
    )

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        models: tuple[str, ...] | None = None,
        allowlist: tuple[str, ...] | None = None,
        candidate_limit: int = OPENROUTER_DEFAULT_CANDIDATE_LIMIT,
        mode: str = "live",
        catalog_ttl: float = OPENROUTER_DEFAULT_CATALOG_TTL,
    ) -> None:
        self.base_url = _non_empty_string(base_url, "base_url")
        self.model = _string(model, "model")
        self.api_key = _non_empty_string(api_key, "api_key")
        self.mode = _mode(mode)
        self.catalog_ttl = _catalog_ttl(catalog_ttl)
        resolved_models = (model,) if models is None else models
        self.models = _string_tuple(resolved_models, "models", allow_empty=True)
        if self.model:
            if not self.models:
                raise ValueError("models must contain model when model is set.")
            if self.models[0] != self.model:
                raise ValueError("model must be the first entry in models.")
        elif self.models:
            raise ValueError("model must be set when models is non-empty.")
        self.allowlist = (
            None
            if allowlist is None
            else _string_tuple(allowlist, "allowlist", allow_empty=False)
        )
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("candidate_limit must be an integer.")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be >= 1.")
        self.candidate_limit = candidate_limit

    def __repr__(self) -> str:
        return (
            f"ChatConfig(base_url={self.base_url!r}, model={self.model!r}, "
            f"models={self.models!r}, allowlist={self.allowlist!r}, "
            f"candidate_limit={self.candidate_limit!r}, mode={self.mode!r}, "
            f"catalog_ttl={self.catalog_ttl!r}, api_key=<hidden>)"
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
    mode = _openrouter_mode()
    allowlist = _openrouter_allowlist()
    if mode == "static" and allowlist is None:
        raise ValueError(
            f"{OPENROUTER_MODE_ENV}=static requires {OPENROUTER_MODELS_ENV}."
        )
    return ChatConfig(
        base_url=_openrouter_base_url(),
        model="",
        api_key=_required_env(OPENROUTER_API_KEY_ENV),
        models=(),
        allowlist=allowlist,
        candidate_limit=_openrouter_candidate_limit(),
        mode=mode,
        catalog_ttl=_openrouter_catalog_ttl(),
    )


def _openrouter_base_url() -> str:
    value = os.environ.get(OPENROUTER_BASE_URL_ENV)
    if value is None:
        return OPENROUTER_DEFAULT_BASE_URL
    if value == "":
        raise ValueError(f"{OPENROUTER_BASE_URL_ENV} must not be empty.")
    _validate_openrouter_base_url(value)
    return value


def _validate_openrouter_base_url(value: str) -> None:
    if value.rstrip("/") == OPENROUTER_DEFAULT_BASE_URL.rstrip("/"):
        return
    parsed = urlparse(value)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"{OPENROUTER_BASE_URL_ENV} custom URLs must use https.")
    if not parsed.netloc:
        raise ValueError(f"{OPENROUTER_BASE_URL_ENV} custom URLs must include a host.")
    if not _truthy_env(OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV):
        raise ValueError(
            f"{OPENROUTER_BASE_URL_ENV} points to a custom host. Set "
            f"{OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV}=1 to explicitly allow sending "
            "the OpenRouter API key to that endpoint."
        )


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    if value in {"1", "true", "True", "yes", "YES", "on", "ON"}:
        return True
    if value in {"0", "false", "False", "no", "NO", "off", "OFF", ""}:
        return False
    raise ValueError(f"{name} must be a boolean flag.")


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


def _openrouter_mode() -> str:
    value = os.environ.get(OPENROUTER_MODE_ENV, "live")
    if value not in {"live", "static"}:
        raise ValueError(f"{OPENROUTER_MODE_ENV} must be 'live' or 'static'.")
    return value


def _openrouter_catalog_ttl() -> float:
    value = os.environ.get(OPENROUTER_CATALOG_TTL_ENV)
    if value is None:
        return OPENROUTER_DEFAULT_CATALOG_TTL
    if value == "":
        raise ValueError(f"{OPENROUTER_CATALOG_TTL_ENV} must not be empty.")
    try:
        ttl = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{OPENROUTER_CATALOG_TTL_ENV} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(ttl) or ttl < 0.0:
        raise ValueError(
            f"{OPENROUTER_CATALOG_TTL_ENV} must be a finite non-negative number."
        )
    return ttl


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


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _non_empty_string(value: object, name: str) -> str:
    resolved = _string(value, name)
    if not resolved:
        raise ValueError(f"{name} must not be empty.")
    return resolved


def _string_tuple(
    value: object,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of strings.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    resolved = tuple(
        _non_empty_string(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must contain unique values.")
    return resolved


def _mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("mode must be a string.")
    if value not in {"live", "static"}:
        raise ValueError("mode must be 'live' or 'static'.")
    return value


def _catalog_ttl(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("catalog_ttl must be a finite non-negative number.")
    ttl = float(value)
    if not math.isfinite(ttl) or ttl < 0.0:
        raise ValueError("catalog_ttl must be a finite non-negative number.")
    return ttl
