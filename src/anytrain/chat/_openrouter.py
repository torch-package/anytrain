from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

from .config import ChatConfig

LOGGER = logging.getLogger("anytrain.chat")
MODELS_PATH = "/models"
MODELS_TIMEOUT = 60.0
OPENROUTER_FREE_ROUTER_ID = "openrouter/free"
INSTALL_HINT = (
    'Install chat dependencies with `python -m pip install "openai>=1.0" "requests>=2.31" "zai-sdk>=0.2.3"`.'
)
_CATALOG_CACHE: dict[tuple[str, str], tuple[float, tuple[dict[str, Any], ...]]] = {}
_CATALOG_CACHE_LOCK = Lock()
_CATALOG_KEY_LOCKS: dict[tuple[str, str], Lock] = {}


def resolve_models(config: ChatConfig, *, refresh: bool = False) -> ChatConfig:
    if config.mode == "static":
        return _resolve_static(config)
    catalog = _catalog(config, refresh=refresh)
    ranked = _rank(catalog, allowlist=config.allowlist)
    if len(ranked) == 0:
        raise ValueError("OpenRouter model selection matched no chat models.")
    limit = config.candidate_limit
    if limit < 1:
        raise ValueError("OpenRouter candidate_limit must be >= 1.")
    models = tuple(item["id"] for item in ranked[:limit])
    LOGGER.info(
        "openrouter selected candidates=%s free=%s",
        list(models),
        [item["id"] for item in ranked[:limit] if item["free"]],
    )
    return ChatConfig(
        base_url=config.base_url,
        model=models[0],
        api_key=config.api_key,
        models=models,
        allowlist=config.allowlist,
        candidate_limit=config.candidate_limit,
        mode=config.mode,
        catalog_ttl=config.catalog_ttl,
    )


def _resolve_static(config: ChatConfig) -> ChatConfig:
    if config.allowlist is None:
        raise ValueError("OpenRouter static mode requires a non-empty model list.")
    models = config.allowlist[: config.candidate_limit]
    if not models:
        raise ValueError("OpenRouter static mode requires at least one model.")
    LOGGER.info("openrouter selected static candidates=%s", list(models))
    return ChatConfig(
        base_url=config.base_url,
        model=models[0],
        api_key=config.api_key,
        models=models,
        allowlist=config.allowlist,
        candidate_limit=config.candidate_limit,
        mode=config.mode,
        catalog_ttl=config.catalog_ttl,
    )


def _catalog(config: ChatConfig, *, refresh: bool) -> list[dict[str, Any]]:
    key = (config.base_url.rstrip("/"), _api_key_digest(config.api_key))
    with _CATALOG_CACHE_LOCK:
        key_lock = _CATALOG_KEY_LOCKS.setdefault(key, Lock())
    with key_lock:
        now = time.monotonic()
        with _CATALOG_CACHE_LOCK:
            cached = _CATALOG_CACHE.get(key)
        if not refresh and cached is not None and now - cached[0] < config.catalog_ttl:
            return [dict(item) for item in cached[1]]

        # The per-key lock makes concurrent construction single-flight while
        # allowing unrelated OpenRouter endpoints to fetch independently.
        catalog = _fetch_catalog(config)
        with _CATALOG_CACHE_LOCK:
            _CATALOG_CACHE[key] = (now, tuple(dict(item) for item in catalog))
        return catalog


def _fetch_catalog(config: ChatConfig) -> list[dict[str, Any]]:
    requests = _load_requests()
    response = requests.get(
        url=f"{config.base_url.rstrip('/')}{MODELS_PATH}",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        timeout=MODELS_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("OpenRouter models response must be a JSON object.")
    data = payload.get("data")
    if isinstance(data, str) or not isinstance(data, Sequence):
        raise TypeError("OpenRouter models response must contain a data sequence.")
    models: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise TypeError("OpenRouter models response data items must be JSON objects.")
        models.append(dict(item))
    return models


def _api_key_digest(api_key: str) -> str:
    import hashlib

    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def clear_model_catalog_cache() -> None:
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE.clear()


def _rank(
    catalog: Sequence[Mapping[str, Any]],
    *,
    allowlist: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    allowed = None if allowlist is None else set(allowlist)
    if allowlist is not None:
        available = {str(item.get("id")) for item in catalog if item.get("id") is not None}
        missing = [model for model in allowlist if model not in available]
        if missing:
            raise ValueError(
                "OpenRouter allowlist models were not found in the live catalog: "
                + ", ".join(missing)
            )
    selected: list[dict[str, Any]] = []
    for item in catalog:
        model_id = item.get("id")
        if not isinstance(model_id, str) or model_id == "":
            continue
        if allowed is not None and model_id not in allowed:
            continue
        if not _is_text_chat_model(item):
            continue
        pricing = _pricing(item)
        if pricing is None:
            continue
        prompt, completion, discount = pricing
        if prompt < 0 or completion < 0:
            continue
        selected.append(
            {
                "id": model_id,
                "prompt": prompt,
                "completion": completion,
                "discount": discount,
                "free": _is_free(prompt, completion, discount, model_id),
            }
        )
    selected.sort(key=_rank_key)
    return selected


def _rank_key(item: Mapping[str, Any]) -> tuple[object, ...]:
    free = bool(item["free"])
    discount = float(item["discount"])
    if free or discount >= 1.0:
        tier = 0
    elif discount > 0.0:
        tier = 1
    else:
        tier = 2
    prefer_router = 0 if item["id"] == OPENROUTER_FREE_ROUTER_ID else 1
    cost = float(item["prompt"]) + float(item["completion"])
    return (tier, prefer_router, cost, str(item["id"]))


def _is_free(prompt: float, completion: float, discount: float, model_id: str) -> bool:
    return (
        (prompt == 0.0 and completion == 0.0)
        or discount >= 1.0
        or model_id.endswith(":free")
    )


def _is_text_chat_model(item: Mapping[str, Any]) -> bool:
    architecture = item.get("architecture")
    if not isinstance(architecture, Mapping):
        return False
    inputs = architecture.get("input_modalities")
    outputs = architecture.get("output_modalities")
    if isinstance(inputs, str) or not isinstance(inputs, Sequence):
        return False
    if isinstance(outputs, str) or not isinstance(outputs, Sequence):
        return False
    if "text" not in inputs or "text" not in outputs:
        return False
    if "audio" in outputs:
        return False
    parameters = item.get("supported_parameters")
    if isinstance(parameters, str) or not isinstance(parameters, Sequence):
        return False
    return "temperature" in parameters


def _pricing(item: Mapping[str, Any]) -> tuple[float, float, float] | None:
    pricing = item.get("pricing")
    if not isinstance(pricing, Mapping):
        return None
    try:
        prompt = float(pricing.get("prompt", "nan"))
        completion = float(pricing.get("completion", "nan"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(prompt) or not math.isfinite(completion):
        return None
    discount_raw = pricing.get("discount", 0.0)
    try:
        discount = float(discount_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(discount):
        return None
    return prompt, completion, discount


def _load_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` OpenRouter model selection requires `requests`. {INSTALL_HINT}"
        ) from exc
    return requests


__all__ = ["clear_model_catalog_cache", "resolve_models"]
