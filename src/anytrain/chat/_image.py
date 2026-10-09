from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._constants import GLM_IMAGE_GENERATIONS_PATH, INSTALL_HINT
from .config import ChatConfig
from .types import ImageGeneration, ImageOutput, ModelType


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
    if model_type == ModelType.LLM_PROXY:
        raise NotImplementedError("LLM proxy image generation is not supported.")
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
    except ImportError as exc:  # pragma: no cover - exercised without the extra.
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
