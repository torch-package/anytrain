from __future__ import annotations

from ._openrouter import resolve_models
from ._provider import (
    GLM_IMAGE_MODEL,
    GLM_IMAGE_SIZE,
    GLM_IMAGE_TIMEOUT,
    create_client,
    generate_image,
    initial_messages,
    request,
)
from .config import config_from_env
from .types import ChatMessage, ImageGeneration, ModelType


class Chat:
    def __init__(self, model_type: ModelType | str = ModelType.DEEPSEEK):
        self.model_type = ModelType(model_type)
        config = config_from_env(self.model_type)
        if self.model_type == ModelType.OPENROUTER:
            config = resolve_models(config)
        self._config = config
        self._client = create_client(self.model_type, self._config)
        self._messages = initial_messages(self.model_type)

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
        self._messages = initial_messages(self.model_type)

    def __call__(self, prompt: str, *, refresh: bool = False) -> str:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        if refresh:
            self.refresh()
        messages: list[ChatMessage] = [
            *self._messages,
            {"role": "user", "content": prompt},
        ]
        response = request(
            self.model_type,
            self._client,
            self._config.model,
            messages,
            models=self._config.models,
        )
        self._messages = [*messages, {"role": "assistant", "content": response}]
        return response

    def image(
        self,
        prompt: str,
        *,
        size: str = GLM_IMAGE_SIZE,
        model: str = GLM_IMAGE_MODEL,
        timeout: float | None = GLM_IMAGE_TIMEOUT,
    ) -> ImageGeneration:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        if size == "":
            raise ValueError("size must not be empty.")
        if model == "":
            raise ValueError("model must not be empty.")
        return generate_image(
            self.model_type,
            self._config,
            prompt=prompt,
            model=model,
            size=size,
            timeout=timeout,
        )
