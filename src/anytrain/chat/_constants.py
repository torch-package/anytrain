from __future__ import annotations

import logging

DEEPSEEK_REASONING_EFFORT = "high"
DEEPSEEK_SYSTEM_PROMPT = "You are a helpful assistant"
GLM_MAX_TOKENS = 65536
GLM_TEMPERATURE = 1.0
GLM_IMAGE_MODEL = "glm-image"
GLM_IMAGE_SIZE = "1280x1280"
GLM_IMAGE_TIMEOUT = 120.0
GLM_IMAGE_GENERATIONS_PATH = "/images/generations"
INSTALL_HINT = (
    'Install chat dependencies with `python -m pip install "openai>=1.0" '
    '"requests>=2.31" "zai-sdk>=0.2.3"`.'
)
LOGGER = logging.getLogger("anytrain.chat")

__all__ = [
    "DEEPSEEK_REASONING_EFFORT",
    "DEEPSEEK_SYSTEM_PROMPT",
    "GLM_IMAGE_GENERATIONS_PATH",
    "GLM_IMAGE_MODEL",
    "GLM_IMAGE_SIZE",
    "GLM_IMAGE_TIMEOUT",
    "GLM_MAX_TOKENS",
    "GLM_TEMPERATURE",
    "INSTALL_HINT",
    "LOGGER",
]
