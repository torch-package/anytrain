"""Register the upstream CosyVoice model implementation with vLLM.

The vLLM plugin entry point loads this module in both the client and spawned
engine-core processes. The registered class stays lazy so importing the plugin
does not initialize the CosyVoice model or CUDA.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast


def register() -> None:
    """Register CosyVoice's bias-aware Qwen2 implementation."""

    registry = cast(Any, import_module("vllm")).ModelRegistry
    registry.register_model(
        "CosyVoice2ForCausalLM",
        "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
    )


__all__ = ["register"]
