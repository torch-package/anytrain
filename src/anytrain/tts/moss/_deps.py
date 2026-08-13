from __future__ import annotations

from typing import Any

INSTALL_HINT = (
    "Install MOSS-TTS dependencies with `python -m pip install 'anytrain[moss-tts]'`."
)


def load_transformers_auto_model_class() -> type[Any]:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "`anytrain.tts.moss` requires `transformers` to load remote-code "
            f"MOSS-TTS checkpoints. {INSTALL_HINT}"
        ) from exc
    return AutoModel


def load_transformers_auto_processor_class() -> type[Any]:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "`anytrain.tts.moss` requires `transformers` to load remote-code "
            f"MOSS-TTS processors. {INSTALL_HINT}"
        ) from exc
    return AutoProcessor
