from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from .._cache import hf_cache_dir, hf_snapshot

DEFAULT_HF_REPO_ID = "SparkAudio/Spark-TTS-0.5B"
SNAPSHOT_PATTERNS = (
    "config.yaml",
    "BiCodec/*",
    "wav2vec2-large-xlsr-53/*",
)


class BiCodecAssets(TypedDict):
    cache_dir: Path
    model_dir: Path


def ensure_bicodec_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    allow_unpinned_revision: bool = False,
) -> BiCodecAssets:
    root = hf_cache_dir("bicodec", cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    model_dir = hf_snapshot(
        component="BiCodec checkpoint download",
        repo_id=repo_id,
        cache_dir=root,
        allow_patterns=SNAPSHOT_PATTERNS,
        revision=revision,
        local_files_only=local_files_only,
        force_download=force_download,
        allow_unpinned_revision=allow_unpinned_revision,
    )
    _validate_model_dir(model_dir)
    return {
        "cache_dir": root,
        "model_dir": model_dir,
    }


def _validate_model_dir(model_dir: Path) -> None:
    required = (
        model_dir / "config.yaml",
        model_dir / "BiCodec" / "config.yaml",
        model_dir / "BiCodec" / "model.safetensors",
        model_dir / "wav2vec2-large-xlsr-53" / "config.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"BiCodec model directory is incomplete: {formatted}.")
