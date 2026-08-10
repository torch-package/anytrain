from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from .._cache import hf_cache_dir, hf_file

DEFAULT_HF_REPO_ID = "Yidiii/UniCodec_ckpt"
DEFAULT_CONFIG_NAME = "unicodec_frame75_10s_nq1_code16384_dim512_acousitic.yaml"
DEFAULT_CHECKPOINT_FILENAME = "unicode.ckpt"


class UniCodecAssets(TypedDict):
    cache_dir: Path
    config: Path
    checkpoint: Path


def ensure_unicodec_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    revision: str | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    allow_unpinned_revision: bool = False,
) -> UniCodecAssets:
    root = hf_cache_dir("unicodec", cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = hf_file(
        component="UniCodec checkpoint download",
        repo_id=repo_id,
        cache_dir=root,
        filename=checkpoint_filename,
        target=root / checkpoint_filename,
        revision=revision,
        local_files_only=local_files_only,
        force_download=force_download,
        allow_unpinned_revision=allow_unpinned_revision,
    )

    return {
        "cache_dir": root,
        "config": _default_config_path(config_name),
        "checkpoint": checkpoint,
    }


def _default_config_path(name: str) -> Path:
    try:
        from unicodec import default_config_path  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "UniCodec integration requires the installable UniCodec fork. Install it with "
            '`python -m pip install "unicodec @ git+https://github.com/rookiejune/UniCodec.git"`.'
        ) from exc

    path = Path(default_config_path(name))
    if not path.exists():
        raise FileNotFoundError(f"UniCodec config does not exist: {path}.")
    return path
