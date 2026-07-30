from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..env import anytrain_home, hf_home


def hf_cache_dir(
    name: str,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return hf_home() / name


def anytrain_cache_dir(
    name: str,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return anytrain_home() / name


def hf_file(
    *,
    component: str,
    repo_id: str,
    filename: str,
    cache_dir: Path,
    target: Path,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    if target.exists() and not force_download:
        return target

    download = require_hf_function("hf_hub_download", component)
    downloaded = Path(
        download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(cache_dir),
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )
    if not target.exists():
        raise FileNotFoundError(
            f"Expected Hugging Face download to create {target}, but got {downloaded}."
        )
    return target


def hf_snapshot(
    *,
    component: str,
    repo_id: str,
    cache_dir: Path,
    allow_patterns: Sequence[str],
    local_files_only: bool,
    force_download: bool,
) -> Path:
    download = require_hf_function("snapshot_download", component)
    return Path(
        download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )


def require_hf_function(name: str, component: str) -> Callable[..., Any]:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            f"{component} requires `huggingface-hub`. Install it with "
            '`python -m pip install "huggingface-hub>=0.23"`.'
        ) from exc
    return getattr(huggingface_hub, name)
