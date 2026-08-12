from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from anytrain.env import anytrain_home, hf_home


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
    revision: str | None = None,
    local_files_only: bool,
    force_download: bool,
    allow_unpinned_revision: bool = False,
) -> Path:
    _bool(local_files_only, name="local_files_only")
    _bool(force_download, name="force_download")
    _bool(allow_unpinned_revision, name="allow_unpinned_revision")
    if target.exists() and not force_download:
        return target

    revision_kwargs = _revision_kwargs(
        component,
        revision=revision,
        local_files_only=local_files_only,
        allow_unpinned_revision=allow_unpinned_revision,
    )
    download = require_hf_function("hf_hub_download", component)
    downloaded = Path(
        download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(cache_dir),
            **revision_kwargs,
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
    revision: str | None = None,
    local_files_only: bool,
    force_download: bool,
    allow_unpinned_revision: bool = False,
) -> Path:
    _bool(local_files_only, name="local_files_only")
    _bool(force_download, name="force_download")
    _bool(allow_unpinned_revision, name="allow_unpinned_revision")
    revision_kwargs = _revision_kwargs(
        component,
        revision=revision,
        local_files_only=local_files_only,
        allow_unpinned_revision=allow_unpinned_revision,
    )
    download = require_hf_function("snapshot_download", component)
    return Path(
        download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            **revision_kwargs,
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


def _revision_kwargs(
    component: str,
    *,
    revision: str | None,
    local_files_only: bool,
    allow_unpinned_revision: bool,
) -> dict[str, str]:
    if revision is not None:
        if not isinstance(revision, str):
            raise TypeError(f"{component} revision must be a string.")
        if revision == "":
            raise ValueError(f"{component} revision must not be empty.")
        return {"revision": revision}
    if local_files_only or allow_unpinned_revision:
        return {}
    raise ValueError(
        f"{component} remote downloads require a pinned Hugging Face revision. "
        "Pass revision=<commit-sha> or set allow_unpinned_revision=True for an "
        "explicitly unsafe, non-reproducible download."
    )


def _bool(value: object, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
