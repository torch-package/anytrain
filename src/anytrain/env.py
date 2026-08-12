"""Environment-backed cache roots for optional anytrain integrations.

The module exposes one anytrain root and resolves third-party cache variables from
it only when the user has not already set those variables in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

ANYTRAIN_HOME_ENV = "ANYTRAIN_HOME"
HF_HOME_ENV = "HF_HOME"
TORCH_HOME_ENV = "TORCH_HOME"
WHISPER_ROOT_ENV = "ANYTRAIN_WHISPER_ROOT"

DEFAULT_ANYTRAIN_HOME = Path.home() / ".anytrain"
DEFAULT_HF_HOME = DEFAULT_ANYTRAIN_HOME / "huggingface"
DEFAULT_TORCH_HOME = DEFAULT_ANYTRAIN_HOME / "torch"
DEFAULT_WHISPER_ROOT = DEFAULT_ANYTRAIN_HOME / "whisper"


def anytrain_home() -> Path:
    value = os.environ.get(ANYTRAIN_HOME_ENV)
    if value is None:
        return DEFAULT_ANYTRAIN_HOME
    if value == "":
        raise ValueError(f"{ANYTRAIN_HOME_ENV} must not be empty.")
    return Path(value).expanduser()


def hf_home() -> Path:
    return env_path(HF_HOME_ENV, "huggingface")


def torch_home() -> Path:
    return env_path(TORCH_HOME_ENV, "torch")


def whisper_root() -> Path:
    return env_path(WHISPER_ROOT_ENV, "whisper")


def env_path(name: str, relpath: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        path = anytrain_home() / relpath
        os.environ[name] = str(path)
        return path
    if value == "":
        raise ValueError(f"{name} must not be empty.")
    return Path(value).expanduser()


__all__ = [
    "ANYTRAIN_HOME_ENV",
    "DEFAULT_ANYTRAIN_HOME",
    "DEFAULT_HF_HOME",
    "DEFAULT_TORCH_HOME",
    "DEFAULT_WHISPER_ROOT",
    "HF_HOME_ENV",
    "TORCH_HOME_ENV",
    "WHISPER_ROOT_ENV",
    "anytrain_home",
    "env_path",
    "hf_home",
    "torch_home",
    "whisper_root",
]
