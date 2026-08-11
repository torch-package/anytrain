from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

type LightningCheckpoint = dict[str, Any]
type _CheckpointHook = Callable[[LightningCheckpoint], None]


def load_lightning_checkpoint(
    module: nn.Module,
    path: str | PathLike[str],
    *,
    call_module_hook: bool = True,
) -> LightningCheckpoint:
    """Load one full Lightning checkpoint into ``module`` with strict state matching.

    Checkpoints are loaded onto CPU. When enabled, an ``on_load_checkpoint`` hook is
    called if the module provides one. The validated payload is returned so the
    caller can interpret project-owned metadata.
    """
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Lightning checkpoint does not exist: {checkpoint_path}."
        )

    value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = _checkpoint(value)
    if call_module_hook:
        _call_module_hook(module, checkpoint)
    module.load_state_dict(_state_dict(checkpoint), strict=True)
    return checkpoint


def _checkpoint(value: object) -> LightningCheckpoint:
    if not isinstance(value, Mapping):
        raise TypeError("Lightning checkpoint root must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("Lightning checkpoint root keys must be strings.")
    return dict(cast(Mapping[str, Any], value))


def _state_dict(checkpoint: LightningCheckpoint) -> Mapping[str, Any]:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Lightning checkpoint state_dict must be a mapping.")
    if not all(isinstance(key, str) for key in state):
        raise TypeError("Lightning checkpoint state_dict keys must be strings.")
    return cast(Mapping[str, Any], state)


def _call_module_hook(module: nn.Module, checkpoint: LightningCheckpoint) -> None:
    hook = getattr(module, "on_load_checkpoint", None)
    if hook is None:
        return
    if not callable(hook):
        raise TypeError("module on_load_checkpoint must be callable.")
    cast(_CheckpointHook, hook)(checkpoint)


__all__ = ["LightningCheckpoint", "load_lightning_checkpoint"]
