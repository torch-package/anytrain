from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_STEP_CHECKPOINT = re.compile(r"^step-(?P<step>[0-9]+)\.ckpt$")


@dataclass(frozen=True)
class StepCheckpoint:
    """One durable checkpoint in a numerically ordered training series."""

    path: Path
    step: int

    @property
    def name(self) -> str:
        return self.path.stem


def discover_step_checkpoints(directory: str | Path) -> tuple[StepCheckpoint, ...]:
    """Snapshot durable ``step-<digits>.ckpt`` files from one directory."""

    root = _directory(directory)
    checkpoints: dict[int, StepCheckpoint] = {}
    for path in root.iterdir():
        match = _STEP_CHECKPOINT.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        checkpoint = StepCheckpoint(path=path, step=int(match.group("step")))
        previous = checkpoints.get(checkpoint.step)
        if previous is not None:
            raise ValueError(
                "checkpoint directory contains multiple files for step "
                f"{checkpoint.step}: {previous.path.name!r} and {path.name!r}."
            )
        checkpoints[checkpoint.step] = checkpoint
    if not checkpoints:
        raise FileNotFoundError(
            f"checkpoint directory contains no step-<digits>.ckpt files: {root}"
        )
    return tuple(checkpoints[step] for step in sorted(checkpoints))


def evaluate_checkpoints(
    directory: str | Path,
    evaluate: Callable[[StepCheckpoint], object],
) -> tuple[StepCheckpoint, ...]:
    """Evaluate one snapshot of a checkpoint series in step order, failing fast."""

    if not callable(evaluate):
        raise TypeError("evaluate must be callable.")
    checkpoints = discover_step_checkpoints(directory)
    for checkpoint in checkpoints:
        evaluate(checkpoint)
    return checkpoints


def _directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("checkpoint directory must be a string or pathlib.Path.")
    if isinstance(value, str) and not value:
        raise ValueError("checkpoint directory must be non-empty.")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"checkpoint directory is not a directory: {path}")
    return path


__all__ = [
    "StepCheckpoint",
    "discover_step_checkpoints",
    "evaluate_checkpoints",
]
