"""Rank agreement and deterministic error selection for lifecycle adapters."""
from __future__ import annotations

from typing import cast

from lightning import Trainer

from . import _capability


def _sync_rank_errors(
    trainer: Trainer,
    error: Exception | None,
    *,
    requirement: str,
    invalid: str,
) -> tuple[int, str, str] | None:
    rank, world_size = _rank_world(trainer)
    payload = None if error is None else (type(error).__name__, str(error))
    if world_size == 1:
        if payload is None:
            return None
        return (rank, *payload)
    failures: list[tuple[int, str, str]] = []
    for source in range(world_size):
        value = payload if source == rank else None
        received = _capability.strategy_broadcast(
            trainer,
            value,
            src=source,
            error=requirement,
        )
        if received is None:
            continue
        if (
            not isinstance(received, tuple)
            or len(received) != 2
            or any(not isinstance(entry, str) for entry in received)
        ):
            raise TypeError(invalid)
        error_type, message = cast(tuple[str, str], received)
        failures.append((source, error_type, message))
    if not failures:
        return None
    return min(failures, key=lambda failure: failure[0])


def _sync_active_state(
    trainer: Trainer, active: bool, label: str, *, requirement: str, invalid: str
) -> bool:
    rank, world_size = _rank_world(trainer)
    if world_size == 1:
        return active
    states: list[bool] = []
    for source in range(world_size):
        received = _capability.strategy_broadcast(
            trainer,
            active if source == rank else None,
            src=source,
            error=requirement,
        )
        if not isinstance(received, bool):
            raise TypeError(invalid)
        states.append(received)
    if len(set(states)) != 1:
        details = ", ".join(
            f"rank {source}={state}" for source, state in enumerate(states)
        )
        raise RuntimeError(f"{label} active state differs across ranks: {details}.")
    return states[0]


def _rank_world(trainer: Trainer) -> tuple[int, int]:
    rank = _global_rank(trainer)
    world_size = int(getattr(trainer, "world_size", 1))
    if world_size < 1:
        raise ValueError("trainer world_size must be positive.")
    if rank >= world_size:
        raise ValueError("trainer global_rank must be smaller than world_size.")
    return rank, world_size


def _global_rank(trainer: Trainer) -> int:
    value = getattr(trainer, "global_rank", None)
    if value is None:
        return 0 if trainer.is_global_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("trainer global_rank must be a non-negative integer.")
    return value
