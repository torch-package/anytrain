from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, cast

from lightning import pytorch as pl


class _ReduceStrategy(Protocol):
    def reduce(self, tensor: object, *, reduce_op: str) -> object: ...


class _BarrierStrategy(Protocol):
    def barrier(self) -> object: ...


class _HyperparameterLogger(Protocol):
    def log_hyperparams(self, params: object) -> object: ...


_T = TypeVar("_T")


def global_step(trainer: pl.Trainer) -> int:
    return int(getattr(trainer, "global_step", 0))


def global_rank(trainer: pl.Trainer) -> int:
    return int(getattr(trainer, "global_rank", 0))


def world_size(trainer: pl.Trainer) -> int:
    return int(getattr(trainer, "world_size", 1))


def is_distributed(trainer: pl.Trainer) -> bool:
    return world_size(trainer) > 1


def is_global_zero(trainer: pl.Trainer) -> bool:
    return bool(getattr(trainer, "is_global_zero", True))


def loggers(trainer: pl.Trainer) -> Sequence[Any]:
    return tuple(getattr(trainer, "loggers", ()) or ())


def logger_experiment(trainer: pl.Trainer) -> object | None:
    logger = getattr(trainer, "logger", None)
    if logger is not None:
        experiment = getattr(logger, "experiment", None)
        if experiment is not None:
            return experiment
    for logger in loggers(trainer):
        experiment = getattr(logger, "experiment", None)
        if experiment is not None:
            return experiment
    return None


def strategy_name(trainer: pl.Trainer) -> str:
    return type(getattr(trainer, "strategy", None)).__name__


def strategy_reduce(
    trainer: pl.Trainer,
    tensor: _T,
    *,
    reduce_op: str,
    error: str | None = None,
) -> _T:
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        raise RuntimeError(
            error
            if error is not None
            else "Distributed operation requires trainer.strategy.reduce."
        )
    return cast(_T, cast(_ReduceStrategy, strategy).reduce(tensor, reduce_op=reduce_op))


def strategy_reduce_or_none(
    trainer: pl.Trainer,
    tensor: _T,
    *,
    reduce_op: str,
) -> _T | None:
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        return None
    return cast(_T, cast(_ReduceStrategy, strategy).reduce(tensor, reduce_op=reduce_op))


def strategy_barrier(
    trainer: pl.Trainer,
    *,
    error: str | None = None,
) -> None:
    strategy = getattr(trainer, "strategy", None)
    barrier = getattr(strategy, "barrier", None)
    if not callable(barrier):
        raise RuntimeError(
            error
            if error is not None
            else "Distributed operation requires trainer.strategy.barrier."
        )
    cast(_BarrierStrategy, strategy).barrier()


def log_hyperparams(trainer: pl.Trainer, params: object) -> None:
    for logger in loggers(trainer):
        log_hyperparams = getattr(logger, "log_hyperparams", None)
        if callable(log_hyperparams):
            cast(_HyperparameterLogger, logger).log_hyperparams(params)


__all__ = [
    "global_rank",
    "global_step",
    "is_distributed",
    "is_global_zero",
    "logger_experiment",
    "log_hyperparams",
    "loggers",
    "strategy_barrier",
    "strategy_name",
    "strategy_reduce",
    "strategy_reduce_or_none",
    "world_size",
]
