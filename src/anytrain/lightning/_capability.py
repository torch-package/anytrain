from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypedDict, TypeVar, cast

from lightning import pytorch as pl


class _ReduceStrategy(Protocol):
    def reduce(self, tensor: object, *, reduce_op: str) -> object: ...


class _BarrierStrategy(Protocol):
    def barrier(self, name: str | None = None) -> object: ...


class _AllGatherStrategy(Protocol):
    def all_gather(self, tensor: object) -> object: ...


class _BroadcastStrategy(Protocol):
    def broadcast(self, value: object, *, src: int) -> object: ...


class _HyperparameterLogger(Protocol):
    def log_hyperparams(self, params: object) -> object: ...


_T = TypeVar("_T")


class _CollectiveFields(TypedDict, total=False):
    rank: int
    world_size: int
    global_step: int
    strategy: str
    value_type: str
    tensor_dtype: object
    tensor_device: object
    tensor_shape: object


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
    from .audit import audit_collective

    return cast(
        _T,
        audit_collective(
            "strategy.reduce",
            lambda: cast(_ReduceStrategy, strategy).reduce(
                tensor,
                reduce_op=reduce_op,
            ),
            **_collective_fields(trainer, tensor),
            reduce_op=reduce_op,
        ),
    )


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
    return strategy_reduce(trainer, tensor, reduce_op=reduce_op)


def strategy_barrier(
    trainer: pl.Trainer,
    *,
    name: str | None = None,
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
    from .audit import audit_collective

    audit_collective(
        "strategy.barrier",
        lambda: (
            cast(_BarrierStrategy, strategy).barrier()
            if name is None
            else cast(_BarrierStrategy, strategy).barrier(name)
        ),
        **_collective_fields(trainer),
        barrier_name=name,
    )


def strategy_all_gather(
    trainer: pl.Trainer,
    tensor: _T,
    *,
    error: str | None = None,
) -> _T:
    strategy = getattr(trainer, "strategy", None)
    all_gather = getattr(strategy, "all_gather", None)
    if not callable(all_gather):
        raise RuntimeError(
            error
            if error is not None
            else "Distributed operation requires trainer.strategy.all_gather."
        )
    from .audit import audit_collective

    return cast(
        _T,
        audit_collective(
            "strategy.all_gather",
            lambda: cast(_AllGatherStrategy, strategy).all_gather(tensor),
            **_collective_fields(trainer, tensor),
        ),
    )


def strategy_broadcast(
    trainer: pl.Trainer,
    value: _T,
    *,
    src: int,
    error: str | None = None,
) -> _T:
    strategy = getattr(trainer, "strategy", None)
    broadcast = getattr(strategy, "broadcast", None)
    if not callable(broadcast):
        raise RuntimeError(
            error
            if error is not None
            else "Distributed operation requires trainer.strategy.broadcast."
        )
    from .audit import audit_collective

    return cast(
        _T,
        audit_collective(
            "strategy.broadcast",
            lambda: cast(_BroadcastStrategy, strategy).broadcast(value, src=src),
            **_collective_fields(trainer, value),
            source_rank=src,
        ),
    )


def _collective_fields(
    trainer: pl.Trainer, value: object | None = None
) -> _CollectiveFields:
    fields: _CollectiveFields = {
        "rank": global_rank(trainer),
        "world_size": world_size(trainer),
        "global_step": global_step(trainer),
        "strategy": strategy_name(trainer),
    }
    if value is None:
        return fields
    fields["value_type"] = type(value).__name__
    for field, attribute in (
        ("tensor_dtype", "dtype"),
        ("tensor_device", "device"),
        ("tensor_shape", "shape"),
    ):
        resolved = getattr(value, attribute, None)
        if resolved is not None:
            fields[field] = resolved
    return fields


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
    "strategy_all_gather",
    "strategy_broadcast",
    "strategy_name",
    "strategy_reduce",
    "strategy_reduce_or_none",
    "world_size",
]
