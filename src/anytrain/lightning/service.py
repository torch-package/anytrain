from __future__ import annotations

import signal
from collections.abc import Callable
from contextlib import suppress
from types import FrameType
from typing import Any, Protocol, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from . import _capability

_SignalHandler = Callable[[int, FrameType | None], Any]


class ManagedService(Protocol):
    """A task-agnostic long-lived service owned by the global training rank."""

    def start(self, *, owner: bool) -> None: ...

    def check(self, *, owner: bool) -> None: ...

    def close(self, *, owner: bool) -> None: ...


ManagedServiceProvider = Callable[[Trainer], ManagedService | None]


class _OnceSignalHandler:
    """Delegate only the first signal, including re-entrant delivery."""

    def __init__(self, handler: _SignalHandler) -> None:
        self._handler = handler
        self._handled = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        if self._handled:
            return
        self._handled = True
        self._handler(signum, frame)


class ManagedServiceCallback(Callback):
    """Own and monitor one long-lived service on the global training rank."""

    def __init__(
        self,
        service: ManagedServiceProvider,
        *,
        check_every_n_steps: int = 1,
        label: str = "managed service",
        guard_sigterm: bool = True,
    ) -> None:
        super().__init__()
        if not callable(service):
            raise TypeError("service must be callable.")
        _positive_int(check_every_n_steps, "check_every_n_steps")
        if not isinstance(guard_sigterm, bool):
            raise TypeError("guard_sigterm must be a boolean.")
        self.service = service
        self.check_every_n_steps = check_every_n_steps
        self.label = _nonempty_string(label, "label")
        self.guard_sigterm = guard_sigterm
        self._active: ManagedService | None = None
        self._last_checked_step: int | None = None
        self._sigterm_guard: _OnceSignalHandler | None = None

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        service: ManagedService | None = None
        discovery_error: Exception | None = None
        try:
            service = self.service(trainer)
        except Exception as error:
            discovery_error = error
        try:
            discovery_failure = _sync_rank_errors(trainer, discovery_error)
        except Exception:
            _close_owner_safely(trainer, service)
            raise
        if discovery_failure is not None:
            _close_owner_safely(trainer, service)
            self._raise_rank_failure(
                trainer,
                "discovery",
                discovery_failure,
                discovery_error,
            )

        try:
            active = _sync_active_state(trainer, service is not None, self.label)
        except Exception:
            _close_owner_safely(trainer, service)
            raise
        if not active:
            self._active = None
            return
        if service is None:
            raise RuntimeError(f"{self.label} is active without a local service.")

        start_error = _owner_error(trainer, service.start)
        try:
            start_failure = _broadcast_owner_error(trainer, start_error, self.label)
        except Exception:
            _close_owner_safely(trainer, service)
            raise
        if start_failure is not None:
            _close_owner_safely(trainer, service)
            self._raise_owner_failure("start", start_failure, start_error)
        self._active = service
        self._last_checked_step = None

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        if self._active is None or not self.guard_sigterm:
            return
        guard_error: Exception | None = None
        try:
            self._install_sigterm_guard()
        except Exception as error:
            guard_error = error
        try:
            guard_failure = _sync_rank_errors(trainer, guard_error)
        except Exception:
            self._close_active_safely(trainer)
            raise
        if guard_failure is not None:
            self._close_active_safely(trainer)
            self._raise_rank_failure(
                trainer,
                "SIGTERM guard installation",
                guard_failure,
                guard_error,
            )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        service = self._active
        if service is None or not self._should_check(trainer):
            return
        check_error = _owner_error(trainer, service.check)
        try:
            check_failure = _broadcast_owner_error(trainer, check_error, self.label)
        except Exception:
            self._close_active_safely(trainer)
            raise
        if check_failure is not None:
            self._close_active_safely(trainer)
            self._raise_owner_failure("health check", check_failure, check_error)

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        self._close_active_safely(trainer)

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        self._close_active(trainer)

    def on_fit_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        self._close_active(trainer)

    def _install_sigterm_guard(self) -> None:
        current = signal.getsignal(signal.SIGTERM)
        if current is self._sigterm_guard:
            return
        if not callable(current):
            raise RuntimeError(
                "Lightning must register its SIGTERM handler before managed services start."
            )
        guard = _OnceSignalHandler(cast(_SignalHandler, current))
        # Lightning owns restoration. Retaining this guard through teardown prevents a
        # repeated process-group SIGTERM from re-entering its distributed handler.
        signal.signal(signal.SIGTERM, guard)
        self._sigterm_guard = guard

    def _should_check(self, trainer: Trainer) -> bool:
        step = _capability.global_step(trainer)
        if step % self.check_every_n_steps != 0 or self._last_checked_step == step:
            return False
        self._last_checked_step = step
        return True

    def _close_active(self, trainer: Trainer) -> None:
        service = self._active
        self._active = None
        if service is None:
            return
        close_error = _owner_error(trainer, service.close)
        close_failure = _broadcast_owner_error(trainer, close_error, self.label)
        if close_failure is not None:
            self._raise_owner_failure("close", close_failure, close_error)

    def _close_active_safely(self, trainer: Trainer) -> None:
        service = self._active
        self._active = None
        _close_owner_safely(trainer, service)

    def _raise_owner_failure(
        self,
        operation: str,
        failure: tuple[str, str],
        local_error: Exception | None,
    ) -> None:
        error_type, message = failure
        wrapped = RuntimeError(
            f"{self.label} {operation} failed on the global owner: "
            f"{error_type}: {message}"
        )
        if local_error is not None:
            raise wrapped from local_error
        raise wrapped

    def _raise_rank_failure(
        self,
        trainer: Trainer,
        operation: str,
        failure: tuple[int, str, str],
        local_error: Exception | None,
    ) -> None:
        rank, error_type, message = failure
        wrapped = RuntimeError(
            f"{self.label} {operation} failed on rank {rank}: "
            f"{error_type}: {message}"
        )
        if local_error is not None and rank == _global_rank(trainer):
            raise wrapped from local_error
        raise wrapped


def _owner_error(
    trainer: Trainer,
    operation: Callable[..., None],
) -> Exception | None:
    if not trainer.is_global_zero:
        return None
    try:
        operation(owner=True)
    except Exception as error:
        return error
    return None


def _close_owner_safely(
    trainer: Trainer,
    service: ManagedService | None,
) -> None:
    if service is None or not trainer.is_global_zero:
        return
    with suppress(Exception):
        service.close(owner=True)


def _broadcast_owner_error(
    trainer: Trainer,
    error: Exception | None,
    label: str,
) -> tuple[str, str] | None:
    payload = None if error is None else (type(error).__name__, str(error))
    received = _capability.strategy_broadcast(
        trainer,
        payload,
        src=0,
        error=f"{label} owner error requires trainer.strategy.broadcast.",
    )
    if received is None:
        return None
    if (
        not isinstance(received, tuple)
        or len(received) != 2
        or any(not isinstance(value, str) for value in received)
    ):
        raise TypeError(f"{label} owner error broadcast returned an invalid payload.")
    return cast(tuple[str, str], received)


def _sync_rank_errors(
    trainer: Trainer,
    error: Exception | None,
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
            error="managed service rank error requires trainer.strategy.broadcast.",
        )
        if received is None:
            continue
        if (
            not isinstance(received, tuple)
            or len(received) != 2
            or any(not isinstance(entry, str) for entry in received)
        ):
            raise TypeError("managed service rank error broadcast was invalid.")
        error_type, message = cast(tuple[str, str], received)
        failures.append((source, error_type, message))
    if not failures:
        return None
    return min(failures, key=lambda failure: failure[0])


def _sync_active_state(trainer: Trainer, active: bool, label: str) -> bool:
    rank, world_size = _rank_world(trainer)
    if world_size == 1:
        return active
    states: list[bool] = []
    for source in range(world_size):
        received = _capability.strategy_broadcast(
            trainer,
            active if source == rank else None,
            src=source,
            error="managed service active-state requires trainer.strategy.broadcast.",
        )
        if not isinstance(received, bool):
            raise TypeError("managed service active-state broadcast was invalid.")
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


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


__all__ = [
    "ManagedService",
    "ManagedServiceCallback",
    "ManagedServiceProvider",
]
