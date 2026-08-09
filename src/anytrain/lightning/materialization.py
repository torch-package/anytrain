from __future__ import annotations

import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import auto
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from queue import Empty
from typing import Protocol, TypeVar, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from anytrain._compat import StrEnum

_RESULT_POLL_SECONDS = 0.1
_RESULT_FINAL_WAIT_SECONDS = 1.0
_MAX_ERROR_MESSAGE_CHARS = 16_384
_MAX_ERROR_TRACEBACK_CHARS = 32_768

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class MaterializationPhase(StrEnum):
    FALLBACK = auto()
    MATERIALIZING = auto()
    READY = auto()
    FAILED = auto()


class MaterializationProducer(Protocol[T_co]):
    """Publish an asset in a worker, then load it in the training process."""

    def __call__(self) -> None: ...

    def load(self) -> T_co: ...


class Materialization(Protocol):
    """Task-agnostic controller consumed by :class:`MaterializationCallback`."""

    @property
    def pending(self) -> bool: ...

    def start(self, *, owner: bool) -> None: ...

    def finish(self, *, owner: bool) -> None: ...

    def refresh(self) -> None: ...

    def close(self) -> None: ...


MaterializationProvider = Callable[[Trainer], Materialization | None]


class _ResultQueue(Protocol):
    def put(self, value: _WorkerResult) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> _WorkerResult: ...

    def close(self) -> None: ...

    def join_thread(self) -> None: ...


@dataclass(frozen=True)
class _WorkerResult:
    error_type: str | None = None
    message: str = ""
    traceback: str = ""


class BackgroundMaterialization[T]:
    """Run one publish operation in a spawned background process."""

    def __init__(
        self,
        request_id: str,
        producer: MaterializationProducer[T],
        *,
        worker_name: str = "anytrain-materialization",
        label: str = "materialization",
        daemon: bool = False,
    ) -> None:
        self.request_id = _nonempty_string(request_id, "request_id")
        self.producer = producer
        self.worker_name = _nonempty_string(worker_name, "worker_name")
        self.label = _nonempty_string(label, "label")
        if not isinstance(daemon, bool):
            raise TypeError("daemon must be a boolean.")
        self.daemon = daemon
        self._phase = MaterializationPhase.FALLBACK
        self._process: BaseProcess | None = None
        self._results: _ResultQueue | None = None
        self._error: Exception | None = None

    @property
    def phase(self) -> MaterializationPhase:
        return self._phase

    def start(self, *, owner: bool) -> None:
        if self._phase is not MaterializationPhase.FALLBACK:
            return
        if not owner:
            self._phase = MaterializationPhase.MATERIALIZING
            return
        context = get_context("spawn")
        results = cast(_ResultQueue, context.Queue(maxsize=1))
        process = context.Process(
            target=_run_producer,
            args=(self.producer, results),
            name=self.worker_name,
            daemon=self.daemon,
        )
        try:
            process.start()
        except Exception as error:
            results.close()
            results.join_thread()
            self._error = error
            self._phase = MaterializationPhase.FAILED
            raise
        self._process = process
        self._results = results
        self._phase = MaterializationPhase.MATERIALIZING

    def finish(self, *, owner: bool) -> None:
        if self._phase is MaterializationPhase.READY:
            return
        if self._phase is MaterializationPhase.FAILED:
            if self._error is None:
                raise RuntimeError(f"{self.label} failed without an exception.")
            raise RuntimeError(f"{self.label} previously failed.") from self._error
        if not owner:
            if self._phase is not MaterializationPhase.MATERIALIZING:
                raise RuntimeError(
                    f"{self.label} non-owner was not waiting for publication."
                )
            self._phase = MaterializationPhase.READY
            return
        process = self._process
        if process is None:
            raise RuntimeError(f"{self.label} owner did not start its worker.")
        try:
            failure = self._wait_for_worker(process)
            if failure is None:
                self._phase = MaterializationPhase.READY
                return
            self._error = failure
            self._phase = MaterializationPhase.FAILED
            raise failure
        finally:
            self._cleanup_worker()

    def _wait_for_worker(self, process: BaseProcess) -> Exception | None:
        results = self._results
        if results is None:
            return RuntimeError(f"{self.label} worker has no result queue.")
        while True:
            try:
                result = results.get(timeout=_RESULT_POLL_SECONDS)
            except Empty:
                if process.is_alive():
                    continue
                process.join()
                try:
                    result = results.get(timeout=_RESULT_FINAL_WAIT_SECONDS)
                except Empty:
                    return self._missing_worker_result(process)
            break
        process.join()
        if process.exitcode not in (0, None):
            return RuntimeError(f"{self.label} worker exited with code {process.exitcode}.")
        if not isinstance(result, _WorkerResult):
            return RuntimeError(f"{self.label} worker returned an invalid result.")
        if result.error_type is None:
            return None
        details = f"{result.error_type}: {result.message}"
        if result.traceback:
            details = f"{details}\n{result.traceback}"
        return RuntimeError(f"{self.label} worker failed: {details}")

    def _missing_worker_result(self, process: BaseProcess) -> Exception:
        if process.exitcode not in (0, None):
            return RuntimeError(f"{self.label} worker exited with code {process.exitcode}.")
        return RuntimeError(f"{self.label} worker returned no result.")

    def load_ready(self) -> T:
        if self._phase is not MaterializationPhase.READY:
            raise RuntimeError(
                f"{self.label} cannot load before synchronized publication."
            )
        return self.producer.load()

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            self._error = RuntimeError(f"{self.label} was cancelled during teardown.")
            self._phase = MaterializationPhase.FAILED
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        process = self._process
        if process is not None and not process.is_alive():
            process.close()
        results = self._results
        if results is not None:
            results.close()
            results.join_thread()
        self._process = None
        self._results = None


class MaterializationCallback(Callback):
    """Finalize a background asset at an epoch boundary on every rank."""

    def __init__(
        self,
        materialization: MaterializationProvider,
        *,
        barrier_name: str = "materialization_finished",
        label: str = "materialization",
    ) -> None:
        super().__init__()
        if not callable(materialization):
            raise TypeError("materialization must be callable.")
        self.materialization = materialization
        self.barrier_name = _nonempty_string(barrier_name, "barrier_name")
        self.label = _nonempty_string(label, "label")
        self._active: Materialization | None = None

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        controller: Materialization | None = None
        discovery_error: Exception | None = None
        active = False
        try:
            controller = self.materialization(trainer)
            active = controller is not None and controller.pending
        except Exception as error:
            discovery_error = error
        discovery_failure = _sync_rank_errors(trainer, discovery_error)
        if discovery_failure is not None:
            if controller is not None:
                _close_materialization(controller)
            self._raise_rank_failure(
                trainer,
                "discovery",
                discovery_failure,
                discovery_error,
            )
        try:
            active = _sync_active_state(trainer, active, self.label)
        except Exception:
            if controller is not None:
                _close_materialization(controller)
            raise
        if not active:
            self._active = None
            return
        if controller is None:
            raise RuntimeError(f"{self.label} is active without a local controller.")

        start_error: Exception | None = None
        try:
            controller.start(owner=bool(trainer.is_global_zero))
        except Exception as error:
            start_error = error
        start_failure = _sync_rank_errors(trainer, start_error)
        if start_failure is not None:
            _close_materialization(controller)
            self._raise_rank_failure(trainer, "start", start_failure, start_error)
        self._active = controller

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        controller = self._active
        if controller is None:
            return

        owner_error: Exception | None = None
        if trainer.is_global_zero:
            try:
                controller.finish(owner=True)
            except Exception as error:
                owner_error = error
        payload = _broadcast_error(trainer, owner_error)
        if payload is not None:
            _close_materialization(controller)
            self._active = None
            error_type, message = payload
            failure = RuntimeError(
                f"{self.label} failed on the global owner: {error_type}: {message}"
            )
            if owner_error is not None:
                raise failure from owner_error
            raise failure

        try:
            trainer.strategy.barrier(self.barrier_name)
        except Exception:
            _close_materialization(controller)
            self._active = None
            raise

        publish_error: Exception | None = None
        try:
            controller.finish(owner=False)
        except Exception as error:
            publish_error = error
        publish_failure = _sync_rank_errors(trainer, publish_error)
        if publish_failure is not None:
            _close_materialization(controller)
            self._active = None
            self._raise_rank_failure(
                trainer,
                "publish synchronization",
                publish_failure,
                publish_error,
            )

        refresh_error: Exception | None = None
        try:
            controller.refresh()
        except Exception as error:
            refresh_error = error
        refresh_failure = _sync_rank_errors(trainer, refresh_error)
        if refresh_failure is not None:
            _close_materialization(controller)
            self._active = None
            self._raise_rank_failure(trainer, "refresh", refresh_failure, refresh_error)
        self._active = None

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._close_active()

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        self._close_active()

    def _close_active(self) -> None:
        controller = self._active
        self._active = None
        if controller is not None:
            _close_materialization(controller)

    def _raise_rank_failure(
        self,
        trainer: Trainer,
        stage: str,
        failure: tuple[int, str, str],
        local_error: Exception | None,
    ) -> None:
        rank, error_type, message = failure
        wrapped = RuntimeError(
            f"{self.label} {stage} failed on rank {rank}: {error_type}: {message}"
        )
        if local_error is not None and rank == _global_rank(trainer):
            raise wrapped from local_error
        raise wrapped


def _run_producer(
    producer: MaterializationProducer[object],
    results: _ResultQueue,
) -> None:
    try:
        producer()
    except Exception as error:
        results.put(
            _WorkerResult(
                error_type=type(error).__name__,
                message=_bounded_text(str(error), _MAX_ERROR_MESSAGE_CHARS),
                traceback=_bounded_text(
                    traceback.format_exc(),
                    _MAX_ERROR_TRACEBACK_CHARS,
                ),
            )
        )
    else:
        results.put(_WorkerResult())


def _close_materialization(materialization: Materialization) -> None:
    with suppress(Exception):
        materialization.close()


def _broadcast_error(
    trainer: Trainer,
    error: Exception | None,
) -> tuple[str, str] | None:
    payload = None if error is None else (type(error).__name__, str(error))
    received = trainer.strategy.broadcast(payload, src=0)
    if received is None:
        return None
    if (
        not isinstance(received, tuple)
        or len(received) != 2
        or any(not isinstance(value, str) for value in received)
    ):
        raise TypeError("materialization error broadcast returned an invalid payload.")
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
        received = trainer.strategy.broadcast(value, src=source)
        if received is None:
            continue
        if (
            not isinstance(received, tuple)
            or len(received) != 2
            or any(not isinstance(entry, str) for entry in received)
        ):
            raise TypeError("materialization rank error broadcast was invalid.")
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
        received = trainer.strategy.broadcast(active if source == rank else None, src=source)
        if not isinstance(received, bool):
            raise TypeError("materialization active-state broadcast was invalid.")
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


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n... <truncated {omitted} characters>"


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


__all__ = [
    "BackgroundMaterialization",
    "Materialization",
    "MaterializationCallback",
    "MaterializationPhase",
    "MaterializationProducer",
    "MaterializationProvider",
]
