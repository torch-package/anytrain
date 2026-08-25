"""Bounded, rank-local background production for dataset cost metadata."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from queue import Full, Queue


@dataclass(frozen=True)
class CostProducerStats:
    """Cumulative observations for one producer."""

    submitted: int
    completed: int
    failed: int
    dropped: int


class BackgroundCostProducer:
    """Run rank-local cost misses ahead of the planner in one bounded thread.

    The callback owns persistence (normally an anydataset manifest-backed cost
    source). The producer only controls lifecycle and backpressure, so a
    missing or full queue never changes planner correctness: the caller can
    compute the cost synchronously and continue.
    """

    def __init__(
        self,
        compute: Callable[[int], int],
        *,
        max_queue_size: int = 256,
        thread_name: str = "anytrain-cost-producer",
    ) -> None:
        if not callable(compute):
            raise TypeError("compute must be callable.")
        _positive_int(max_queue_size, "max_queue_size")
        _nonempty_string(thread_name, "thread_name")
        self.compute = compute
        self.max_queue_size = max_queue_size
        self.thread_name = thread_name
        self._queue: Queue[int | None] = Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._inflight: set[int] = set()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._closing = False
        self._closed = False

    @property
    def started(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> CostProducerStats:
        with self._lock:
            return CostProducerStats(
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                dropped=self._dropped,
            )

    def start(self) -> None:
        """Start the producer thread once."""

        with self._lock:
            if self._closed:
                raise RuntimeError("cost producer is closed.")
            if self._thread is not None:
                if self._error is not None:
                    raise RuntimeError("cost producer previously failed.") from self._error
                return
            thread = threading.Thread(
                target=self._run,
                name=self.thread_name,
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def submit(self, index: int) -> bool:
        """Queue one index without blocking; return false under backpressure."""

        resolved = _non_negative_int(index, "index")
        if self._thread is None:
            raise RuntimeError("cost producer must be started before submit().")
        self.check()
        with self._lock:
            if self._closed or self._closing:
                return False
            if self._error is not None:
                raise RuntimeError("background cost producer failed.") from self._error
            if resolved in self._inflight:
                return True
            self._inflight.add(resolved)
            try:
                self._queue.put_nowait(resolved)
            except Full:
                self._inflight.discard(resolved)
                self._dropped += 1
                return False
            self._submitted += 1
        return True

    def prefetch(self, indexes: Iterable[int]) -> int:
        """Submit as many future indexes as the bounded queue accepts."""

        accepted = 0
        for index in indexes:
            if self.submit(index):
                accepted += 1
        return accepted

    def check(self) -> None:
        """Raise the worker failure at a caller-controlled boundary."""

        error = self._error
        if error is not None:
            raise RuntimeError("background cost producer failed.") from error

    def close(self, *, timeout: float | None = None) -> None:
        """Drain queued work and stop the producer thread."""

        thread = self._thread
        if thread is None:
            return
        self.check()
        with self._lock:
            if self._closed and not thread.is_alive():
                return
            enqueue_stop = not self._closing
            self._closing = True
        if enqueue_stop:
            try:
                self._queue.put(None, timeout=timeout)
            except Full as exc:
                with self._lock:
                    self._closing = False
                raise TimeoutError(
                    "background cost producer did not accept the stop request in time."
                ) from exc
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("background cost producer did not stop in time.")
        with self._lock:
            self._closed = True
        self.check()

    def __enter__(self) -> BackgroundCostProducer:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _run(self) -> None:
        while True:
            index = self._queue.get()
            try:
                if index is None:
                    return
                self.compute(index)
            except BaseException as error:
                with self._lock:
                    self._failed += 1
                    self._error = error
                return
            finally:
                if index is not None:
                    with self._lock:
                        self._inflight.discard(index)
                        if self._error is None:
                            self._completed += 1
                self._queue.task_done()


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


__all__ = ["BackgroundCostProducer", "CostProducerStats"]
