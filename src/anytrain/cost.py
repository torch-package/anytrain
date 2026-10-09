"""Bounded, rank-local background production for dataset cost metadata."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from queue import Empty, PriorityQueue


@dataclass(frozen=True)
class CostProducerStats:
    """Cumulative observations for one producer."""

    submitted: int
    completed: int
    failed: int
    dropped: int


class BackgroundCostProducer:
    """Run rank-local cost misses ahead of the planner.

    The callback owns persistence (normally an anydataset manifest-backed cost
    source). The producer only controls lifecycle and backpressure, so a
    missing or full queue never changes planner correctness: the caller can
    compute the cost synchronously and continue.

    Normal prefetch submits are bounded by ``max_queue_size`` and may drop under
    backpressure. ``urgent=True`` submits jump ahead of that backlog so a
    planner miss is not stuck behind thousands of speculative indexes.
    ``num_workers>1`` drains urgent and normal work in parallel.
    When ``compute_many`` is set, each worker drains up to ``batch_size``
    queued indexes and hands them to one coalesced write path.
    """

    _PRIORITY_URGENT = 0
    _PRIORITY_NORMAL = 1
    _PRIORITY_STOP = 2

    def __init__(
        self,
        compute: Callable[[int], int],
        *,
        compute_many: Callable[[Sequence[int]], Sequence[int]] | None = None,
        batch_size: int = 1,
        max_queue_size: int = 256,
        num_workers: int = 1,
        thread_name: str = "anytrain-cost-producer",
    ) -> None:
        if not callable(compute):
            raise TypeError("compute must be callable.")
        if compute_many is not None and not callable(compute_many):
            raise TypeError("compute_many must be callable or None.")
        _positive_int(batch_size, "batch_size")
        _positive_int(max_queue_size, "max_queue_size")
        _positive_int(num_workers, "num_workers")
        _nonempty_string(thread_name, "thread_name")
        if batch_size > 1 and compute_many is None:
            raise ValueError("batch_size>1 requires compute_many.")
        self.compute = compute
        self.compute_many = compute_many
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.thread_name = thread_name
        # (priority, sequence, index|None) — lower priority value runs first.
        self._queue: PriorityQueue[tuple[int, int, int | None]] = PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._inflight: set[int] = set()
        self._active: set[int] = set()
        self._queued_normal = 0
        self._queued_urgent = 0
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._urgent_submitted = 0
        self._closing = False
        self._closed = False

    @property
    def started(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    @property
    def pending(self) -> int:
        with self._lock:
            return self._queued_normal + self._queued_urgent

    @property
    def stats(self) -> CostProducerStats:
        with self._lock:
            return CostProducerStats(
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                dropped=self._dropped,
            )

    def wait_snapshot(self, index: int | None = None) -> dict[str, int | bool]:
        """Return wait-attribution fields for planner miss diagnostics."""

        with self._lock:
            alive = any(thread.is_alive() for thread in self._threads)
            inflight = set(self._inflight)
            active = set(self._active)
            submitted = self._submitted
            completed = self._completed
            failed = self._failed
            dropped = self._dropped
            queue_size = self._queued_normal + self._queued_urgent
            urgent = self._queued_urgent
            error = self._error
        if index is not None and index in active:
            current = index
        elif active:
            current = next(iter(sorted(active)))
        else:
            current = -1
        return {
            "alive": alive,
            "queue_size": queue_size,
            "pending_count": len(inflight),
            "index_pending": index is not None and index in inflight,
            "index_active": index is not None and index in active,
            "in_cost_row_index": current,
            "submitted": submitted,
            "completed": completed,
            "failed": failed,
            "dropped": dropped,
            "inflight_count": len(inflight),
            "urgent_queued": urgent,
            "active_workers": len(active),
            "error": "" if error is None else f"{type(error).__name__}: {error}",
        }

    def start(self) -> None:
        """Start producer worker threads once."""

        with self._lock:
            if self._closed:
                raise RuntimeError("cost producer is closed.")
            if self._threads:
                if self._error is not None:
                    raise RuntimeError("cost producer previously failed.") from self._error
                return
            threads = [
                threading.Thread(
                    target=self._run,
                    name=f"{self.thread_name}-{worker}",
                    daemon=True,
                )
                for worker in range(self.num_workers)
            ]
            self._threads = threads
            for thread in threads:
                thread.start()

    def submit(self, index: int, *, urgent: bool = False) -> bool:
        """Queue one index without blocking; return false under backpressure."""

        resolved = _non_negative_int(index, "index")
        if type(urgent) is not bool:
            raise TypeError("urgent must be a boolean.")
        if not self._threads:
            raise RuntimeError("cost producer must be started before submit().")
        self.check()
        with self._lock:
            if self._closed or self._closing:
                return False
            if self._error is not None:
                raise RuntimeError("background cost producer failed.") from self._error
            if resolved in self._inflight:
                if urgent:
                    self._enqueue_locked(resolved, urgent=True, count_submit=False)
                return True
            if not urgent and self._queued_normal >= self.max_queue_size:
                self._dropped += 1
                return False
            self._inflight.add(resolved)
            self._enqueue_locked(resolved, urgent=urgent, count_submit=True)
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
        """Drain queued work and stop producer threads."""

        threads = list(self._threads)
        if not threads:
            return
        with self._lock:
            if self._closed and not any(thread.is_alive() for thread in threads):
                return
            enqueue_stop = not self._closing
            self._closing = True
            if enqueue_stop:
                for _ in range(self.num_workers):
                    self._queue.put((self._PRIORITY_STOP, next(self._seq), None))
        for thread in threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("background cost producer did not stop in time.")
        with self._lock:
            self._closed = True
            # A failed worker cannot consume pending items or the stop sentinel.
            while True:
                try:
                    index = self._queue.get_nowait()
                except Empty:
                    break
                if index is not None:
                    self._inflight.discard(index)
                    self._dropped += 1
                self._queue.task_done()
        self.check()

    def __enter__(self) -> BackgroundCostProducer:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _enqueue_locked(self, index: int, *, urgent: bool, count_submit: bool) -> None:
        priority = self._PRIORITY_URGENT if urgent else self._PRIORITY_NORMAL
        if urgent:
            self._queued_urgent += 1
            self._urgent_submitted += 1
        else:
            self._queued_normal += 1
        self._queue.put((priority, next(self._seq), index))
        if count_submit:
            self._submitted += 1

    def _note_dequeued(self, priority: int) -> None:
        with self._lock:
            if priority == self._PRIORITY_NORMAL:
                self._queued_normal = max(0, self._queued_normal - 1)
            elif priority == self._PRIORITY_URGENT:
                self._queued_urgent = max(0, self._queued_urgent - 1)

    def _run(self) -> None:
        while True:
            priority, _seq, index = self._queue.get()
            self._note_dequeued(priority)
            if index is None:
                self._queue.task_done()
                return

            batch: list[int] = []
            got_counts = 1  # first get(); extras from get_nowait
            try:
                with self._lock:
                    if index in self._inflight and index not in self._active:
                        self._active.add(index)
                        batch.append(index)

                while self.compute_many is not None and len(batch) < self.batch_size:
                    try:
                        next_priority, next_seq, next_index = self._queue.get_nowait()
                    except Empty:
                        break
                    got_counts += 1
                    self._note_dequeued(next_priority)
                    if next_index is None:
                        # Preserve stop for this worker; unfinished count stays
                        # balanced by task_done + re-put below.
                        self._queue.put((next_priority, next_seq, next_index))
                        break
                    with self._lock:
                        if (
                            next_index in self._inflight
                            and next_index not in self._active
                            and next_index not in batch
                        ):
                            self._active.add(next_index)
                            batch.append(next_index)

                if not batch:
                    continue

                if self.compute_many is not None:
                    self.compute_many(batch)
                else:
                    self.compute(batch[0])
            except BaseException as error:
                with self._lock:
                    self._failed += 1
                    self._error = error
                return
            finally:
                if batch:
                    with self._lock:
                        for done_index in batch:
                            self._active.discard(done_index)
                            self._inflight.discard(done_index)
                        if self._error is None:
                            self._completed += len(batch)
                for _ in range(got_counts):
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
