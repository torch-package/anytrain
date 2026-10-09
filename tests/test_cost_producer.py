from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Event, Thread
from time import monotonic, sleep

from anytrain.cost import BackgroundCostProducer


def _wait_until(predicate: Callable[[], bool]) -> None:
    deadline = monotonic() + 1.0
    while not predicate() and monotonic() < deadline:
        sleep(0.001)
    assert predicate()


def test_background_cost_producer_is_bounded_and_drains() -> None:
    completed: list[int] = []
    release = Event()
    started = Event()

    def compute(index: int) -> int:
        started.set()
        release.wait()
        completed.append(index)
        return index + 1

    producer = BackgroundCostProducer(
        compute,
        max_queue_size=1,
    )
    producer.start()
    assert producer.submit(1)
    assert started.wait(timeout=1)
    assert producer.submit(2)
    assert not producer.submit(3)
    release.set()
    producer.close()
    assert sorted(completed) == [1, 2]
    assert producer.stats.submitted == 2
    assert producer.stats.completed == 2


def test_background_cost_producer_reports_worker_failure() -> None:
    failed = Event()

    def fail(index: int) -> int:
        failed.set()
        raise ValueError(f"bad {index}")

    producer = BackgroundCostProducer(fail)
    producer.start()
    producer.submit(0)
    assert failed.wait(timeout=1)
    _wait_until(lambda: producer.stats.failed == 1)
    try:
        producer.check()
    except RuntimeError as error:
        assert "background cost producer failed" in str(error)
    else:
        raise AssertionError("producer failure was not reported")


def test_background_cost_producer_wait_snapshot_exposes_error() -> None:
    """The wait snapshot must carry the worker error for waiters to surface."""

    failed = Event()

    def fail(index: int) -> int:
        failed.set()
        raise TimeoutError(f"stuck {index}")

    producer = BackgroundCostProducer(fail)
    producer.start()
    producer.submit(0)
    assert failed.wait(timeout=1)
    _wait_until(lambda: producer.stats.failed == 1)
    snapshot = producer.wait_snapshot()
    assert snapshot["failed"] == 1
    assert "TimeoutError: stuck 0" in str(snapshot["error"])
    healthy = BackgroundCostProducer(lambda index: index)
    assert healthy.wait_snapshot()["error"] == ""


def test_background_cost_producer_rejects_submissions_once_close_starts() -> None:
    release = Event()
    started = Event()

    def compute(index: int) -> int:
        del index
        started.set()
        release.wait()
        return 0

    producer = BackgroundCostProducer(compute, max_queue_size=2)
    producer.start()
    assert producer.submit(1)
    assert started.wait(timeout=1)

    closer = Thread(target=producer.close)
    closer.start()
    _wait_until(lambda: not producer.submit(2))
    release.set()
    closer.join(timeout=1)
    assert not closer.is_alive()
    assert not producer.submit(3)


def test_background_cost_producer_wait_snapshot_tracks_inflight() -> None:
    release = Event()
    started = Event()

    def compute(index: int) -> int:
        started.set()
        release.wait(timeout=5.0)
        return index

    producer = BackgroundCostProducer(compute, max_queue_size=2)
    producer.start()
    assert producer.submit(7)
    assert started.wait(timeout=1)
    snap = producer.wait_snapshot(7)
    assert snap["alive"] is True
    assert snap["index_pending"] is True
    assert snap["in_cost_row_index"] == 7
    assert snap["inflight_count"] >= 1
    assert snap["submitted"] >= 1
    release.set()
    producer.close()
    done = producer.wait_snapshot(7)
    assert done["alive"] is False
    assert done["completed"] >= 1
    assert done["index_pending"] is False


def test_background_cost_producer_urgent_jumps_normal_backlog() -> None:
    order: list[int] = []
    release = Event()
    started = Event()

    def compute(index: int) -> int:
        if index == 1:
            started.set()
            release.wait(timeout=5.0)
        order.append(index)
        return index

    producer = BackgroundCostProducer(compute, max_queue_size=8)
    producer.start()
    assert producer.submit(1)
    assert started.wait(timeout=1)
    # Fill a normal backlog while the first item is in-flight.
    for index in (2, 3, 4, 5):
        assert producer.submit(index)
    assert producer.submit(99, urgent=True)
    release.set()
    producer.close()
    # 1 finishes first (already running), then urgent 99 before normals 2..5.
    assert order[0] == 1
    assert order[1] == 99
    assert order[2:] == [2, 3, 4, 5]


def test_background_cost_producer_batch_drains_with_compute_many() -> None:
    batches: list[list[int]] = []
    release = Event()
    started = Event()

    def compute(index: int) -> int:
        return index

    def compute_many(indexes: Sequence[int]) -> list[int]:
        batch = list(indexes)
        batches.append(batch)
        if not started.is_set():
            started.set()
            release.wait(timeout=5.0)
        return batch

    producer = BackgroundCostProducer(
        compute,
        compute_many=compute_many,
        batch_size=4,
        max_queue_size=8,
        num_workers=1,
    )
    producer.start()
    assert producer.submit(1)
    assert started.wait(timeout=1)
    for index in (2, 3, 4, 5):
        assert producer.submit(index)
    release.set()
    producer.close()
    assert producer.stats.completed == 5
    assert max(len(batch) for batch in batches) >= 4
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == [1, 2, 3, 4, 5]


def test_background_cost_producer_multi_worker_drains_in_parallel() -> None:
    started = {i: Event() for i in (1, 2, 3, 4)}
    release = Event()
    completed: list[int] = []

    def compute(index: int) -> int:
        started[index].set()
        release.wait(timeout=5.0)
        completed.append(index)
        return index

    producer = BackgroundCostProducer(compute, max_queue_size=8, num_workers=4)
    producer.start()
    for index in (1, 2, 3, 4):
        assert producer.submit(index)
    assert all(event.wait(timeout=1) for event in started.values())
    snap = producer.wait_snapshot(1)
    assert snap["alive"] is True
    assert snap["active_workers"] == 4
    release.set()
    producer.close()
    assert sorted(completed) == [1, 2, 3, 4]
    assert producer.stats.completed == 4


def test_background_cost_producer_urgent_accepted_when_normal_full() -> None:
    release = Event()
    started = Event()
    completed: list[int] = []

    def compute(index: int) -> int:
        if not started.is_set():
            started.set()
            release.wait(timeout=5.0)
        completed.append(index)
        return index

    producer = BackgroundCostProducer(compute, max_queue_size=1)
    producer.start()
    assert producer.submit(1)
    assert started.wait(timeout=1)
    assert producer.submit(2)
    assert not producer.submit(3)
    assert producer.submit(99, urgent=True)
    release.set()
    producer.close()
    assert 99 in completed
    assert producer.stats.dropped >= 1
