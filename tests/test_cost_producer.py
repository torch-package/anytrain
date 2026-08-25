from __future__ import annotations

from collections.abc import Callable
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
    _wait_until(lambda: producer.pending == 1)

    assert not producer.submit(2)
    release.set()
    closer.join(timeout=1)
    assert not closer.is_alive()
    assert not producer.submit(3)
