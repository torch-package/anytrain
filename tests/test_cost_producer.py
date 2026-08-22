from __future__ import annotations

from threading import Event
from time import sleep

from anytrain.cost import BackgroundCostProducer


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
    while not started.is_set():
        sleep(0.001)
    assert producer.submit(2)
    assert not producer.submit(3)
    release.set()
    producer.close()
    assert sorted(completed) == [1, 2]
    assert producer.stats.submitted == 2
    assert producer.stats.completed == 2


def test_background_cost_producer_reports_worker_failure() -> None:
    def fail(index: int) -> int:
        raise ValueError(f"bad {index}")

    producer = BackgroundCostProducer(fail)
    producer.start()
    producer.submit(0)
    while producer.stats.failed == 0:
        sleep(0.001)
    try:
        producer.check()
    except RuntimeError as error:
        assert "background cost producer failed" in str(error)
    else:
        raise AssertionError("producer failure was not reported")
