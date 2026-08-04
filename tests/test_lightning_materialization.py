from __future__ import annotations

import time
import unittest
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast

from anytrain.lightning import (
    BackgroundMaterialization,
    MaterializationCallback,
    MaterializationPhase,
)

_BROADCAST_INPUT = object()


def _touch(path: Path) -> None:
    path.touch()


class _Producer:
    def __init__(self, marker: Path, value: str = "ready") -> None:
        self.marker = marker
        self.value = value
        self.loads = 0

    def __call__(self) -> None:
        self.marker.touch()

    def load(self) -> str:
        self.loads += 1
        return self.value


class _NestedProcessProducer(_Producer):
    def __call__(self) -> None:
        process = get_context("spawn").Process(target=_touch, args=(self.marker,))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"nested worker exited with code {process.exitcode}")


class _LargeFailureProducer:
    def __call__(self) -> None:
        raise RuntimeError("worker failure " + "x" * 1_000_000)

    def load(self) -> str:
        raise AssertionError("failed materialization must not load")


class _BlockingProducer(_Producer):
    def __call__(self) -> None:
        self.marker.touch()
        while True:
            time.sleep(1.0)


class BackgroundMaterializationTest(unittest.TestCase):
    def test_load_ready_requires_synchronized_publication(self) -> None:
        with TemporaryDirectory() as directory:
            job = BackgroundMaterialization(
                "request-v1",
                _Producer(Path(directory) / "published"),
            )

            with self.assertRaisesRegex(RuntimeError, "synchronized publication"):
                job.load_ready()

    def test_non_owner_becomes_ready_only_after_finish(self) -> None:
        with TemporaryDirectory() as directory:
            producer = _Producer(Path(directory) / "published")
            job = BackgroundMaterialization("request-v1", producer)

            job.start(owner=False)
            with self.assertRaisesRegex(RuntimeError, "synchronized publication"):
                job.load_ready()
            job.finish(owner=False)

            self.assertEqual(job.load_ready(), "ready")
            self.assertEqual(producer.loads, 1)

    def test_finishes_and_loads_in_the_training_process(self) -> None:
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "published"
            producer = _Producer(marker)
            job = BackgroundMaterialization("request-v1", producer)

            job.start(owner=True)
            self.assertIs(job.phase, MaterializationPhase.MATERIALIZING)
            self.assertTrue(job._process is not None and not job._process.daemon)
            job.finish(owner=True)
            value = job.load_ready()
            job.close()

            self.assertTrue(marker.is_file())
        self.assertEqual(value, "ready")
        self.assertEqual(producer.loads, 1)
        self.assertIs(job.phase, MaterializationPhase.READY)

    def test_default_non_daemon_worker_can_spawn_nested_workers(self) -> None:
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "nested"
            job = BackgroundMaterialization("nested-v1", _NestedProcessProducer(marker))

            job.start(owner=True)
            job.finish(owner=True)
            job.close()

            self.assertTrue(marker.is_file())

    def test_large_worker_error_is_drained_and_bounded(self) -> None:
        job = BackgroundMaterialization("failure-v1", _LargeFailureProducer())

        started = time.monotonic()
        job.start(owner=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "materialization worker failed: RuntimeError: worker failure",
        ) as caught:
            job.finish(owner=True)
        job.close()

        self.assertLess(time.monotonic() - started, 10.0)
        self.assertLess(len(str(caught.exception)), 60_000)
        self.assertIs(job.phase, MaterializationPhase.FAILED)

    def test_close_terminates_a_running_worker_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            job = BackgroundMaterialization("blocking-v1", _BlockingProducer(marker))
            job.start(owner=True)
            deadline = time.monotonic() + 10.0
            while not marker.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.is_file())

            started = time.monotonic()
            job.close()
            job.close()

        self.assertLess(time.monotonic() - started, 6.0)
        self.assertIs(job.phase, MaterializationPhase.FAILED)


class _Controller:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        pending: bool = True,
        refresh_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.pending = pending
        self.refresh_error = refresh_error

    def start(self, *, owner: bool) -> None:
        self.events.append(("start", owner))

    def finish(self, *, owner: bool) -> None:
        self.events.append(("finish", owner))

    def refresh(self) -> None:
        self.events.append(("refresh",))
        if self.refresh_error is not None:
            raise self.refresh_error

    def close(self) -> None:
        self.events.append(("close",))


class _Strategy:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        broadcast_result: object = _BROADCAST_INPUT,
        broadcast_results: list[object] | None = None,
    ) -> None:
        self.events = events
        self.broadcast_result = broadcast_result
        self.broadcast_results = broadcast_results

    def broadcast(self, value: object, src: int = 0) -> object:
        self.events.append(("broadcast", value, src))
        if self.broadcast_results is not None:
            return self.broadcast_results.pop(0)
        if self.broadcast_result is _BROADCAST_INPUT:
            return value
        return self.broadcast_result

    def barrier(self, name: str | None = None) -> None:
        self.events.append(("barrier", name))


class MaterializationCallbackTest(unittest.TestCase):
    def test_discovery_failure_closes_the_local_controller(self) -> None:
        events: list[tuple[object, ...]] = []

        class Controller:
            @property
            def pending(self) -> bool:
                raise RuntimeError("pending lookup failed")

            def start(self, *, owner: bool) -> None:
                raise AssertionError("failed discovery must not start")

            def finish(self, *, owner: bool) -> None:
                raise AssertionError("failed discovery must not finish")

            def refresh(self) -> None:
                raise AssertionError("failed discovery must not refresh")

            def close(self) -> None:
                events.append(("close",))

        trainer = SimpleNamespace(
            is_global_zero=True,
            strategy=_Strategy(events),
        )

        with self.assertRaisesRegex(RuntimeError, "pending lookup failed"):
            MaterializationCallback(lambda _: Controller()).on_train_start(
                cast(Any, trainer),
                cast(Any, object()),
            )

        self.assertEqual(events, [("close",)])

    def test_owner_finishes_then_barriers_and_refreshes(self) -> None:
        events: list[tuple[object, ...]] = []
        controller = _Controller(events)
        trainer = SimpleNamespace(
            is_global_zero=True,
            strategy=_Strategy(events),
        )
        callback = MaterializationCallback(
            lambda _: controller,
            barrier_name="assets_finished",
        )

        callback.on_train_start(cast(Any, trainer), cast(Any, object()))
        callback.on_train_epoch_end(cast(Any, trainer), cast(Any, object()))

        self.assertEqual(
            events,
            [
                ("start", True),
                ("finish", True),
                ("broadcast", None, 0),
                ("barrier", "assets_finished"),
                ("finish", False),
                ("refresh",),
            ],
        )

    def test_inconsistent_active_state_fails_before_start(self) -> None:
        events: list[tuple[object, ...]] = []
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=2,
            strategy=_Strategy(
                events,
                broadcast_results=[None, None, True, False],
            ),
        )
        callback = MaterializationCallback(lambda _: _Controller(events))

        with self.assertRaisesRegex(RuntimeError, "active state differs across ranks"):
            callback.on_train_start(cast(Any, trainer), cast(Any, object()))

        self.assertNotIn(("start", True), events)

    def test_epoch_end_uses_train_start_controller_without_rechecking_pending(self) -> None:
        events: list[tuple[object, ...]] = []
        controller = _Controller(events)
        calls = 0

        def provider(_: object) -> _Controller:
            nonlocal calls
            calls += 1
            return controller

        trainer = SimpleNamespace(
            is_global_zero=True,
            strategy=_Strategy(events),
        )
        callback = MaterializationCallback(cast(Any, provider))

        callback.on_train_start(cast(Any, trainer), cast(Any, object()))
        controller.pending = False
        callback.on_train_epoch_end(cast(Any, trainer), cast(Any, object()))

        self.assertEqual(calls, 1)
        self.assertIn(("refresh",), events)

    def test_start_failure_on_another_rank_closes_local_worker(self) -> None:
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            job = BackgroundMaterialization(
                "blocking-v1",
                _BlockingProducer(marker),
            )

            class Controller:
                pending = True

                def start(self, *, owner: bool) -> None:
                    job.start(owner=owner)

                def finish(self, *, owner: bool) -> None:
                    job.finish(owner=owner)

                def refresh(self) -> None:
                    raise AssertionError("failed start must not refresh")

                def close(self) -> None:
                    job.close()

            trainer = SimpleNamespace(
                is_global_zero=True,
                global_rank=0,
                world_size=2,
                strategy=_Strategy(
                    [],
                    broadcast_results=[
                        None,
                        None,
                        True,
                        True,
                        None,
                        ("RuntimeError", "peer start failed"),
                    ],
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "peer start failed"):
                MaterializationCallback(lambda _: Controller()).on_train_start(
                    cast(Any, trainer),
                    cast(Any, object()),
                )

            self.assertIsNone(job._process)
            self.assertIs(job.phase, MaterializationPhase.FAILED)


if __name__ == "__main__":
    unittest.main()
