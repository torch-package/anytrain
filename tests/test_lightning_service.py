from __future__ import annotations

import signal
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from anytrain.lightning import ManagedServiceCallback

_BROADCAST_INPUT = object()


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


class _Service:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        start_error: Exception | None = None,
        check_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.start_error = start_error
        self.check_error = check_error
        self.close_error = close_error

    def start(self, *, owner: bool) -> None:
        self.events.append(("start", owner))
        if self.start_error is not None:
            raise self.start_error

    def check(self, *, owner: bool) -> None:
        self.events.append(("check", owner))
        if self.check_error is not None:
            raise self.check_error

    def close(self, *, owner: bool) -> None:
        self.events.append(("close", owner))
        if self.close_error is not None:
            raise self.close_error


class ManagedServiceCallbackTest(unittest.TestCase):
    def test_owner_lifecycle_checks_at_most_once_per_configured_step(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events)
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=1,
            global_step=0,
            strategy=_Strategy(events),
        )
        callback = ManagedServiceCallback(
            lambda _: service,
            check_every_n_steps=2,
            guard_sigterm=False,
        )

        callback.on_fit_start(cast(Any, trainer), cast(Any, object()))
        callback.on_train_batch_start(
            cast(Any, trainer), cast(Any, object()), None, 0
        )
        callback.on_train_batch_start(
            cast(Any, trainer), cast(Any, object()), None, 1
        )
        trainer.global_step = 1
        callback.on_train_batch_start(
            cast(Any, trainer), cast(Any, object()), None, 2
        )
        trainer.global_step = 2
        callback.on_train_batch_start(
            cast(Any, trainer), cast(Any, object()), None, 3
        )
        callback.on_train_end(cast(Any, trainer), cast(Any, object()))
        callback.on_fit_end(cast(Any, trainer), cast(Any, object()))

        self.assertEqual(
            [event for event in events if event[0] != "broadcast"],
            [
                ("start", True),
                ("check", True),
                ("check", True),
                ("close", True),
            ],
        )

    def test_start_failure_is_broadcast_and_partially_started_service_is_closed(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events, start_error=RuntimeError("cannot start"))
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=1,
            strategy=_Strategy(events),
        )
        callback = ManagedServiceCallback(
            lambda _: service,
            guard_sigterm=False,
            label="producer",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "producer start failed on the global owner: RuntimeError: cannot start",
        ):
            callback.on_fit_start(cast(Any, trainer), cast(Any, object()))

        self.assertEqual(
            [event for event in events if event[0] != "broadcast"],
            [("start", True), ("close", True)],
        )

    def test_remote_discovery_failure_closes_the_local_controller(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events)
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=2,
            strategy=_Strategy(
                events,
                broadcast_results=[
                    None,
                    ("RuntimeError", "peer discovery failed"),
                ],
            ),
        )
        callback = ManagedServiceCallback(
            lambda _: service,
            guard_sigterm=False,
            label="producer",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "producer discovery failed on rank 1: "
            "RuntimeError: peer discovery failed",
        ):
            callback.on_fit_start(cast(Any, trainer), cast(Any, object()))

        self.assertEqual(
            [event for event in events if event[0] != "broadcast"],
            [("close", True)],
        )

    def test_remote_health_failure_is_raised_on_non_owner(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events)
        trainer = SimpleNamespace(
            is_global_zero=False,
            global_rank=1,
            world_size=2,
            global_step=0,
            strategy=_Strategy(
                events,
                broadcast_results=[
                    None,
                    None,
                    True,
                    True,
                    None,
                    ("RuntimeError", "producer exited"),
                ],
            ),
        )
        callback = ManagedServiceCallback(
            lambda _: service,
            guard_sigterm=False,
            label="producer",
        )

        callback.on_fit_start(cast(Any, trainer), cast(Any, object()))
        with self.assertRaisesRegex(
            RuntimeError,
            "producer health check failed on the global owner: "
            "RuntimeError: producer exited",
        ):
            callback.on_train_batch_start(
                cast(Any, trainer), cast(Any, object()), None, 0
            )

        self.assertFalse(any(event[0] in {"start", "check", "close"} for event in events))

    def test_exception_cleanup_does_not_mask_the_training_failure(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events, close_error=RuntimeError("close failed"))
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=1,
            strategy=_Strategy(events),
        )
        callback = ManagedServiceCallback(
            lambda _: service,
            guard_sigterm=False,
        )
        callback.on_fit_start(cast(Any, trainer), cast(Any, object()))

        callback.on_exception(
            cast(Any, trainer),
            cast(Any, object()),
            RuntimeError("training failed"),
        )

        self.assertIn(("close", True), events)
        self.assertIsNone(callback._active)

    def test_normal_close_failure_is_broadcast(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events, close_error=RuntimeError("close failed"))
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=1,
            strategy=_Strategy(events),
        )
        callback = ManagedServiceCallback(
            lambda _: service,
            guard_sigterm=False,
            label="producer",
        )
        callback.on_fit_start(cast(Any, trainer), cast(Any, object()))

        with self.assertRaisesRegex(
            RuntimeError,
            "producer close failed on the global owner: RuntimeError: close failed",
        ):
            callback.on_train_end(cast(Any, trainer), cast(Any, object()))

    def test_sigterm_guard_delegates_only_the_first_delivery(self) -> None:
        events: list[tuple[object, ...]] = []
        service = _Service(events)
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_rank=0,
            world_size=1,
            strategy=_Strategy(events),
        )
        callback = ManagedServiceCallback(lambda _: service)
        delegate = Mock()
        callback.on_fit_start(cast(Any, trainer), cast(Any, object()))

        with (
            patch("anytrain.lightning.service.signal.getsignal", return_value=delegate),
            patch("anytrain.lightning.service.signal.signal") as register,
        ):
            callback.on_train_start(cast(Any, trainer), cast(Any, object()))

        guard = callback._sigterm_guard
        self.assertIsNotNone(guard)
        assert guard is not None
        guard(signal.SIGTERM, None)
        guard(signal.SIGTERM, None)

        register.assert_called_once_with(signal.SIGTERM, guard)
        delegate.assert_called_once_with(signal.SIGTERM, None)

    def test_constructor_rejects_invalid_cadence_and_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "check_every_n_steps must be positive"):
            ManagedServiceCallback(lambda _: None, check_every_n_steps=0)
        with self.assertRaisesRegex(TypeError, "check_every_n_steps must be an integer"):
            ManagedServiceCallback(lambda _: None, check_every_n_steps=cast(Any, True))
        with self.assertRaisesRegex(TypeError, "guard_sigterm must be a boolean"):
            ManagedServiceCallback(lambda _: None, guard_sigterm=cast(Any, 1))


if __name__ == "__main__":
    unittest.main()
