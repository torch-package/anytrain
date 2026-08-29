from __future__ import annotations

import unittest
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

from anytrain.lightning.schedule import (
    Constant,
    Cosine,
    Linear,
    LRSchedule,
    Phase,
    ScheduleRuntime,
    StepClock,
    UnitBatch,
    UnitClock,
    UnitScheduleCallback,
    UnitScheduleUpdate,
    padding_ratio,
    require_unit_name,
    units_per_second,
)


def SimpleNamespace(**kwargs: object) -> Any:
    return _SimpleNamespace(**kwargs)


class _Module:
    def __init__(self) -> None:
        self.log_dict_calls = []
        self.device = torch.device("cpu")

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((dict(values), dict(kwargs)))


class _Provider:
    def __init__(self, *batches: UnitBatch) -> None:
        self.batches = list(batches)

    def __call__(self, **kwargs):
        del kwargs
        return self.batches.pop(0)


def _module() -> Any:
    return _Module()


def _optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=lr)


class ScheduleTest(unittest.TestCase):
    def test_lightning_api_exports_schedule_not_legacy_throughput(self):
        import anytrain.lightning as lightning_api
        import anytrain.lightning.callback as callback_api
        import anytrain.lightning.perf as perf_api

        self.assertIn("schedule", lightning_api.__all__)
        for module in (lightning_api, callback_api):
            self.assertFalse(hasattr(module, "DataThroughputCallback"))
            self.assertFalse(hasattr(module, "DataUnits"))
            self.assertFalse(hasattr(module, "DataUnitsProvider"))
        self.assertNotIn("padding_ratio", perf_api.__all__)
        self.assertNotIn("units_realtime", perf_api.__all__)

    def test_progress_helpers_validate_units(self):
        self.assertEqual(
            padding_ratio(valid_units=75.0, padded_units=100.0),
            0.25,
        )
        self.assertEqual(
            units_per_second(valid_units=10.0, step_time=2.0),
            5.0,
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            padding_ratio(valid_units=2.0, padded_units=1.0)
        with self.assertRaisesRegex(ValueError, "unit must match"):
            require_unit_name("Audio-Seconds")

    def test_unit_batch_rejects_bool_and_tensor_counts(self):
        for value in (True, torch.tensor(1.0)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "host real number"):
                    UnitBatch(valid=cast(Any, value), unit="tokens")
                with self.assertRaisesRegex(TypeError, "host real number"):
                    UnitBatch(valid=0.0, padded=cast(Any, value), unit="tokens")

    def test_schedule_specs_own_strict_numeric_validation(self):
        provider = _Provider()
        for value in (True, "1", torch.tensor(1.0)):
            with (
                self.subTest(field="curve", value=value),
                self.assertRaisesRegex(TypeError, "host real number"),
            ):
                Constant(cast(Any, value))
            with (
                self.subTest(field="duration", value=value),
                self.assertRaisesRegex(TypeError, "host real number"),
            ):
                Phase("main", duration=cast(Any, value), lr=Constant())
            with (
                self.subTest(field="log_every_n_units", value=value),
                self.assertRaisesRegex(TypeError, "host real number"),
            ):
                UnitClock(
                    unit="tokens",
                    provider=provider,
                    log_every_n_units=cast(Any, value),
                )
            with (
                self.subTest(field="units_per_epoch", value=value),
                self.assertRaisesRegex(TypeError, "host real number"),
            ):
                UnitClock(
                    unit="tokens",
                    provider=provider,
                    units_per_epoch=cast(Any, value),
                )
            with (
                self.subTest(field="stop_at", value=value),
                self.assertRaisesRegex(TypeError, "host real number"),
            ):
                LRSchedule(
                    clock=UnitClock(unit="tokens", provider=provider),
                    phases=(Phase("main", duration=1.0, lr=Constant()),),
                    stop_at=cast(Any, value),
                )

        for value in (True, 1.0):
            with (
                self.subTest(field="measure_window_batches", value=value),
                self.assertRaisesRegex(TypeError, "must be an integer"),
            ):
                UnitClock(
                    unit="tokens",
                    provider=provider,
                    measure_window_batches=cast(Any, value),
                )

    def test_schedule_requires_unique_phase_names(self):
        phases = (
            Phase("main", duration=1.0, lr=Constant()),
            Phase("main", duration=2.0, lr=Constant()),
        )

        with self.assertRaisesRegex(ValueError, "phase names must be unique"):
            LRSchedule(clock=StepClock(), phases=phases)

    def test_step_clock_runtime_returns_lightning_scheduler(self):
        optimizer = _optimizer()
        schedule = LRSchedule(
            clock=StepClock(),
            phases=[
                Phase("warmup", duration=2, lr=Linear(0.0, 1.0)),
                Phase("stable", duration=3, lr=Constant(1.0)),
                Phase("decay", duration=5, lr=Cosine(1.0, 0.1)),
            ],
        )
        runtime = ScheduleRuntime(schedule)

        self.assertEqual(
            schedule.milestones(),
            (("warmup", 2.0), ("stable", 5.0), ("decay", 10.0)),
        )
        configured = runtime.configure_optimizers(optimizer)

        self.assertEqual(runtime.callbacks(), [])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        scheduler = configured["lr_scheduler"]["scheduler"]
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)

    def test_unit_clock_runtime_returns_optimizer_and_callback_logs_progress(self):
        optimizer = _optimizer()
        provider = _Provider(UnitBatch(valid=5.0, padded=8.0, unit="tokens"))
        schedule = LRSchedule(
            clock=UnitClock(
                unit="tokens",
                provider=provider,
                measure_window_batches=2,
                sync_cuda=False,
            ),
            phases=[Phase("warmup", duration=10.0, lr=Linear(0.0, 1.0))],
        )
        runtime = ScheduleRuntime(schedule)
        callback = runtime.callbacks()[0]
        trainer = SimpleNamespace(global_step=0, optimizers=[optimizer], lr_scheduler_configs=[])
        module = _module()

        self.assertIs(runtime.configure_optimizers(optimizer), optimizer)
        callback.on_train_start(trainer, module)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        with patch("time.perf_counter", side_effect=[10.0, 10.5]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        values, kwargs = module.log_dict_calls[-1]
        self.assertEqual(values["progress/tokens"], 5.0)
        self.assertEqual(values["progress/tokens_total"], 5.0)
        self.assertEqual(values["progress/tokens_padded"], 8.0)
        self.assertEqual(values["progress/tokens_padded_total"], 8.0)
        self.assertEqual(values["progress/tokens_per_second"], 10.0)
        self.assertEqual(values["progress/padding_ratio"], 0.375)
        self.assertEqual(values["progress/lr_ratio"], 0.5)
        self.assertEqual(
            kwargs,
            {"on_step": True, "on_epoch": False, "logger": True, "sync_dist": False},
        )

    def test_unit_clock_stop_at_and_state_round_trip(self):
        optimizer = _optimizer()
        provider = _Provider(UnitBatch(valid=6.0, unit="tokens"))
        schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=provider, sync_cuda=False),
            phases=[Phase("main", duration=5.0, lr=Constant(1.0))],
            stop_at=5.0,
        )
        callback = UnitScheduleCallback(schedule)
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            should_stop=False,
        )
        module = _module()

        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )

        self.assertTrue(trainer.should_stop)
        state = callback.state_dict()
        restored = UnitScheduleCallback(schedule)
        restored.load_state_dict(state)
        self.assertEqual(restored.total_valid, 6.0)
        self.assertEqual(state["stop_at"], 5.0)

    def test_schedule_accepts_only_explicit_stop_at_budget(self):
        with self.assertRaises(TypeError):
            cast(Any, LRSchedule)(
                clock=UnitClock(unit="tokens", provider=_Provider(), sync_cuda=False),
                phases=[Phase("main", duration=5.0, lr=Constant(1.0))],
                stop_at_end=True,
            )

    def test_unit_clock_budget_is_not_part_of_fingerprint(self):
        provider = _Provider()
        old_schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=provider, sync_cuda=False),
            phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            stop_at=100.0,
        )
        new_schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=provider, sync_cuda=False),
            phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            stop_at=300.0,
        )
        self.assertEqual(old_schedule.fingerprint(), new_schedule.fingerprint())

        source = UnitScheduleCallback(old_schedule)
        source.total_valid = 100.0
        restored = UnitScheduleCallback(new_schedule)
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored.total_valid, 100.0)

    def test_unit_clock_accepts_legacy_budgeted_fingerprint(self):
        schedule = LRSchedule(
            clock=UnitClock(
                unit="tokens",
                provider=_Provider(),
                units_per_epoch=2.0,
                sync_cuda=False,
            ),
            phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            stop_at=300.0,
        )
        legacy_fingerprint = repr(
            (
                "unit",
                "tokens",
                2.0,
                (("main", 10.0, "Constant", "Constant(value=1.0)"),),
                100.0,
                False,
            )
        )
        state = {
            "unit": "tokens",
            "total_valid": 100.0,
            "total_padded": 100.0,
            "base_lrs": None,
            "last_global_step": None,
            "last_lrs": None,
            "schedule": legacy_fingerprint,
        }
        restored = UnitScheduleCallback(schedule)
        restored.load_state_dict(state)
        self.assertEqual(restored.total_valid, 100.0)

    def test_unit_clock_resume_rejects_budget_reduction(self):
        old_schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=_Provider(), sync_cuda=False),
            phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            stop_at=100.0,
        )
        new_schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=_Provider(), sync_cuda=False),
            phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            stop_at=50.0,
        )
        state = UnitScheduleCallback(old_schedule).state_dict()
        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            UnitScheduleCallback(new_schedule).load_state_dict(state)

    def test_unit_clock_resume_marks_checkpoint_at_budget(self):
        schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=_Provider(), sync_cuda=False),
            phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            stop_at=5.0,
        )
        source = UnitScheduleCallback(schedule)
        source.total_valid = 6.0
        restored = UnitScheduleCallback(schedule)
        restored.load_state_dict(source.state_dict())
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[_optimizer()],
            lr_scheduler_configs=[],
            should_stop=False,
        )
        restored.on_train_start(trainer, _module())
        self.assertTrue(trainer.should_stop)

    def test_unit_clock_exposes_epoch_boundaries(self):
        optimizer = _optimizer()
        provider = _Provider(
            UnitBatch(valid=3.0, unit="tokens"),
            UnitBatch(valid=8.0, unit="tokens"),
        )
        runtime = ScheduleRuntime(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=provider,
                    units_per_epoch=5.0,
                    sync_cuda=False,
                ),
                phases=[Phase("main", duration=20.0, lr=Constant(1.0))],
            )
        )
        callback = runtime.unit_callback()
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            should_stop=False,
        )
        module = _module()

        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[1.0, 2.0, 3.0, 4.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )
            self.assertEqual(callback.completed_unit_epochs, 0)
            self.assertEqual(callback.unit_epochs_advanced, 0)
            self.assertFalse(callback.crossed_unit_epoch_interval(1))

            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
            trainer.global_step = 2
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=1,
            )

        self.assertIs(runtime.unit_callback(), callback)
        self.assertEqual(callback.completed_unit_epochs, 2)
        self.assertEqual(callback.unit_epochs_advanced, 2)
        self.assertTrue(callback.crossed_unit_epoch_interval(1))
        self.assertTrue(callback.crossed_unit_epoch_interval(2))
        values, _ = module.log_dict_calls[-1]
        self.assertEqual(values["progress/unit_epoch"], 2)
        self.assertAlmostEqual(values["progress/unit_epoch_progress"], 0.2)

    def test_unit_schedule_dispatches_read_only_updates_to_consumers(self):
        optimizer = _optimizer()
        runtime = ScheduleRuntime(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(UnitBatch(valid=5.0, unit="tokens")),
                    units_per_epoch=5.0,
                    sync_cuda=False,
                ),
                phases=[Phase("main", duration=20.0, lr=Constant(1.0))],
            )
        )
        callback = runtime.unit_callback()
        updates: list[UnitScheduleUpdate] = []

        class Consumer:
            def on_unit_schedule_update(self, trainer, pl_module, update):
                del trainer, pl_module
                updates.append(update)

        callback.register_unit_schedule_consumer(Consumer())
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            should_stop=False,
            fit_loop=SimpleNamespace(
                epoch_loop=SimpleNamespace(total_batch_idx=0),
            ),
        )
        module = _module()
        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer, module, outputs=None, batch=None, batch_idx=0
            )

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].completed_unit_epochs, 1)
        self.assertEqual(updates[0].previous_unit_epochs, 0)
        self.assertEqual(updates[0].advanced_unit_epochs, 1)
        self.assertTrue(updates[0].crossed_unit_epoch_interval(1))

    def test_unit_clock_state_rejects_coerced_or_malformed_lrs(self):
        schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=_Provider(), sync_cuda=False),
            phases=[Phase("main", duration=5.0, lr=Constant(1.0))],
        )
        callback = UnitScheduleCallback(schedule)
        valid_state = {
            "unit": "tokens",
            "total_valid": 0.0,
            "total_padded": 0.0,
            "base_lrs": [[0.1]],
            "last_global_step": 0,
            "last_lrs": [[0.1]],
            "schedule": schedule.fingerprint(),
        }

        for field in ("base_lrs", "last_lrs"):
            for value in ([], [0.1], [[]], [["0.1"]], [[True]], [[float("nan")]]):
                with self.subTest(field=field, value=value):
                    state = dict(valid_state)
                    state[field] = value
                    with self.assertRaises((TypeError, ValueError)):
                        callback.load_state_dict(state)

        for field in ("total_valid", "total_padded"):
            state = dict(valid_state)
            state[field] = "0.0"
            with self.subTest(field=field), self.assertRaises(TypeError):
                callback.load_state_dict(state)

    def test_unit_clock_rejects_lightning_scheduler_conflict(self):
        optimizer = _optimizer()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(UnitBatch(valid=1.0, unit="tokens")),
                    sync_cuda=False,
                ),
                phases=[Phase("main", duration=1.0, lr=Constant(1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[object()],
        )

        with self.assertRaisesRegex(RuntimeError, "Lightning lr_schedulers"):
            callback.on_train_start(trainer, _module())

    def test_unit_clock_rejects_external_lr_changes(self):
        optimizer = _optimizer()
        provider = _Provider(
            UnitBatch(valid=5.0, unit="tokens"),
            UnitBatch(valid=5.0, unit="tokens"),
        )
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(unit="tokens", provider=provider, sync_cuda=False),
                phases=[Phase("warmup", duration=10.0, lr=Linear(0.0, 1.0))],
            )
        )
        trainer = SimpleNamespace(global_step=0, optimizers=[optimizer], lr_scheduler_configs=[])
        module = _module()
        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[1.0, 2.0, 3.0, 4.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
            optimizer.param_groups[0]["lr"] = 0.9
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
            trainer.global_step = 2
            with self.assertRaisesRegex(RuntimeError, "modified outside"):
                callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=1)

    def test_unit_clock_reduces_global_units(self):
        optimizer = _optimizer()

        class Strategy:
            def all_gather(self, tensor):
                return torch.stack((tensor, tensor))

        provider = _Provider(UnitBatch(valid=5.0, padded=8.0, unit="tokens"))
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(unit="tokens", provider=provider, sync_cuda=False),
                phases=[Phase("warmup", duration=20.0, lr=Linear(0.0, 1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            world_size=2,
            strategy=Strategy(),
        )
        module = _module()
        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

        self.assertEqual(callback.total_valid, 10.0)
        self.assertEqual(callback.total_padded, 16.0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)

    def test_unit_clock_all_ranks_reject_different_accumulation_windows(self):
        class Strategy:
            def all_gather(self, tensor):
                # Three ranks contributed 1, 2, and 3 batches.
                return tensor.new_tensor(
                    [
                        [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                        [2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 2.0],
                        [3.0, 0.0, 0.0, 3.0, 9.0, 0.0, 0.0, 3.0],
                    ]
                )

        for local_batches in (1, 2, 3):
            optimizer = _optimizer()
            callback = UnitScheduleCallback(
                LRSchedule(
                    clock=UnitClock(
                        unit="tokens",
                        provider=_Provider(
                            *(UnitBatch(valid=1.0, unit="tokens") for _ in range(local_batches))
                        ),
                    ),
                    phases=[Phase("main", duration=20.0, lr=Constant(1.0))],
                )
            )
            trainer = SimpleNamespace(
                global_step=0,
                optimizers=[optimizer],
                lr_scheduler_configs=[],
                world_size=3,
                strategy=Strategy(),
            )
            module = _module()
            callback.on_train_start(trainer, module)

            with (
                patch("time.perf_counter", side_effect=[1.0, 2.0]),
                self.assertRaisesRegex(RuntimeError, "accumulation windows"),
            ):
                for batch_idx in range(local_batches):
                    callback.on_train_batch_start(
                        trainer,
                        module,
                        batch=None,
                        batch_idx=batch_idx,
                    )
                    if batch_idx == local_batches - 1:
                        trainer.global_step = 1
                    callback.on_train_batch_end(
                        trainer,
                        module,
                        outputs=None,
                        batch=None,
                        batch_idx=batch_idx,
                    )

            self.assertEqual(callback.total_valid, 0.0)

    def test_unit_clock_all_ranks_reject_different_padded_batch_counts(self):
        class Strategy:
            def all_gather(self, tensor):
                # Every rank contributed 2 batches, with padded counts 0, 1, and 2.
                return tensor.new_tensor(
                    [
                        [2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0],
                        [2.0, 1.0, 1.0, 2.0, 4.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 4.0, 1.0],
                    ]
                )

        for local_padded_batches in (0, 1, 2):
            optimizer = _optimizer()
            callback = UnitScheduleCallback(
                LRSchedule(
                    clock=UnitClock(
                        unit="tokens",
                        provider=_Provider(
                            *(
                                UnitBatch(
                                    valid=1.0,
                                    padded=(1.0 if batch_idx < local_padded_batches else None),
                                    unit="tokens",
                                )
                                for batch_idx in range(2)
                            )
                        ),
                    ),
                    phases=[Phase("main", duration=20.0, lr=Constant(1.0))],
                )
            )
            trainer = SimpleNamespace(
                global_step=0,
                optimizers=[optimizer],
                lr_scheduler_configs=[],
                world_size=3,
                strategy=Strategy(),
            )
            module = _module()
            callback.on_train_start(trainer, module)

            with (
                patch("time.perf_counter", side_effect=[1.0, 2.0]),
                self.assertRaisesRegex(ValueError, "all ranks"),
            ):
                for batch_idx in range(2):
                    callback.on_train_batch_start(
                        trainer,
                        module,
                        batch=None,
                        batch_idx=batch_idx,
                    )
                    if batch_idx == 1:
                        trainer.global_step = 1
                    callback.on_train_batch_end(
                        trainer,
                        module,
                        outputs=None,
                        batch=None,
                        batch_idx=batch_idx,
                    )

            self.assertEqual(callback.total_valid, 0.0)

    def test_unit_clock_defers_sync_and_progress_until_optimizer_step(self):
        optimizer = _optimizer()
        provider = _Provider(
            UnitBatch(valid=2.0, padded=3.0, unit="tokens"),
            UnitBatch(valid=3.0, padded=4.0, unit="tokens"),
        )

        class Strategy:
            def __init__(self) -> None:
                self.all_gather_calls = 0

            def all_gather(self, tensor):
                self.all_gather_calls += 1
                return torch.stack((tensor, tensor))

        strategy = Strategy()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=provider,
                    sync_cuda=False,
                ),
                phases=[Phase("warmup", duration=20.0, lr=Linear(0.0, 1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            world_size=2,
            strategy=strategy,
        )
        module = _module()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize") as synchronize,
            patch("time.perf_counter", side_effect=[1.0, 3.0]),
        ):
            callback.on_train_start(trainer, module)
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )

            self.assertEqual(strategy.all_gather_calls, 0)
            self.assertEqual(callback.total_valid, 0.0)
            self.assertEqual(module.log_dict_calls, [])

            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=1,
            )

        self.assertFalse(callback.clock.sync_cuda)
        synchronize.assert_not_called()
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(callback.total_valid, 10.0)
        self.assertEqual(callback.total_padded, 14.0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        values, _ = module.log_dict_calls[-1]
        self.assertEqual(values["progress/tokens"], 10.0)
        self.assertEqual(values["progress/tokens_per_second"], 5.0)
        self.assertEqual(values["progress/padding_ratio"], 2.0 / 7.0)

    def test_unit_clock_ddp_throughput_uses_each_updates_slowest_rank(self):
        optimizer = _optimizer()

        class Strategy:
            def __init__(self) -> None:
                self.payloads = [
                    [
                        [2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                        [3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0],
                    ],
                    [
                        [4.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0],
                        [6.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                ]
                self.all_gather_calls = 0

            def all_gather(self, tensor):
                self.all_gather_calls += 1
                return tensor.new_tensor(self.payloads.pop(0))

        strategy = Strategy()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(
                        UnitBatch(valid=2.0, unit="tokens"),
                        UnitBatch(valid=4.0, unit="tokens"),
                    ),
                    sync_cuda=False,
                ),
                phases=[Phase("main", duration=20.0, lr=Constant(1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            world_size=2,
            strategy=strategy,
        )
        module = _module()
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[0.0, 1.0, 10.0, 15.0]):
            for batch_idx in range(2):
                callback.on_train_batch_start(
                    trainer,
                    module,
                    batch=None,
                    batch_idx=batch_idx,
                )
                trainer.global_step += 1
                callback.on_train_batch_end(
                    trainer,
                    module,
                    outputs=None,
                    batch=None,
                    batch_idx=batch_idx,
                )

        self.assertEqual(strategy.all_gather_calls, 2)
        first, second = (call[0] for call in module.log_dict_calls)
        self.assertEqual(first["progress/tokens_per_second"], 5.0 / 4.0)
        self.assertEqual(second["progress/tokens_per_second"], 2.0)
        self.assertAlmostEqual(second["progress/tokens_per_second_window"], 15.0 / 9.0)

    def test_unit_clock_default_cuda_timing_syncs_once_per_optimizer_step(self):
        optimizer = _optimizer()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(
                        UnitBatch(valid=2.0, unit="tokens"),
                        UnitBatch(valid=3.0, unit="tokens"),
                    ),
                ),
                phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
        )
        module = _module()
        module.device = torch.device("cuda", 3)

        with (
            patch("torch.cuda.synchronize") as synchronize,
            patch("time.perf_counter", side_effect=[1.0, 2.0]),
        ):
            callback.on_train_start(trainer, module)
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=1,
            )

        self.assertTrue(callback.clock.sync_cuda)
        self.assertEqual(synchronize.call_count, 2)
        synchronize.assert_called_with(module.device)

    def test_unit_clock_default_cuda_timing_does_not_sync_cpu_module(self):
        optimizer = _optimizer()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(UnitBatch(valid=1.0, unit="tokens")),
                ),
                phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
        )
        module = _module()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize") as synchronize,
            patch("time.perf_counter", side_effect=[1.0, 2.0]),
        ):
            callback.on_train_start(trainer, module)
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )

        synchronize.assert_not_called()

    def test_unit_clock_measure_window_remains_bounded_by_microbatches(self):
        optimizer = _optimizer()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(
                        UnitBatch(valid=2.0, unit="tokens"),
                        UnitBatch(valid=2.0, unit="tokens"),
                        UnitBatch(valid=3.0, unit="tokens"),
                        UnitBatch(valid=3.0, unit="tokens"),
                    ),
                    measure_window_batches=2,
                ),
                phases=[Phase("main", duration=20.0, lr=Constant(1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
        )
        module = _module()
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[0.0, 2.0, 10.0, 11.0]):
            for batch_idx in range(4):
                callback.on_train_batch_start(
                    trainer,
                    module,
                    batch=None,
                    batch_idx=batch_idx,
                )
                if batch_idx in (1, 3):
                    trainer.global_step += 1
                callback.on_train_batch_end(
                    trainer,
                    module,
                    outputs=None,
                    batch=None,
                    batch_idx=batch_idx,
                )

        first, second = (call[0] for call in module.log_dict_calls)
        self.assertEqual(first["progress/tokens_per_second_window"], 2.0)
        self.assertEqual(second["progress/tokens_per_second_window"], 6.0)

    def test_unit_clock_rejects_uncommitted_units_at_epoch_end(self):
        optimizer = _optimizer()
        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="tokens",
                    provider=_Provider(UnitBatch(valid=2.0, unit="tokens")),
                ),
                phases=[Phase("main", duration=10.0, lr=Constant(1.0))],
            )
        )
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
        )
        module = _module()
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", return_value=1.0):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )

        with self.assertRaisesRegex(RuntimeError, "optimizer-step boundary"):
            callback.on_train_epoch_end(trainer, module)

    def test_unit_clock_handles_partial_final_accumulation_in_trainer(self):
        from lightning import pytorch as pl
        from torch.utils.data import DataLoader, TensorDataset

        class TrainingModule(pl.LightningModule):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def training_step(self, batch, batch_idx):
                del batch_idx
                (values,) = batch
                return self.linear(values).square().mean()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        callback = UnitScheduleCallback(
            LRSchedule(
                clock=UnitClock(
                    unit="samples",
                    provider=_Provider(
                        UnitBatch(valid=1.0, unit="samples"),
                        UnitBatch(valid=1.0, unit="samples"),
                        UnitBatch(valid=1.0, unit="samples"),
                    ),
                    sync_cuda=False,
                ),
                phases=[Phase("main", duration=3.0, lr=Constant(1.0))],
            )
        )
        trainer = pl.Trainer(
            max_epochs=1,
            accumulate_grad_batches=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(
            TrainingModule(),
            train_dataloaders=DataLoader(
                TensorDataset(torch.ones(3, 1)),
                batch_size=1,
            ),
        )

        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(callback.total_valid, 3.0)


if __name__ == "__main__":
    unittest.main()
