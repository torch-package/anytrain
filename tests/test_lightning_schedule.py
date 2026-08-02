from __future__ import annotations

import unittest
from types import SimpleNamespace
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
    padding_ratio,
    require_unit_name,
    units_per_second,
)


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
        module = _Module()

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

    def test_unit_clock_stop_at_end_and_state_round_trip(self):
        optimizer = _optimizer()
        provider = _Provider(UnitBatch(valid=6.0, unit="tokens"))
        schedule = LRSchedule(
            clock=UnitClock(unit="tokens", provider=provider, sync_cuda=False),
            phases=[Phase("main", duration=5.0, lr=Constant(1.0))],
            stop_at_end=True,
        )
        callback = UnitScheduleCallback(schedule)
        trainer = SimpleNamespace(
            global_step=0,
            optimizers=[optimizer],
            lr_scheduler_configs=[],
            should_stop=False,
        )
        module = _Module()

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
            callback.on_train_start(trainer, _Module())

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
        module = _Module()
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
        case = self

        class Strategy:
            def reduce(self, tensor, *, reduce_op):
                case.assertEqual(reduce_op, "sum")
                return tensor * 2

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
        module = _Module()
        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

        self.assertEqual(callback.total_valid, 10.0)
        self.assertEqual(callback.total_padded, 16.0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)


if __name__ == "__main__":
    unittest.main()
