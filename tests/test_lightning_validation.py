from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from anytrain.lightning import validation


class _UnequalCountModule(LightningModule):
    def validation_step(self, batch, batch_idx):
        del batch_idx
        values, weights = batch
        validation.log(self, {"score": validation.Metric(values, weights)})


class ValidationTest(unittest.TestCase):
    def test_metric_reduces_values_by_integer_counts(self):
        metric = validation.Metric(
            values=torch.tensor([1.0, 4.0]),
            weights=torch.tensor([1.0, 3.0]),
        )

        value, count = metric.reduced()

        torch.testing.assert_close(value, torch.tensor(3.25))
        self.assertEqual(count, 4)

    def test_metric_accepts_boolean_observation_counts(self):
        metric = validation.Metric(
            values=torch.tensor([1.0, 4.0]),
            weights=torch.tensor([True, False]),
        )

        value, count = metric.reduced()

        torch.testing.assert_close(value, torch.tensor(1.0))
        self.assertEqual(count, 1)

    def test_metric_ignores_nonfinite_values_with_zero_count(self):
        metric = validation.Metric(
            values=torch.tensor([float("nan"), float("inf"), 3.0]),
            weights=torch.tensor([0, 0, 2]),
        )

        value, count = metric.reduced()

        torch.testing.assert_close(value, torch.tensor(3.0))
        self.assertEqual(count, 2)

    def test_metric_accumulates_low_precision_values_in_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                metric = validation.Metric(
                    values=torch.tensor([1.0, 2.0], dtype=dtype),
                    weights=torch.tensor([100_000, 100_000]),
                )

                value, count = metric.reduced()

                torch.testing.assert_close(value, torch.tensor(1.5))
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(count, 200_000)

    def test_metric_rejects_misaligned_or_non_count_weights(self):
        cases = (
            (
                validation.Metric(torch.ones(2), torch.ones(1)),
                "same shape",
            ),
            (
                validation.Metric(torch.ones(2), torch.tensor([0.5, 1.5])),
                "integer counts",
            ),
            (
                validation.Metric(torch.ones(2), torch.tensor([-1, 2])),
                "non-negative",
            ),
            (
                validation.Metric(torch.ones(2), torch.zeros(2)),
                "positive total",
            ),
        )

        for metric, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                metric.reduced()

    def test_log_uses_validation_epoch_and_distributed_defaults(self):
        module = SimpleNamespace(log=Mock())
        metric = validation.Metric(torch.tensor([1.0, 3.0]), torch.tensor([1, 3]))

        validation.log(module, {"score": metric})

        module.log.assert_called_once()
        args = module.log.call_args.args
        kwargs = module.log.call_args.kwargs
        self.assertEqual(args[0], "val/score")
        torch.testing.assert_close(args[1], torch.tensor(2.5))
        self.assertEqual(
            kwargs,
            {
                "on_step": False,
                "on_epoch": True,
                "sync_dist": True,
                "batch_size": 4,
            },
        )

    def test_log_resolves_shared_counts_once(self):
        module = SimpleNamespace(log=Mock())
        weights = torch.tensor([1, 3])
        metrics = {
            "loss": validation.Metric(torch.tensor([1.0, 3.0]), weights),
            "accuracy": validation.Metric(torch.tensor([0.0, 1.0]), weights),
        }
        count = validation.Metric.count

        with patch.object(
            validation.Metric,
            "count",
            autospec=True,
            side_effect=count,
        ) as resolve:
            validation.log(module, metrics)

        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(module.log.call_count, 2)

    def test_lightning_aggregates_unequal_validation_counts(self):
        class Module(LightningModule):
            def validation_step(self, batch, batch_idx):
                del batch_idx
                values, weights = batch
                validation.log(self, {"score": validation.Metric(values, weights)})

        history = validation.History()
        trainer = Trainer(
            accelerator="cpu",
            callbacks=[history],
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
        )
        data = TensorDataset(
            torch.tensor([1.0, 3.0, 10.0]),
            torch.tensor([1, 1, 8]),
        )

        trainer.validate(Module(), dataloaders=DataLoader(data, batch_size=2))

        report = history.report()
        self.assertEqual(report["sanity"], [])
        self.assertEqual(len(report["interval"]), 1)
        self.assertAlmostEqual(
            report["interval"][0]["metrics"]["val/score"],
            8.4,
            places=6,
        )

    def test_lightning_aggregates_count_not_divisible_by_world_size(self):
        trainer = Trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_fork",
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
        )
        data = TensorDataset(
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1, 2, 2, 2]),
        )

        reports = trainer.validate(
            _UnequalCountModule(),
            dataloaders=DataLoader(data, batch_size=2),
        )

        self.assertEqual(len(reports), 1)
        self.assertAlmostEqual(reports[0]["val/score"], 1 / 7, places=7)

    def test_history_separates_sanity_and_interval_and_restores_state(self):
        history = validation.History(prefix="dev")
        trainer = SimpleNamespace(
            callback_metrics={"dev/loss": torch.tensor(2.0), "train/loss": torch.tensor(3.0)},
            global_step=0,
            sanity_checking=True,
        )

        history.on_validation_end(trainer, SimpleNamespace())
        trainer.callback_metrics = {"dev/loss": torch.tensor(1.5)}
        trainer.global_step = 100
        trainer.sanity_checking = False
        history.on_validation_end(trainer, SimpleNamespace())

        restored = validation.History(prefix="dev")
        restored.load_state_dict(history.state_dict())
        self.assertEqual(restored.report(), history.report())

        trainer.callback_metrics = {"dev/loss": torch.tensor(1.25)}
        restored.on_validation_end(trainer, SimpleNamespace())
        self.assertEqual(len(restored.interval), 1)
        self.assertEqual(restored.interval[0]["metrics"], {"dev/loss": 1.25})

    def test_history_can_ignore_validation_without_matching_metrics(self):
        history = validation.History(strict=False)
        trainer = SimpleNamespace(
            callback_metrics={"train/loss": torch.tensor(2.0)},
            global_step=0,
            sanity_checking=False,
        )

        history.on_validation_end(trainer, SimpleNamespace())

        self.assertEqual(history.report(), {"sanity": [], "interval": []})

    def test_history_rejects_non_scalar_metrics(self):
        history = validation.History()
        trainer = SimpleNamespace(
            callback_metrics={"val/loss": torch.tensor([1.0, 2.0])},
            global_step=0,
            sanity_checking=False,
        )

        with self.assertRaisesRegex(ValueError, "val/loss.*scalar"):
            history.on_validation_end(trainer, SimpleNamespace())

    def test_history_rejects_invalid_configuration_and_state(self):
        with self.assertRaisesRegex(TypeError, "strict mode must be boolean"):
            validation.History(strict=1)

        history = validation.History()
        state = history.state_dict()
        state["prefix"] = "dev/"
        with self.assertRaisesRegex(ValueError, "prefix does not match"):
            history.load_state_dict(state)

    def test_history_state_load_is_atomic_and_requires_its_prefix(self):
        history = validation.History()
        trainer = SimpleNamespace(
            callback_metrics={"val/loss": torch.tensor(1.0)},
            global_step=1,
            sanity_checking=True,
        )
        history.on_validation_end(trainer, SimpleNamespace())
        original = history.report()
        state = history.state_dict()
        state["sanity"] = [{"step": 2, "metrics": {"val/loss": 2.0}}]
        state["interval"] = [{"step": 3, "metrics": {"train/loss": 3.0}}]

        with self.assertRaisesRegex(ValueError, "must match prefix"):
            history.load_state_dict(state)

        self.assertEqual(history.report(), original)

    def test_history_state_rejects_duplicate_steps(self):
        history = validation.History()
        state = history.state_dict()
        state["interval"] = [
            {"step": 2, "metrics": {"val/loss": 2.0}},
            {"step": 2, "metrics": {"val/loss": 1.0}},
        ]

        with self.assertRaisesRegex(ValueError, "strictly step ordered"):
            history.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
