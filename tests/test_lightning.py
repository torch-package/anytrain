import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


class _LogModule:
    def __init__(self):
        self.log_dict_calls = []
        self.log_calls = []

    def log(self, name, value):
        self.log_calls.append((name, float(value)))

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class _CheckpointStrategy:
    def __init__(self):
        self.barrier_count = 0

    def reduce_boolean_decision(self, decision, *, all=False):
        return decision

    def barrier(self, *args, **kwargs):
        self.barrier_count += 1


class _CheckpointTrainer:
    def __init__(self, payload: bytes = b"checkpoint"):
        self.global_step = 7
        self.is_global_zero = True
        self.loggers = []
        self.strategy = _CheckpointStrategy()
        self.saved_paths = []
        self.payload = payload

    def save_checkpoint(self, filepath, weights_only):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)
        self.saved_paths.append((path, weights_only))


def _loss_summary_ddp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    import json
    from datetime import timedelta
    from types import SimpleNamespace

    import torch
    from lightning.pytorch.strategies import DDPStrategy

    from anytrain.lightning import LossSummaryCallback
    from anytrain.loss import LossItem

    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = SimpleNamespace(world_size=world_size, strategy=DDPStrategy())
        module = SimpleNamespace(device=torch.device("cpu"))

        def callback(
            count: int,
            *,
            include_token: bool = False,
            window_capacity: int = 2,
        ):
            result = LossSummaryCallback(window_capacity=window_capacity)
            for index in range(count):
                outputs = {"loss": torch.tensor(float(index + 1 + 2 * rank))}
                if include_token:
                    outputs["token"] = LossItem(torch.tensor(float(index + 1)))
                result.on_train_batch_end(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    outputs,
                    None,
                    index,
                )
            return result

        payload = {}
        equal = callback(3)
        equal.on_train_end(trainer, module)
        payload["equal"] = {"report": equal.report()}

        uneven = callback(3 - rank)
        try:
            uneven.on_train_end(trainer, module)
        except RuntimeError as exc:
            payload["uneven"] = {"error": str(exc)}
        else:
            payload["uneven"] = {"report": uneven.report()}

        missing = callback(1, include_token=rank == 0)
        try:
            missing.on_train_end(trainer, module)
        except RuntimeError as exc:
            payload["missing"] = {"error": str(exc)}
        else:
            payload["missing"] = {"report": missing.report()}

        capacity = callback(1, window_capacity=rank + 2)
        try:
            capacity.on_train_end(trainer, module)
        except RuntimeError as exc:
            payload["capacity"] = {"error": str(exc)}
        else:
            payload["capacity"] = {"report": capacity.report()}

        path = Path(output_dir) / f"rank-{rank}.json"
        path.write_text(json.dumps(payload))
    finally:
        torch.distributed.destroy_process_group()


class LightningTest(unittest.TestCase):
    def test_debug_callback_reports_first_gradient_after_backward(self):
        import torch

        from anytrain.lightning import DebugCallback

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clean = torch.nn.Linear(1, 1, bias=False)
                self.bad = torch.nn.Linear(1, 1, bias=False)

        module = Module()
        module.clean.weight.grad = torch.ones_like(module.clean.weight)
        module.bad.weight.grad = torch.full_like(module.bad.weight, float("inf"))
        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        callback = DebugCallback()

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(RuntimeError, "Non-finite gradient"):
            callback.on_after_backward(trainer, module)

        self.assertIn("bad.weight", stderr.getvalue())

    def test_debug_callback_checks_frozen_parameters_once_at_train_start(self):
        import torch

        from anytrain.lightning import DebugCallback

        module = torch.nn.Linear(1, 1, bias=False)
        module.weight.requires_grad_(False)
        module.weight.fill_(float("nan"))
        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        with self.assertRaisesRegex(RuntimeError, "at train start"):
            DebugCallback().on_train_start(trainer, module)

    def test_debug_callback_rejects_nonfinite_loss_before_backward(self):
        import torch

        from anytrain.lightning import DebugCallback

        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        with self.assertRaisesRegex(RuntimeError, "Non-finite loss"):
            DebugCallback().on_before_backward(
                trainer,
                torch.nn.Linear(1, 1),
                torch.tensor(float("nan"), requires_grad=True),
            )

    def test_debug_callback_uses_aggregate_fast_path_for_finite_tensors(self):
        from unittest.mock import patch

        import torch

        from anytrain.lightning import DebugCallback

        module = torch.nn.Sequential(*(torch.nn.Linear(2, 2, bias=False) for _ in range(8)))
        for parameter in module.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        with (
            patch(
                "anytrain.lightning.callback.debug._all_finite",
                return_value=True,
            ) as aggregate,
            patch("anytrain.lightning.callback.debug._find_nonfinite_tensor") as locate,
        ):
            DebugCallback().on_after_backward(trainer, module)

        aggregate.assert_called_once()
        locate.assert_not_called()

    def test_prefixed_log_dict(self):
        from anytrain.lightning import prefixed_log_dict

        self.assertEqual(
            prefixed_log_dict("train/loss", {"recon": 1.0, "/adv/": 0.5}),
            {"train/loss/recon": 1.0, "train/loss/adv": 0.5},
        )

    def test_window_summary_reports_edges_and_window_means(self):
        from anytrain.lightning import window_summary

        summary = window_summary([1.0, 2.0, 4.0], window=2)

        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["window"], 2)
        self.assertEqual(summary["first"], 1.0)
        self.assertEqual(summary["last"], 4.0)
        self.assertEqual(summary["first_mean"], 1.5)
        self.assertEqual(summary["last_mean"], 3.0)
        self.assertEqual(summary["last_to_first"], 2.0)

    def test_window_summary_exposes_zero_baseline(self):
        from anytrain.lightning import window_summary

        summary = window_summary([0.0, 0.0])

        self.assertIsNone(summary["last_to_first"])

    def test_loss_summary_callback_collects_output_loss_items(self):
        import torch

        from anytrain.lightning import LossSummaryCallback
        from anytrain.loss import LossItem

        callback = LossSummaryCallback()
        outputs = {
            "loss": torch.tensor(2.0, requires_grad=True),
            "token": LossItem(torch.tensor([1.0, 3.0])),
            "ignored": object(),
        }

        callback.on_train_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            outputs,
            None,
            0,
        )

        self.assertEqual(callback.values, {"loss": [2.0], "token": [2.0]})
        self.assertFalse(callback._values["loss"].first[0].requires_grad)
        with self.assertRaisesRegex(RuntimeError, "training-end synchronization"):
            callback.report()
        callback.on_train_end(SimpleNamespace(), SimpleNamespace())
        self.assertEqual(callback.report(window=1)["token"]["last_mean"], 2.0)

    def test_loss_summary_callback_uses_injected_item_order(self):
        import torch

        from anytrain.lightning import LossSummaryCallback
        from anytrain.loss import LossItem

        callback = LossSummaryCallback(item_names=("flow", "token"))
        outputs = {
            "loss": torch.tensor(5.0),
            "token": LossItem(torch.tensor([1.0])),
            "flow": LossItem(torch.tensor([2.0])),
        }

        callback.on_train_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            outputs,
            None,
            0,
        )

        self.assertEqual(list(callback.values), ["loss", "flow", "token"])

    def test_loss_summary_callback_keeps_bounded_device_windows(self):
        import torch

        from anytrain.lightning import LossSummaryCallback

        callback = LossSummaryCallback(window_capacity=2)
        for value in range(5):
            callback.on_train_batch_end(
                SimpleNamespace(),
                SimpleNamespace(),
                {"loss": torch.tensor(float(value + 1))},
                None,
                value,
            )

        values = callback._values["loss"]
        self.assertEqual(values.steps, 5)
        self.assertEqual(len(values.first), 2)
        self.assertEqual(len(values.last), 2)
        self.assertEqual(callback.values["loss"], [1.0, 2.0, 4.0, 5.0])
        callback.on_train_end(SimpleNamespace(), SimpleNamespace())
        self.assertEqual(callback.report()["loss"]["window"], 2)
        self.assertEqual(callback.report(window=2)["loss"]["first_mean"], 1.5)
        self.assertEqual(callback.report(window=2)["loss"]["last_mean"], 4.5)
        with self.assertRaisesRegex(ValueError, "retained loss summary capacity"):
            callback.report(window=3)

    def test_loss_summary_callback_does_not_materialize_batch_values(self):
        import torch

        from anytrain.lightning import LossSummaryCallback

        class DeviceOnlyTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, value):
                return torch.Tensor._make_subclass(cls, torch.tensor(value), False)

            def __float__(self):
                raise AssertionError("batch hook converted a tensor to float")

            def item(self):
                raise AssertionError("batch hook materialized a tensor item")

            def cpu(self, *args, **kwargs):
                raise AssertionError("batch hook copied a tensor to the host")

        callback = LossSummaryCallback()
        callback.on_train_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            {"loss": DeviceOnlyTensor(2.0)},
            None,
            0,
        )

        self.assertIsInstance(callback._values["loss"].first[0], DeviceOnlyTensor)

    def test_loss_summary_callback_reduces_windows_at_train_end(self):
        import torch

        from anytrain.lightning import LossSummaryCallback

        class Strategy:
            def __init__(self):
                self.operations = []

            def reduce(self, value, *, reduce_op):
                self.operations.append(reduce_op)
                if reduce_op == "mean":
                    return value + 1
                return value

        callback = LossSummaryCallback(window_capacity=2)
        for value in (1.0, 2.0, 3.0):
            callback.on_train_batch_end(
                SimpleNamespace(),
                SimpleNamespace(),
                {"loss": torch.tensor(value)},
                None,
                0,
            )
        strategy = Strategy()
        trainer = SimpleNamespace(world_size=2, strategy=strategy)

        callback.on_train_end(trainer, SimpleNamespace())

        summary = callback.report(window=2)["loss"]
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["first"], 2.0)
        self.assertEqual(summary["last"], 4.0)
        self.assertEqual(summary["first_mean"], 2.5)
        self.assertEqual(summary["last_mean"], 3.5)
        operations = ["min", "max", "min", "max", "mean", "mean"]
        self.assertEqual(strategy.operations, operations)

        callback.on_fit_end(trainer, SimpleNamespace())
        self.assertEqual(strategy.operations, operations)

    def test_loss_summary_callback_rejects_uneven_distributed_counts(self):
        import torch

        from anytrain.lightning import LossSummaryCallback

        class Strategy:
            def reduce(self, value, *, reduce_op):
                if value.ndim != 0:
                    return value
                if reduce_op == "min":
                    return value
                if reduce_op == "max":
                    return value + 1
                return value

        callback = LossSummaryCallback()
        callback.on_train_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            {"loss": torch.tensor(1.0)},
            None,
            0,
        )

        with self.assertRaisesRegex(RuntimeError, "different observation counts"):
            callback.on_train_end(
                SimpleNamespace(world_size=2, strategy=Strategy()),
                SimpleNamespace(),
            )

    def test_loss_summary_callback_validates_real_gloo_ranks(self):
        import json
        from tempfile import TemporaryDirectory

        import torch

        if not (
            torch.distributed.is_available()
            and torch.distributed.is_gloo_available()
        ):
            self.skipTest("Gloo distributed backend is unavailable")

        with TemporaryDirectory() as tmp_dir:
            init_method = (Path(tmp_dir) / "init").as_uri()
            torch.multiprocessing.spawn(
                _loss_summary_ddp_worker,
                args=(2, init_method, tmp_dir),
                nprocs=2,
                join=True,
            )
            payloads = [
                json.loads((Path(tmp_dir) / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]

        self.assertEqual(payloads[0], payloads[1])
        report = payloads[0]["equal"]["report"]
        self.assertEqual(report["loss"]["steps"], 3)
        self.assertEqual(report["loss"]["first_mean"], 2.5)
        self.assertEqual(report["loss"]["last_mean"], 3.5)
        self.assertIn("different observation counts", payloads[0]["uneven"]["error"])
        self.assertIn("metric names differ", payloads[0]["missing"]["error"])
        self.assertIn("window capacity is inconsistent", payloads[0]["capacity"]["error"])

    def test_loss_summary_callback_runs_through_trainer_lifecycle(self):
        import torch
        from lightning import pytorch as pl
        from torch.utils.data import DataLoader

        from anytrain.lightning import LossSummaryCallback

        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def training_step(self, batch, batch_idx):
                del batch_idx
                loss = (self.weight - batch).square().mean()
                return {"loss": loss}

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        callback = LossSummaryCallback(window_capacity=1)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(Module(), train_dataloaders=DataLoader(torch.ones(2), batch_size=1))

        summary = callback.report()["loss"]
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["window"], 1)

    def test_loss_summary_callback_finishes_empty_fit_without_collective(self):
        import torch
        from lightning import pytorch as pl
        from torch.utils.data import DataLoader

        from anytrain.lightning import LossSummaryCallback

        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def training_step(self, batch, batch_idx):
                del batch_idx
                return {"loss": (self.weight - batch).square().mean()}

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        callback = LossSummaryCallback()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=0,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(Module(), train_dataloaders=DataLoader(torch.ones(1), batch_size=1))

        self.assertEqual(callback.report(), {})

        class Strategy:
            def reduce(self, value, *, reduce_op):
                del value, reduce_op
                raise AssertionError("on_fit_end must not start a collective")

        callback = LossSummaryCallback()
        callback.on_fit_end(
            SimpleNamespace(world_size=2, strategy=Strategy()),
            SimpleNamespace(device=torch.device("cpu")),
        )
        self.assertEqual(callback.report(), {})

    def test_loss_item_logger_callback_groups_loss_items_by_label(self):
        import torch

        from anytrain.lightning import LossItemLoggerCallback
        from anytrain.loss import LossItem

        module = _LogModule()
        callback = LossItemLoggerCallback(
            lambda objective, batch: batch[objective],
            template="{objective}/{key}/{task}",
            label_name="task",
        )
        outputs = {
            "loss": torch.tensor(9.0),
            "token": LossItem(
                torch.tensor([1.0, 3.0, 5.0]),
                details={"tokens": torch.tensor([2.0, 4.0, 6.0])},
            ),
        }

        callback.on_train_batch_end(
            SimpleNamespace(),
            module,
            outputs,
            {"token": ["asr", "mt", "asr"]},
            0,
        )

        self.assertEqual(
            module.log_calls,
            [
                ("token/loss/asr", 4.0),
                ("token/tokens/asr", 4.0),
                ("token/loss/mt", 3.0),
                ("token/tokens/mt", 4.0),
            ],
        )

    def test_loss_item_logger_callback_rejects_label_mismatch(self):
        import torch

        from anytrain.lightning import LossItemLoggerCallback
        from anytrain.loss import LossItem

        callback = LossItemLoggerCallback(lambda objective, batch: ["asr"])
        outputs = {
            "loss": torch.tensor(2.0),
            "token": LossItem(torch.tensor([1.0, 3.0])),
        }

        with self.assertRaisesRegex(ValueError, "token loss rows must align"):
            callback.on_train_batch_end(
                SimpleNamespace(),
                _LogModule(),
                outputs,
                None,
                0,
            )

    def test_loss_time_bucket_logger_records_histogram_and_bucketed_loss(self):
        from unittest.mock import Mock

        import torch

        from anytrain.lightning import LossTimeBucketLoggerCallback
        from anytrain.loss import LossItem

        experiment = Mock()
        strategy = Mock()
        strategy.reduce.side_effect = lambda value, *, reduce_op: value
        trainer = SimpleNamespace(
            global_step=2,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            strategy=strategy,
        )
        callback = LossTimeBucketLoggerCallback(
            item_name="flow",
            detail_key="t",
            histogram_tag="flow/time",
            scalar_template="{item}/{key}/{lower:.1f}_{upper:.1f}",
            every_n_steps=1,
            bucket_count=2,
        )
        outputs = {
            "flow": LossItem(
                loss=torch.tensor([1.0, 3.0, 5.0, 7.0]),
                details={"t": torch.tensor([0.0, 0.2, 0.6, 1.0])},
            ),
        }

        callback.on_train_batch_end(trainer, SimpleNamespace(), outputs, None, 0)
        callback.on_train_batch_end(trainer, SimpleNamespace(), outputs, None, 1)

        experiment.add_histogram.assert_called_once()
        self.assertEqual(experiment.add_histogram.call_args.args[0], "flow/time")
        scalar_calls = [
            (call.args[0], call.args[1], call.args[2])
            for call in experiment.add_scalar.call_args_list
        ]
        self.assertEqual(
            scalar_calls,
            [
                ("flow/t/0.0_0.5", 2.0, 2),
                ("flow/t/0.5_1.0", 6.0, 2),
            ],
        )
        self.assertEqual(strategy.reduce.call_count, 2)

    def test_loss_time_bucket_logger_skips_histogram_on_nonzero_rank(self):
        from unittest.mock import Mock

        import torch

        from anytrain.lightning import LossTimeBucketLoggerCallback
        from anytrain.loss import LossItem

        experiment = Mock()
        trainer = SimpleNamespace(
            global_step=2,
            is_global_zero=False,
            logger=SimpleNamespace(experiment=experiment),
            strategy=SimpleNamespace(reduce=lambda value, *, reduce_op: value),
        )
        callback = LossTimeBucketLoggerCallback(
            item_name="flow",
            detail_key="t",
            histogram_tag="flow/time",
            scalar_template="{item}/{key}/{lower:.1f}_{upper:.1f}",
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            trainer,
            SimpleNamespace(),
            {"flow": LossItem(torch.tensor([1.0]), {"t": torch.tensor([0.5])})},
            None,
            0,
        )

        experiment.add_histogram.assert_not_called()

    def test_gradient_probe_logger_compares_loss_item_gradients(self):
        import math

        import torch

        from anytrain.lightning import (
            GradientComparison,
            GradientProbeLoggerCallback,
            GradientTarget,
            gradient_loss_groups,
        )
        from anytrain.loss import LossItem

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0]))
                self.log_calls = []

            def current_loss_outputs(self):
                return {
                    "token": LossItem(self.weight.square()),
                    "flow": LossItem((3 * self.weight).square()),
                }

            def log(self, name, value, **kwargs):
                self.log_calls.append((name, float(value), kwargs))

        module = Module()
        trainer = SimpleNamespace(global_step=1)
        callback = GradientProbeLoggerCallback(
            (
                GradientComparison(
                    GradientTarget("token"),
                    GradientTarget("flow"),
                ),
            ),
            {"parameter": ("weight",)},
            loss_groups=gradient_loss_groups,
            every_n_steps=2,
        )

        callback.on_before_backward(trainer, module, torch.tensor(0.0))
        self.assertEqual(module.log_calls, [])

        trainer.global_step = 2
        callback.on_before_backward(trainer, module, torch.tensor(0.0))

        values = {name: value for name, value, _ in module.log_calls}
        prefix = "grad/token@batch__flow@batch/parameter"
        self.assertAlmostEqual(values[f"{prefix}/norm/token@batch"], 4.0)
        self.assertAlmostEqual(values[f"{prefix}/norm/flow@batch"], 36.0)
        self.assertAlmostEqual(values[f"{prefix}/log_ratio"], math.log(4.0 / 36.0))
        self.assertAlmostEqual(values[f"{prefix}/cosine"], 1.0)

    def test_gradient_probe_logger_compares_parameter_groups(self):
        import torch

        from anytrain.lightning import (
            GradientComparison,
            GradientProbeLoggerCallback,
            GradientTarget,
            gradient_loss_groups,
        )
        from anytrain.loss import LossItem

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
                self.ffn = torch.nn.Parameter(torch.tensor([3.0]))
                self.log_calls = []

            def current_loss_outputs(self):
                total = self.attn.sum() + self.ffn.sum()
                return {
                    "token": LossItem(total.square()),
                    "flow": LossItem((2 * total).square()),
                }

            def log(self, name, value, **kwargs):
                self.log_calls.append((name, float(value), kwargs))

        module = Module()
        trainer = SimpleNamespace(global_step=2)
        callback = GradientProbeLoggerCallback(
            (
                GradientComparison(
                    GradientTarget("token"),
                    GradientTarget("flow"),
                ),
            ),
            {"block": ("attn", "ffn")},
            loss_groups=gradient_loss_groups,
            every_n_steps=2,
        )

        callback.on_before_backward(trainer, module, torch.tensor(0.0))

        values = {name: value for name, value, _ in module.log_calls}
        self.assertAlmostEqual(
            values["grad/token@batch__flow@batch/block/cosine"],
            1.0,
            places=6,
        )
        self.assertIn("grad/token@batch__flow@batch/block/norm/token@batch", values)
        self.assertIn("grad/token@batch__flow@batch/block/norm/flow@batch", values)

    def test_gradient_probe_logger_compares_source_groups(self):
        import torch

        from anytrain.lightning import (
            GradientComparison,
            GradientProbeLoggerCallback,
            GradientTarget,
            gradient_loss_groups,
        )
        from anytrain.loss import LossItem

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0, 1.0]))
                self.log_calls = []

            def current_gradient_loss_groups(self):
                return {
                    "tts": {"token": LossItem(self.weight.sum().square())},
                    "mt": {
                        "token": LossItem(
                            (self.weight[0] - self.weight[1]).square()
                        )
                    },
                }

            def log(self, name, value, **kwargs):
                self.log_calls.append((name, float(value), kwargs))

        module = Module()
        trainer = SimpleNamespace(global_step=1)
        callback = GradientProbeLoggerCallback(
            comparisons=(
                GradientComparison(
                    GradientTarget("token", "tts"),
                    GradientTarget("token", "mt"),
                ),
            ),
            probes={"layer0_attn": ("weight",)},
            loss_groups=gradient_loss_groups,
            every_n_steps=1,
        )

        callback.on_before_backward(trainer, module, torch.tensor(0.0))

        values = {name: value for name, value, _ in module.log_calls}
        self.assertAlmostEqual(
            values["grad/token@tts__token@mt/layer0_attn/cosine"],
            0.0,
        )

    def test_gradient_probe_logger_rejects_invalid_loss_group_structure(self):
        from typing import Any, cast

        import torch

        from anytrain.lightning import (
            GradientComparison,
            GradientProbeLoggerCallback,
            GradientTarget,
        )

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        callback = GradientProbeLoggerCallback(
            comparisons=(
                GradientComparison(
                    GradientTarget("token"),
                    GradientTarget("flow"),
                ),
            ),
            probes={"parameter": ("weight",)},
            loss_groups=lambda pl_module: cast(Any, {"batch": object()}),
            every_n_steps=1,
        )

        with self.assertRaisesRegex(
            TypeError,
            "loss_groups provider group 'batch' must be a mapping",
        ):
            callback.on_before_backward(
                SimpleNamespace(global_step=0),
                Module(),
                torch.tensor(0.0),
            )

    def test_lightning_log_mixin_logs_prefixed_dict(self):
        from anytrain.lightning import LightningLogMixin

        class Module(LightningLogMixin, _LogModule):
            pass

        module = Module()
        module.log_prefixed_dict("val", {"loss": 2.0}, sync_dist=True)

        self.assertEqual(module.log_dict_calls, [({"val/loss": 2.0}, {"sync_dist": True})])

    def test_lightning_log_mixin_requires_media_logger(self):
        import torch

        from anytrain.lightning import LightningLogMixin

        class Module(LightningLogMixin):
            global_step = 3
            trainer = SimpleNamespace(loggers=[], is_global_zero=True)

        with self.assertRaisesRegex(RuntimeError, "no configured logger"):
            Module().log_audio("sample", torch.zeros(1, 8), sample_rate=16000)

    def test_lightning_log_mixin_skips_media_on_nonzero_rank_by_default(self):
        import torch
        from lightning.pytorch.loggers import TensorBoardLogger

        from anytrain.lightning import LightningLogMixin

        class Experiment:
            def __init__(self):
                self.audio = []
                self.figures = []

            def add_audio(self, tag, audio, *, global_step, sample_rate):
                self.audio.append((tag, audio, global_step, sample_rate))

            def add_figure(self, tag, figure, *, global_step):
                self.figures.append((tag, figure, global_step))

        class Logger(TensorBoardLogger):
            def __init__(self):
                self._experiment = Experiment()

            @property
            def experiment(self):
                return self._experiment

        logger = Logger()

        class Module(LightningLogMixin):
            global_step = 3
            trainer = SimpleNamespace(
                loggers=[logger],
                is_global_zero=False,
                world_size=4,
                global_rank=2,
            )

        module = Module()
        module.log_audio("sample", torch.zeros(1, 8), sample_rate=16000)
        module.log_figure("figure", object())

        self.assertEqual(logger.experiment.audio, [])
        self.assertEqual(logger.experiment.figures, [])

    def test_lightning_log_mixin_prefixes_rank_when_logging_all_ranks(self):
        import torch
        from lightning.pytorch.loggers import TensorBoardLogger

        from anytrain.lightning import LightningLogMixin

        class Experiment:
            def __init__(self):
                self.audio = []
                self.figures = []

            def add_audio(self, tag, audio, *, global_step, sample_rate):
                self.audio.append((tag, audio, global_step, sample_rate))

            def add_figure(self, tag, figure, *, global_step):
                self.figures.append((tag, figure, global_step))

        class Logger(TensorBoardLogger):
            def __init__(self):
                self._experiment = Experiment()

            @property
            def experiment(self):
                return self._experiment

        logger = Logger()
        figure = object()

        class Module(LightningLogMixin):
            global_step = 3
            trainer = SimpleNamespace(
                loggers=[logger],
                is_global_zero=False,
                world_size=4,
                global_rank=2,
            )

        module = Module()
        module.log_audio("sample", torch.zeros(1, 8), sample_rate=16000, rank_mode="all")
        module.log_figure("figure", figure, rank_mode="all")

        self.assertEqual(logger.experiment.audio[0][0], "rank=2/sample")
        self.assertEqual(logger.experiment.audio[0][2:], (3, 16000))
        self.assertEqual(logger.experiment.figures, [("rank=2/figure", figure, 3)])

    def test_debug_callback_can_be_passed_to_trainer(self):
        from lightning import pytorch as pl

        from anytrain.lightning import DebugCallback

        callback = DebugCallback()
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[callback],
        )

        self.assertTrue(any(item is callback for item in trainer.callbacks))

    def test_model_checkpoint_matches_lightning_checkpoint_interface(self):
        import inspect

        from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint

        from anytrain.lightning import ModelCheckpoint

        original = inspect.signature(LightningModelCheckpoint.__init__)
        custom = inspect.signature(ModelCheckpoint.__init__)

        self.assertEqual(list(custom.parameters)[:-2], list(original.parameters))
        self.assertEqual(custom.parameters["async_save"].default, True)
        self.assertEqual(custom.parameters["async_save"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(custom.parameters["max_pending_async_saves"].default, 1)
        self.assertEqual(
            custom.parameters["max_pending_async_saves"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_model_checkpoint_can_be_pickled_before_async_work(self):
        import pickle

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        restored = pickle.loads(pickle.dumps(callback))

        self.assertTrue(restored.async_save)
        restored.wait_async_saves()

    def test_model_checkpoint_async_save_copies_from_local_tmp(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / "nfs" / "model.ckpt"
                trainer = _CheckpointTrainer(payload=b"saved")

                callback._save_checkpoint(trainer, str(target))
                callback.wait_async_saves()

                self.assertEqual(target.read_bytes(), b"saved")
                self.assertNotEqual(trainer.saved_paths[0][0], target)
                self.assertEqual(trainer.saved_paths[0][0].suffix, ".ckpt")
                self.assertFalse(trainer.saved_paths[0][1])
                self.assertEqual(callback._last_checkpoint_saved, str(target))
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_async_save_skips_file_check_on_non_zero_rank(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        class RankNonZeroTrainer(_CheckpointTrainer):
            def __init__(self):
                super().__init__(payload=b"ignored")
                self.is_global_zero = False

            def save_checkpoint(self, filepath, weights_only):
                # Mimic DDP: non-zero ranks enter the collective but do not write.
                self.saved_paths.append((Path(filepath), weights_only))

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / "nfs" / "model.ckpt"
                trainer = RankNonZeroTrainer()

                callback._save_checkpoint(trainer, str(target))
                callback.wait_async_saves()

                self.assertEqual(callback._last_checkpoint_saved, str(target))
                self.assertFalse(target.exists())
                self.assertFalse(trainer.saved_paths[0][0].exists())
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_async_save_exposes_local_path_until_copy_finishes(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / "nfs" / "model.ckpt"
                trainer = _CheckpointTrainer(payload=b"saved")

                callback._save_checkpoint(trainer, str(target))
                local = Path(callback._last_checkpoint_saved)
                self.assertTrue(local.is_file())
                self.assertNotEqual(local, target)
                self.assertFalse(target.exists())

                callback.wait_async_saves()
                self.assertEqual(callback._last_checkpoint_saved, str(target))
                self.assertTrue(target.is_file())
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_notifies_loggers_after_async_copy(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        class Logger:
            def __init__(self, target):
                self.target = target
                self.seen = []

            def after_save_checkpoint(self, callback):
                self.seen.append((callback._last_checkpoint_saved, self.target.exists()))

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / "nfs" / "model.ckpt"
                logger = Logger(target)
                trainer = _CheckpointTrainer(payload=b"saved")
                trainer.loggers = [logger]

                callback._save_checkpoint(trainer, str(target))

                self.assertEqual(logger.seen, [])
                callback.wait_async_saves()
                self.assertEqual(logger.seen, [(str(target), True)])
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_rejects_sharded_async_strategy(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        class FSDPStrategy:
            pass

        with TemporaryDirectory() as tmp_dir:
            trainer = _CheckpointTrainer()
            trainer.strategy = FSDPStrategy()
            callback = ModelCheckpoint(async_save=True)

            with self.assertRaisesRegex(RuntimeError, "async_save.*FSDP"):
                callback._save_checkpoint(trainer, str(Path(tmp_dir) / "model.ckpt"))

    def test_model_checkpoint_rejects_directory_async_output(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        class DirectoryCheckpointTrainer(_CheckpointTrainer):
            def save_checkpoint(self, filepath, weights_only):
                del weights_only
                path = Path(filepath)
                path.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as tmp_dir:
            trainer = DirectoryCheckpointTrainer()
            callback = ModelCheckpoint(async_save=True)

            with self.assertRaisesRegex(RuntimeError, "regular file"):
                callback._save_checkpoint(trainer, str(Path(tmp_dir) / "model.ckpt"))

            callback._close_async_storage()

    def test_model_checkpoint_sync_opt_out_uses_target_path(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model.ckpt"
            trainer = _CheckpointTrainer(payload=b"sync")
            callback = ModelCheckpoint(async_save=False)

            callback._save_checkpoint(trainer, str(target))

            self.assertEqual(target.read_bytes(), b"sync")
            self.assertEqual(trainer.saved_paths, [(target, False)])

    def test_model_checkpoint_async_remove_is_ordered_after_copy(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                source = Path(tmp_dir) / "source.ckpt"
                source.write_bytes(b"old")
                target = Path(tmp_dir) / "target.ckpt"

                callback._submit_async_copy(source, target)
                callback._submit_async_remove(target)
                callback.wait_async_saves()

                self.assertFalse(target.exists())
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_backpressure_precedes_next_local_save(self):
        from tempfile import TemporaryDirectory
        from threading import Event, Thread
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True, max_pending_async_saves=1)
        copy_started = Event()
        release_copy = Event()
        capacity_wait_started = Event()
        save_errors = []
        save_thread = None
        original_copy = callback._copy_to_target
        original_capacity_wait = callback._wait_for_async_capacity

        def blocked_copy(source, target):
            copy_started.set()
            if not release_copy.wait(timeout=5):
                raise TimeoutError("test did not release asynchronous checkpoint copy")
            original_copy(source, target)

        def observed_capacity_wait():
            capacity_wait_started.set()
            original_capacity_wait()

        try:
            with TemporaryDirectory() as tmp_dir:
                first = Path(tmp_dir) / "nfs" / "first.ckpt"
                second = Path(tmp_dir) / "nfs" / "second.ckpt"
                trainer = _CheckpointTrainer(payload=b"saved")

                with patch.object(callback, "_copy_to_target", side_effect=blocked_copy):
                    callback._save_checkpoint(trainer, str(first))
                    self.assertTrue(copy_started.wait(timeout=2))

                    def save_second():
                        try:
                            callback._save_checkpoint(trainer, str(second))
                        except BaseException as error:
                            save_errors.append(error)

                    with patch.object(
                        callback,
                        "_wait_for_async_capacity",
                        side_effect=observed_capacity_wait,
                    ):
                        save_thread = Thread(target=save_second)
                        save_thread.start()
                        self.assertTrue(capacity_wait_started.wait(timeout=2))
                        self.assertTrue(save_thread.is_alive())
                        self.assertEqual(len(trainer.saved_paths), 1)
                        release_copy.set()
                        save_thread.join(timeout=5)

                    self.assertFalse(save_thread.is_alive())
                    self.assertEqual(save_errors, [])
                    self.assertEqual(len(trainer.saved_paths), 2)
                    callback.wait_async_saves()

                self.assertEqual(first.read_bytes(), b"saved")
                self.assertEqual(second.read_bytes(), b"saved")
        finally:
            release_copy.set()
            if save_thread is not None:
                save_thread.join(timeout=5)
            callback._close_async_storage()

    def test_model_checkpoint_backpressure_propagates_copy_error_before_save(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True, max_pending_async_saves=1)
        try:
            with TemporaryDirectory() as tmp_dir:
                trainer = _CheckpointTrainer(payload=b"saved")
                first = Path(tmp_dir) / "first.ckpt"
                second = Path(tmp_dir) / "second.ckpt"

                with patch.object(
                    callback,
                    "_copy_to_target",
                    side_effect=OSError("storage unavailable"),
                ):
                    callback._save_checkpoint(trainer, str(first))
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Asynchronous checkpoint storage failed",
                    ):
                        callback._save_checkpoint(trainer, str(second))

                self.assertEqual(len(trainer.saved_paths), 1)
                self.assertFalse(second.exists())
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_wait_drains_all_futures_after_logger_error(self):
        from tempfile import TemporaryDirectory
        from threading import Event
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        class Logger:
            def __init__(self) -> None:
                self.saved_paths = []

            def after_save_checkpoint(self, callback):
                self.saved_paths.append(callback._last_checkpoint_saved)
                if len(self.saved_paths) == 1:
                    raise RuntimeError("logger unavailable")

        callback = ModelCheckpoint(async_save=True, max_pending_async_saves=2)
        release_copy = Event()
        original_copy = callback._copy_to_target

        def blocked_copy(source, target):
            if not release_copy.wait(timeout=5):
                raise TimeoutError("test did not release asynchronous checkpoint copy")
            original_copy(source, target)

        try:
            with TemporaryDirectory() as tmp_dir:
                first = Path(tmp_dir) / "nfs" / "first.ckpt"
                second = Path(tmp_dir) / "nfs" / "second.ckpt"
                logger = Logger()
                trainer = _CheckpointTrainer(payload=b"saved")
                trainer.loggers = [logger]

                with patch.object(callback, "_copy_to_target", side_effect=blocked_copy):
                    callback._save_checkpoint(trainer, str(first))
                    callback._save_checkpoint(trainer, str(second))
                    local_paths = [path for path, _ in trainer.saved_paths]
                    release_copy.set()
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Asynchronous checkpoint storage failed",
                    ):
                        callback.wait_async_saves()

                self.assertEqual(logger.saved_paths, [str(first), str(second)])
                self.assertEqual(callback._last_checkpoint_saved, str(second))
                self.assertTrue(all(not path.exists() for path in local_paths))
                self.assertEqual(callback._async_futures, [])
                self.assertEqual(callback._async_callbacks, {})
        finally:
            release_copy.set()
            callback._close_async_storage()

    def test_model_checkpoint_ddp_propagates_prior_error_before_save_collective(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        class BroadcastState:
            value = None

        class Strategy(_CheckpointStrategy):
            def __init__(self, state):
                super().__init__()
                self.state = state

            def broadcast(self, value, *, src):
                self.assert_source(src)
                if value is not None:
                    self.state.value = value
                return self.state.value

            @staticmethod
            def assert_source(src):
                if src != 0:
                    raise AssertionError(f"expected broadcast source 0, got {src}")

        callback_zero = ModelCheckpoint(async_save=True)
        callback_nonzero = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                source = Path(tmp_dir) / "source.ckpt"
                source.write_bytes(b"pending")
                state = BroadcastState()

                with patch.object(
                    callback_zero,
                    "_copy_to_target",
                    side_effect=OSError("storage unavailable"),
                ):
                    callback_zero._submit_async_copy(
                        source,
                        Path(tmp_dir) / "failed.ckpt",
                    )
                    failed = callback_zero._async_futures[0]
                    with self.assertRaises(OSError):
                        failed.result()

                trainer_zero = _CheckpointTrainer(payload=b"rank-zero")
                trainer_zero.world_size = 2
                trainer_zero.strategy = Strategy(state)
                trainer_nonzero = _CheckpointTrainer(payload=b"rank-one")
                trainer_nonzero.world_size = 2
                trainer_nonzero.is_global_zero = False
                trainer_nonzero.strategy = Strategy(state)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "Asynchronous checkpoint storage failed",
                ) as zero_error:
                    callback_zero._save_checkpoint(
                        trainer_zero,
                        str(Path(tmp_dir) / "next.ckpt"),
                    )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Asynchronous checkpoint storage failed",
                ):
                    callback_nonzero._save_checkpoint(
                        trainer_nonzero,
                        str(Path(tmp_dir) / "next.ckpt"),
                    )

                self.assertIsInstance(zero_error.exception.__cause__, OSError)
                self.assertEqual(trainer_zero.saved_paths, [])
                self.assertEqual(trainer_nonzero.saved_paths, [])
        finally:
            callback_zero._close_async_storage()
            callback_nonzero._close_async_storage()

    def test_model_checkpoint_ddp_fit_end_propagates_async_error(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        class BroadcastState:
            value = None

        class Strategy(_CheckpointStrategy):
            def __init__(self, state):
                super().__init__()
                self.state = state

            def broadcast(self, value, *, src):
                if src != 0:
                    raise AssertionError(f"expected broadcast source 0, got {src}")
                if value is not None:
                    self.state.value = value
                return self.state.value

        callback_zero = ModelCheckpoint(async_save=True)
        callback_nonzero = ModelCheckpoint(async_save=True)
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.ckpt"
            source.write_bytes(b"pending")
            state = BroadcastState()

            with patch.object(
                callback_zero,
                "_copy_to_target",
                side_effect=OSError("storage unavailable"),
            ):
                callback_zero._submit_async_copy(
                    source,
                    Path(tmp_dir) / "failed.ckpt",
                )

            trainer_zero = _CheckpointTrainer(payload=b"rank-zero")
            trainer_zero.world_size = 2
            trainer_zero.strategy = Strategy(state)
            trainer_nonzero = _CheckpointTrainer(payload=b"rank-one")
            trainer_nonzero.world_size = 2
            trainer_nonzero.is_global_zero = False
            trainer_nonzero.strategy = Strategy(state)
            module = SimpleNamespace()

            with self.assertRaisesRegex(
                RuntimeError,
                "Asynchronous checkpoint storage failed",
            ) as zero_error:
                callback_zero.on_fit_end(trainer_zero, module)
            with self.assertRaisesRegex(
                RuntimeError,
                "Asynchronous checkpoint storage failed",
            ):
                callback_nonzero.on_fit_end(trainer_nonzero, module)

            self.assertIsInstance(zero_error.exception.__cause__, OSError)

    def test_model_checkpoint_fit_end_drains_callbacks_after_super_error(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint

        from anytrain.lightning import ModelCheckpoint

        class Logger:
            def __init__(self) -> None:
                self.saved_paths = []

            def after_save_checkpoint(self, callback):
                self.saved_paths.append(callback._last_checkpoint_saved)

        callback = ModelCheckpoint(async_save=True)
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model.ckpt"
            trainer = _CheckpointTrainer(payload=b"saved")
            logger = Logger()
            trainer.loggers = [logger]
            callback._save_checkpoint(trainer, str(target))
            local = trainer.saved_paths[0][0]

            with (
                patch.object(
                    LightningModelCheckpoint,
                    "on_fit_end",
                    side_effect=RuntimeError("parent hook failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "parent hook failed"),
            ):
                callback.on_fit_end(trainer, SimpleNamespace())

            self.assertEqual(target.read_bytes(), b"saved")
            self.assertEqual(logger.saved_paths, [str(target)])
            self.assertFalse(local.exists())
            self.assertEqual(callback._async_futures, [])
            self.assertEqual(callback._async_callbacks, {})

    def test_model_checkpoint_reuses_topk_serialization_for_async_last(self):
        from tempfile import TemporaryDirectory
        from threading import Event
        from unittest.mock import patch

        import torch

        from anytrain.lightning import ModelCheckpoint

        class Logger:
            def __init__(self, topk, last) -> None:
                self.topk = topk
                self.last = last
                self.seen = []

            def after_save_checkpoint(self, callback):
                self.seen.append(
                    (
                        callback._last_checkpoint_saved,
                        self.topk.exists(),
                        self.last.exists(),
                    )
                )

        release_copy = Event()
        copy_started = Event()
        callback = None

        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    filename="step-{step:02d}",
                    save_last=True,
                    save_top_k=-1,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                    async_save=True,
                )
                checkpoint_dir = Path(callback.dirpath)
                topk = checkpoint_dir / "step-07.ckpt"
                last = checkpoint_dir / "last.ckpt"
                last.write_bytes(b"previous")
                expected_payload = str(last).encode()

                class StateTrainer(_CheckpointTrainer):
                    def save_checkpoint(self, filepath, weights_only):
                        self.payload = callback.last_model_path.encode()
                        super().save_checkpoint(filepath, weights_only)

                trainer = StateTrainer()
                logger = Logger(topk, last)
                trainer.loggers = [logger]
                candidates = {
                    "epoch": torch.tensor(1),
                    "step": torch.tensor(7),
                }
                original_copy = callback._copy_to_target

                def blocked_copy(source, target):
                    if target == topk:
                        copy_started.set()
                        if not release_copy.wait(timeout=5):
                            raise TimeoutError(
                                "test did not release asynchronous checkpoint copy"
                            )
                    original_copy(source, target)

                with patch.object(callback, "_copy_to_target", side_effect=blocked_copy):
                    callback._save_topk_checkpoint(trainer, candidates)
                    self.assertTrue(copy_started.wait(timeout=2))
                    local = trainer.saved_paths[0][0]
                    callback._save_last_checkpoint(trainer, candidates)

                    self.assertEqual(len(trainer.saved_paths), 1)
                    self.assertEqual(last.read_bytes(), b"previous")
                    self.assertEqual(logger.seen, [])
                    release_copy.set()
                    callback.wait_async_saves()

                self.assertEqual(topk.read_bytes(), expected_payload)
                self.assertEqual(last.read_bytes(), expected_payload)
                self.assertEqual(logger.seen, [(str(last), True, True)])
                self.assertEqual(callback._last_checkpoint_saved, str(last))
                self.assertFalse(local.exists())
        finally:
            release_copy.set()
            if callback is not None:
                callback._close_async_storage()

    def test_model_checkpoint_failed_topk_publish_keeps_previous_last(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        import torch

        from anytrain.lightning import ModelCheckpoint

        callback = None
        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    filename="step-{step:02d}",
                    save_last=True,
                    save_top_k=-1,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                    async_save=True,
                )
                topk = Path(tmp_dir) / "step-07.ckpt"
                last = Path(tmp_dir) / "last.ckpt"
                topk.write_bytes(b"stale-topk")
                last.write_bytes(b"previous-last")
                trainer = _CheckpointTrainer(payload=b"new")
                candidates = {
                    "epoch": torch.tensor(1),
                    "step": torch.tensor(7),
                }

                with patch.object(
                    callback,
                    "_copy_to_target",
                    side_effect=OSError("storage unavailable"),
                ):
                    callback._save_topk_checkpoint(trainer, candidates)
                    callback._save_last_checkpoint(trainer, candidates)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Asynchronous checkpoint storage failed",
                    ):
                        callback.wait_async_saves()

                self.assertEqual(len(trainer.saved_paths), 1)
                self.assertEqual(topk.read_bytes(), b"stale-topk")
                self.assertEqual(last.read_bytes(), b"previous-last")
        finally:
            if callback is not None:
                callback._close_async_storage()

    def test_model_checkpoint_save_top_k_zero_serializes_last_once(self):
        from tempfile import TemporaryDirectory

        import torch

        from anytrain.lightning import ModelCheckpoint

        callback = None
        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    save_last=True,
                    save_top_k=0,
                    enable_version_counter=False,
                    async_save=True,
                )
                trainer = _CheckpointTrainer(payload=b"last-only")
                candidates = {
                    "epoch": torch.tensor(1),
                    "step": torch.tensor(7),
                }

                callback._save_topk_checkpoint(trainer, candidates)
                callback._save_last_checkpoint(trainer, candidates)
                callback.wait_async_saves()

                last = Path(callback.dirpath) / "last.ckpt"
                self.assertEqual(len(trainer.saved_paths), 1)
                self.assertEqual(last.read_bytes(), b"last-only")
                self.assertEqual(callback._last_checkpoint_saved, str(last))
        finally:
            if callback is not None:
                callback._close_async_storage()

    def test_model_checkpoint_exception_reuses_serialization_for_last(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        callback = None
        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    filename="step-{step:02d}",
                    save_last=True,
                    save_top_k=-1,
                    save_on_exception=True,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                    async_save=True,
                )
                trainer = _CheckpointTrainer(payload=b"exception")
                trainer.callback_metrics = {}
                trainer.current_epoch = 1
                trainer.fast_dev_run = False
                trainer.sanity_checking = False

                callback.on_exception(
                    trainer,
                    SimpleNamespace(),
                    RuntimeError("training failed"),
                )

                checkpoint_dir = Path(callback.dirpath)
                self.assertEqual(len(trainer.saved_paths), 1)
                self.assertEqual(
                    (checkpoint_dir / "step-07.ckpt").read_bytes(),
                    b"exception",
                )
                self.assertEqual(
                    (checkpoint_dir / "last.ckpt").read_bytes(),
                    b"exception",
                )
        finally:
            if callback is not None:
                callback._close_async_storage()

    def test_model_checkpoint_link_fallback_uses_reaped_persistent_source(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        import torch

        from anytrain.lightning import ModelCheckpoint

        callback = None
        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    filename="step-{step:02d}",
                    save_last="link",
                    save_top_k=-1,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                    async_save=True,
                )
                trainer = _CheckpointTrainer(payload=b"saved")
                candidates = {
                    "epoch": torch.tensor(1),
                    "step": torch.tensor(7),
                }

                callback._save_topk_checkpoint(trainer, candidates)
                reused = callback._reused_last_save
                self.assertIsNotNone(reused)
                assert reused is not None
                local = reused[2]
                prerequisite = reused[4]
                self.assertIsNotNone(prerequisite)
                assert prerequisite is not None
                prerequisite.result()
                self.assertIsNone(callback._collect_done_async_error())
                self.assertFalse(local.exists())

                with patch("os.symlink", side_effect=OSError("links unavailable")):
                    callback._save_last_checkpoint(trainer, candidates)
                callback.wait_async_saves()

                last = Path(callback.dirpath) / "last.ckpt"
                self.assertFalse(last.is_symlink())
                self.assertEqual(last.read_bytes(), b"saved")
        finally:
            if callback is not None:
                callback._close_async_storage()

    def test_model_checkpoint_save_last_link_targets_persistent_checkpoint(self):
        import os
        from tempfile import TemporaryDirectory
        from threading import Event
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        release_copy = Event()
        copy_started = Event()

        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    async_save=True,
                    save_last="link",
                    enable_version_counter=False,
                )
                original_copy = callback._copy_to_target

                def blocked_copy(source, target):
                    copy_started.set()
                    if not release_copy.wait(timeout=5):
                        raise TimeoutError(
                            "test did not release asynchronous checkpoint copy"
                        )
                    original_copy(source, target)

                target = Path(tmp_dir) / "epoch=1.ckpt"
                link = Path(tmp_dir) / "last.ckpt"
                link.write_bytes(b"previous")
                trainer = _CheckpointTrainer(payload=b"saved")

                with patch.object(callback, "_copy_to_target", side_effect=blocked_copy):
                    callback._save_checkpoint(trainer, str(target))
                    self.assertTrue(copy_started.wait(timeout=2))
                    local = Path(callback._last_checkpoint_saved)

                    callback._save_last_checkpoint(trainer, {})

                    self.assertTrue(link.is_symlink())
                    linked = (link.parent / os.readlink(link)).resolve()
                    self.assertEqual(linked, target.resolve())
                    release_copy.set()
                    callback.wait_async_saves()

                self.assertTrue(link.is_symlink())
                linked = (link.parent / os.readlink(link)).resolve()
                self.assertEqual(linked, target.resolve())
                self.assertEqual(link.read_bytes(), b"saved")
                self.assertFalse(local.exists())
        finally:
            release_copy.set()
            callback._close_async_storage()

    def test_model_checkpoint_save_last_link_falls_back_to_persistent_copy(self):
        from tempfile import TemporaryDirectory
        from threading import Event
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        release_copy = Event()
        copy_started = Event()
        callback = None

        try:
            with TemporaryDirectory() as tmp_dir:
                callback = ModelCheckpoint(
                    dirpath=tmp_dir,
                    async_save=True,
                    save_last="link",
                    enable_version_counter=False,
                )
                original_copy = callback._copy_to_target

                def blocked_copy(source, target):
                    copy_started.set()
                    if not release_copy.wait(timeout=5):
                        raise TimeoutError(
                            "test did not release asynchronous checkpoint copy"
                        )
                    original_copy(source, target)

                target = Path(tmp_dir) / "epoch=1.ckpt"
                link = Path(tmp_dir) / "last.ckpt"
                link.write_bytes(b"previous")
                trainer = _CheckpointTrainer(payload=b"saved")

                with (
                    patch.object(callback, "_copy_to_target", side_effect=blocked_copy),
                    patch("os.symlink", side_effect=OSError("links unavailable")),
                ):
                    callback._save_checkpoint(trainer, str(target))
                    self.assertTrue(copy_started.wait(timeout=2))
                    callback._save_last_checkpoint(trainer, {})

                    self.assertFalse(link.is_symlink())
                    self.assertEqual(link.read_bytes(), b"saved")
                    release_copy.set()
                    callback.wait_async_saves()

                self.assertFalse(link.is_symlink())
                self.assertEqual(link.read_bytes(), b"saved")
        finally:
            release_copy.set()
            if callback is not None:
                callback._close_async_storage()

    def test_model_checkpoint_remove_does_not_consume_copy_backpressure(self):
        from tempfile import TemporaryDirectory
        from threading import Event, Thread
        from unittest.mock import patch

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True, max_pending_async_saves=1)
        copy_started = Event()
        release_copy = Event()
        remove_submitted = Event()
        submit_errors = []
        submit_thread = None
        original_copy = callback._copy_to_target

        def blocked_copy(source, target):
            copy_started.set()
            if not release_copy.wait(timeout=5):
                raise TimeoutError("test did not release asynchronous checkpoint copy")
            original_copy(source, target)

        try:
            with TemporaryDirectory() as tmp_dir:
                source = Path(tmp_dir) / "source.ckpt"
                source.write_bytes(b"saved")
                target = Path(tmp_dir) / "target.ckpt"
                removed = Path(tmp_dir) / "old.ckpt"
                removed.write_bytes(b"old")

                with patch.object(callback, "_copy_to_target", side_effect=blocked_copy):
                    callback._submit_async_copy(source, target)
                    self.assertTrue(copy_started.wait(timeout=2))

                    def submit_remove():
                        try:
                            callback._submit_async_remove(removed)
                        except BaseException as error:
                            submit_errors.append(error)
                        finally:
                            remove_submitted.set()

                    submit_thread = Thread(target=submit_remove)
                    submit_thread.start()
                    self.assertTrue(remove_submitted.wait(timeout=2))
                    self.assertEqual(submit_errors, [])
                    self.assertEqual(len(callback._async_futures), 2)
                    self.assertEqual(len(callback._async_copy_futures), 1)
                    release_copy.set()
                    submit_thread.join(timeout=5)
                    callback.wait_async_saves()

                self.assertTrue(target.is_file())
                self.assertFalse(removed.exists())
        finally:
            release_copy.set()
            if submit_thread is not None:
                submit_thread.join(timeout=5)
            callback._close_async_storage()


if __name__ == "__main__":
    unittest.main()
