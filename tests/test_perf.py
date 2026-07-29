import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


class PerfTest(unittest.TestCase):
    def test_count_parameters_respects_trainable_only(self):
        import torch

        from anytrain.lightning.perf import count_parameters

        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        module[1].weight.requires_grad_(False)
        module[1].bias.requires_grad_(False)

        self.assertEqual(count_parameters(module), 26)
        self.assertEqual(count_parameters(module, trainable_only=True), 16)

    def test_training_flops_from_forward_uses_explicit_backward_multiplier(self):
        from anytrain.lightning.perf import training_flops_from_forward

        self.assertEqual(training_flops_from_forward(10.0), 30.0)
        self.assertEqual(training_flops_from_forward(10.0, backward_multiplier=1.5), 25.0)

    def test_model_flops_utilization(self):
        from anytrain.lightning.perf import model_flops_utilization

        self.assertEqual(
            model_flops_utilization(
                model_flops_per_step=100.0,
                step_time=2.0,
                hardware_peak_flops=200.0,
            ),
            0.25,
        )

    def test_perf_helpers_reject_nonfinite_values(self):
        from anytrain.lightning.perf import (
            infer_peak_flops,
            model_flops_utilization,
            training_flops_from_forward,
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            model_flops_utilization(
                model_flops_per_step=float("nan"),
                step_time=1.0,
                hardware_peak_flops=1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            training_flops_from_forward(1.0, backward_multiplier=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            infer_peak_flops(
                dtype="float32",
                device_name="Custom GPU",
                hardware_peak_flops=float("nan"),
            )

    def test_performance_callback_cleans_up_profiler_on_exception(self):
        import torch

        from anytrain.lightning import PerformanceCallback

        class Counter:
            def __init__(self):
                self.exited = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.exited = True

            def get_total_flops(self):
                return 1.0

        trainer = SimpleNamespace(global_step=0, world_size=1)
        counter = Counter()
        callback = PerformanceCallback(profile_flops=True, sync_cuda=False)
        module = SimpleNamespace(device=torch.device("cpu"))
        with patch(
            "anytrain.lightning.callback.performance._flop_counter",
            return_value=counter,
        ):
            callback.on_train_batch_start(trainer, module, None, 0)
            callback.on_exception(trainer, module, RuntimeError("boom"))

        self.assertTrue(counter.exited)
        self.assertIsNone(callback._flop_counter)

    def test_infer_peak_flops_uses_override(self):
        from anytrain.lightning.perf import infer_peak_flops

        peak = infer_peak_flops(
            dtype="bf16",
            device_name="Custom GPU",
            hardware_peak_flops=123.0,
        )

        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertEqual(peak.flops, 123.0)
        self.assertEqual(peak.source, "override")
        self.assertEqual(peak.device_name, "Custom GPU")
        self.assertEqual(peak.dtype, "bfloat16")

    def test_infer_peak_flops_matches_known_gpu(self):
        from anytrain.lightning.perf import infer_peak_flops

        peak = infer_peak_flops(dtype="bf16", device_name="NVIDIA A100-PCIE-40GB")

        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertEqual(peak.source, "auto")
        self.assertEqual(peak.flops, 312.0e12)

    def test_profile_forward_flops_reports_linear_ops(self):
        import torch

        from anytrain.lightning.perf import profile_forward_flops

        module = torch.nn.Linear(4, 3)
        flops = profile_forward_flops(module, args=(torch.ones(2, 4),))

        self.assertGreater(flops, 0)


class _Logger:
    def __init__(self):
        self.hyperparams = []

    def log_hyperparams(self, values):
        self.hyperparams.append(values)


class _Module:
    def __init__(self):
        import torch

        self.linear = torch.nn.Linear(2, 1)
        self.log_dict_calls = []
        self.device = torch.device("cpu")

    def parameters(self):
        return self.linear.parameters()

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class _FlopsProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, *, trainer, pl_module, outputs, batch, batch_idx):
        self.calls.append((trainer, pl_module, outputs, batch, batch_idx))
        return batch["flops"]


class _DistributedStrategy:
    def __init__(self):
        self.barrier_count = 0

    def barrier(self):
        self.barrier_count += 1

    def reduce(self, tensor, *, reduce_op):
        import torch

        if reduce_op == "sum":
            remote = tensor.new_tensor([300.0, 300.0, 400.0, 1.0, 1.0, 1.0])
            return tensor + remote
        if reduce_op == "max":
            return torch.maximum(tensor, tensor.new_tensor([3.0, 3.0]))
        raise AssertionError(f"Unexpected reduce operation: {reduce_op}")


class _AlternatingSlowRankStrategy:
    def __init__(self):
        self.barrier_count = 0

    def barrier(self):
        self.barrier_count += 1

    def reduce(self, tensor, *, reduce_op):
        import torch

        if reduce_op == "sum":
            remote = tensor.new_tensor([100.0, 200.0, 200.0, 1.0, 1.0, 1.0])
            return tensor + remote
        if reduce_op == "max":
            # Current elapsed, followed by the two window measurements.
            return torch.maximum(tensor, tensor.new_tensor([5.0, 1.0, 5.0]))
        raise AssertionError(f"Unexpected reduce operation: {reduce_op}")


class PerformanceCallbackTest(unittest.TestCase):
    def test_callback_logs_static_metrics_and_mfu(self):
        from anytrain.lightning import PerformanceCallback

        logger = _Logger()
        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[logger],
            global_step=0,
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_step=100.0,
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            measure_window_steps=2,
            sync_cuda=False,
        )

        callback.on_train_start(trainer, module)
        self.assertEqual(module.log_dict_calls[0][0]["perf/model_flops_per_step"], 100.0)
        self.assertEqual(module.log_dict_calls[0][0]["perf/hardware_peak_flops"], 200.0)
        self.assertEqual(
            module.log_dict_calls[0][1],
            {"on_step": False, "on_epoch": True, "logger": True},
        )
        self.assertEqual(logger.hyperparams[0]["perf/hardware_peak_flops_source"], "override")
        self.assertEqual(logger.hyperparams[0]["perf/model_flops_source"], "static")

        with patch("time.perf_counter", side_effect=[1.0, 3.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 2.0)
        self.assertEqual(metrics["perf/model_flops_per_step_window"], 100.0)
        self.assertEqual(metrics["perf/mfu"], 0.25)

    def test_callback_uses_ratio_of_window_sums_for_dynamic_flops(self):
        from anytrain.lightning import PerformanceCallback

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=1,
        )
        module = _Module()
        provider = _FlopsProvider()
        callback = PerformanceCallback(
            model_flops_per_batch=provider,
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            measure_window_steps=2,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs={"loss": 1.0},
                batch={"flops": 100.0},
                batch_idx=0,
            )
        with patch("time.perf_counter", side_effect=[3.0, 6.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 900.0}, batch_idx=1)
            trainer.global_step = 2
            callback.on_train_batch_end(
                trainer,
                module,
                outputs={"loss": 2.0},
                batch={"flops": 900.0},
                batch_idx=1,
            )

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 3.0)
        self.assertEqual(metrics["perf/step_time_window"], 2.0)
        self.assertEqual(metrics["perf/model_flops_per_step"], 900.0)
        self.assertEqual(metrics["perf/model_flops_per_step_window"], 500.0)
        self.assertEqual(metrics["perf/mfu"], 1.25)
        self.assertEqual(provider.calls[-1][2:], ({"loss": 2.0}, {"flops": 900.0}, 1))

    def test_callback_accumulates_microbatches_until_optimizer_step(self):
        from anytrain.lightning import PerformanceCallback

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=1,
        )
        module = _Module()
        provider = _FlopsProvider()
        callback = PerformanceCallback(
            model_flops_per_batch=provider,
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=0)
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 100.0},
                batch_idx=0,
            )
        self.assertEqual(len(module.log_dict_calls), 1)

        with patch("time.perf_counter", side_effect=[3.0, 6.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 300.0}, batch_idx=1)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 300.0},
                batch_idx=1,
            )

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 4.0)
        self.assertEqual(metrics["perf/model_flops_per_step"], 400.0)
        self.assertEqual(metrics["perf/mfu"], 0.5)
        self.assertEqual(len(provider.calls), 2)

    def test_callback_aggregates_ddp_work_and_peak_before_computing_mfu(self):
        from anytrain.lightning import PerformanceCallback

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=2,
            strategy=_DistributedStrategy(),
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_batch=_FlopsProvider(),
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 3.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 100.0},
                batch_idx=0,
            )

        metrics, kwargs = module.log_dict_calls[-1]
        self.assertEqual(metrics["perf/step_time"], 3.0)
        self.assertEqual(metrics["perf/model_flops_per_step"], 200.0)
        self.assertAlmostEqual(metrics["perf/mfu"], 400.0 / 3.0 / 600.0)
        self.assertFalse(kwargs["sync_dist"])
        self.assertEqual(trainer.strategy.barrier_count, 1)

    def test_callback_sums_per_measurement_ddp_max_time(self):
        from anytrain.lightning import PerformanceCallback

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=2,
            strategy=_AlternatingSlowRankStrategy(),
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_batch=_FlopsProvider(),
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=2,
            measure_window_steps=2,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 5.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 100.0},
                batch_idx=0,
            )
        with patch("time.perf_counter", side_effect=[6.0, 7.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=1)
            trainer.global_step = 2
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 100.0},
                batch_idx=1,
            )

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 5.0)
        self.assertEqual(metrics["perf/step_time_window"], 4.5)
        self.assertEqual(metrics["perf/model_flops_per_step_window"], 100.0)
        self.assertAlmostEqual(metrics["perf/mfu"], 400.0 / 9.0 / 400.0)

    def test_callback_resets_pending_measurement_on_new_train_start(self):
        from anytrain.lightning import PerformanceCallback

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=1,
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_batch=_FlopsProvider(),
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 3.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=0)
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 100.0},
                batch_idx=0,
            )

        trainer.global_step = 5
        callback.on_train_start(trainer, module)
        with patch("time.perf_counter", side_effect=[4.0, 5.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 20.0}, batch_idx=1)
            trainer.global_step = 6
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 20.0},
                batch_idx=1,
            )

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 1.0)
        self.assertEqual(metrics["perf/model_flops_per_step"], 20.0)

    def test_callback_discards_unstepped_epoch_tail(self):
        from anytrain.lightning import PerformanceCallback

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=1,
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_batch=_FlopsProvider(),
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 3.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 100.0}, batch_idx=0)
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 100.0},
                batch_idx=0,
            )
        callback.on_train_epoch_end(trainer, module)

        with patch("time.perf_counter", side_effect=[4.0, 5.0]):
            callback.on_train_batch_start(trainer, module, batch={"flops": 20.0}, batch_idx=0)
            trainer.global_step = 1
            callback.on_train_batch_end(
                trainer,
                module,
                outputs=None,
                batch={"flops": 20.0},
                batch_idx=0,
            )

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 1.0)
        self.assertEqual(metrics["perf/model_flops_per_step"], 20.0)

    def test_callback_uses_real_lightning_optimizer_step_boundaries(self):
        import torch
        from lightning import pytorch as pl
        from lightning.pytorch.loggers import CSVLogger
        from torch.utils.data import DataLoader, TensorDataset

        from anytrain.lightning import PerformanceCallback

        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                self.performance_logs = []

            def training_step(self, batch, batch_idx):
                del batch_idx
                inputs, targets, _ = batch
                return torch.nn.functional.mse_loss(self.linear(inputs), targets)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

            def log_dict(self, dictionary, *args, **kwargs):
                if "perf/step_time" in dictionary:
                    self.performance_logs.append(dict(dictionary))
                return super().log_dict(dictionary, *args, **kwargs)

        class Provider:
            def __init__(self):
                self.global_steps = []

            def __call__(self, *, trainer, pl_module, outputs, batch, batch_idx):
                del pl_module, outputs, batch_idx
                self.global_steps.append(int(trainer.global_step))
                return float(batch[2].sum())

        dataset = TensorDataset(
            torch.ones(4, 2),
            torch.zeros(4, 1),
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
        )
        provider = Provider()
        callback = PerformanceCallback(
            model_flops_per_batch=provider,
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            measure_window_steps=2,
            sync_cuda=False,
        )
        module = Module()
        with TemporaryDirectory() as directory:
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=2,
                accumulate_grad_batches=2,
                callbacks=[callback],
                logger=CSVLogger(directory),
                log_every_n_steps=1,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )
            trainer.fit(module, train_dataloaders=DataLoader(dataset, batch_size=1))

        self.assertEqual(provider.global_steps, [0, 1, 1, 2])
        self.assertEqual(len(module.performance_logs), 2)
        self.assertEqual(module.performance_logs[0]["perf/model_flops_per_step"], 30.0)
        self.assertEqual(module.performance_logs[1]["perf/model_flops_per_step"], 70.0)

    def test_callback_profiles_real_train_batch_flops(self):
        import torch
        from lightning import pytorch as pl
        from torch.utils.data import DataLoader, TensorDataset

        from anytrain.lightning import PerformanceCallback

        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                self.performance_logs = []

            def training_step(self, batch, batch_idx):
                del batch_idx
                inputs, targets = batch
                return torch.nn.functional.mse_loss(self.linear(inputs), targets)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

            def log_dict(self, dictionary, *args, **kwargs):
                if "perf/step_time" in dictionary:
                    self.performance_logs.append(dict(dictionary))
                return super().log_dict(dictionary, *args, **kwargs)

        callback = PerformanceCallback(
            profile_flops=True,
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            sync_cuda=False,
        )
        module = Module()
        dataset = TensorDataset(torch.ones(1, 2), torch.zeros(1, 1))
        with TemporaryDirectory() as directory:
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=1,
                callbacks=[callback],
                default_root_dir=directory,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )
            trainer.fit(module, train_dataloaders=DataLoader(dataset, batch_size=1))

        self.assertEqual(len(module.performance_logs), 1)
        metrics = module.performance_logs[0]
        self.assertGreater(metrics["perf/model_flops_per_step"], 0.0)
        self.assertGreater(metrics["perf/mfu"], 0.0)

    def test_callback_rejects_static_and_dynamic_flops_together(self):
        from anytrain.lightning import PerformanceCallback

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PerformanceCallback(
                model_flops_per_step=100.0,
                model_flops_per_batch=_FlopsProvider(),
            )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PerformanceCallback(
                model_flops_per_step=100.0,
                profile_flops=True,
            )

    def test_callback_rejects_invalid_dynamic_flops(self):
        from anytrain.lightning import PerformanceCallback

        class InvalidProvider:
            def __call__(self, **kwargs):
                del kwargs
                return 0.0

        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[],
            global_step=0,
            world_size=1,
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_batch=InvalidProvider(),
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            sync_cuda=False,
        )
        callback.on_train_start(trainer, module)

        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            trainer.global_step = 1
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                callback.on_train_batch_end(
                    trainer,
                    module,
                    outputs=None,
                    batch=None,
                    batch_idx=0,
                )


if __name__ == "__main__":
    unittest.main()
