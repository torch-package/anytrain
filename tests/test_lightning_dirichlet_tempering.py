import unittest
from types import SimpleNamespace

import torch

from anytrain.lightning import ADTStatsLoggerCallback, ADTStatsMonitor
from anytrain.module.dirichlet_tempering import ADT, ADTStats


class _LogModule:
    def __init__(self):
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class _NamedLogModule(_LogModule):
    def __init__(self, modules):
        super().__init__()
        self._modules = modules

    def named_modules(self):
        return iter(self._modules)


class _Writer:
    def __init__(self):
        self.histograms = []

    def add_histogram(self, tag, values, global_step):
        self.histograms.append((tag, values, global_step))


class ADTStatsLoggerCallbackTest(unittest.TestCase):
    def test_auto_discovers_adt_modules_by_default(self):
        adt = ADT(num_experts=2)
        module = _NamedLogModule([("encoder.router.gate", adt)])
        callback = ADTStatsLoggerCallback(metrics=("temperature",), every_n_steps=1)

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        self.assertEqual(len(module.log_dict_calls), 1)
        values, _ = module.log_dict_calls[0]
        self.assertIn("adt/gate/temperature_mean", values)

    def test_auto_discovery_uses_shortest_unique_module_suffixes(self):
        encoder_adt = ADT(num_experts=2)
        decoder_adt = ADT(num_experts=2)
        module = _NamedLogModule(
            [
                ("encoder.router.gate", encoder_adt),
                ("decoder.router.gate", decoder_adt),
            ]
        )
        callback = ADTStatsLoggerCallback(metrics=("temperature",), every_n_steps=1)

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        self.assertEqual(len(module.log_dict_calls), 1)
        values, _ = module.log_dict_calls[0]
        self.assertIn("adt/encoder.router.gate/temperature_mean", values)
        self.assertIn("adt/decoder.router.gate/temperature_mean", values)
        self.assertNotIn("adt/gate/temperature_mean", values)

    def test_auto_discovery_noops_without_adt_modules(self):
        module = _NamedLogModule([("linear", torch.nn.Linear(2, 2))])
        callback = ADTStatsLoggerCallback(metrics=("temperature",), every_n_steps=1)

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        self.assertEqual(module.log_dict_calls, [])

    def test_logs_scalar_metrics_from_explicit_stats_monitor(self):
        stats = ADTStats(
            expert_utilization=torch.tensor([0.75, 0.25]),
            expert_vars=torch.tensor([0.01, 0.02]),
            alpha=torch.tensor([3.0, 4.0]),
            temperature=torch.tensor([1.1, 1.2]),
        )
        module = _LogModule()
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: {"router": ADTStatsMonitor(stats)},
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=4, world_size=1),
            module,
            None,
            None,
            0,
        )

        self.assertEqual(len(module.log_dict_calls), 1)
        values, kwargs = module.log_dict_calls[0]
        self.assertEqual(kwargs["on_step"], True)
        self.assertEqual(kwargs["on_epoch"], False)
        self.assertEqual(kwargs["logger"], True)
        self.assertEqual(kwargs["sync_dist"], False)
        self.assertNotIn("adt/router/usage_mean", values)
        self.assertNotIn("adt/router/effective_experts", values)
        torch.testing.assert_close(values["adt/router/variance_max"], torch.tensor(0.02))
        torch.testing.assert_close(values["adt/router/alpha_mean"], torch.tensor(3.5))
        torch.testing.assert_close(
            values["adt/router/temperature_max"],
            torch.tensor(1.2),
        )

    def test_monitor_can_read_live_adt_module_stats(self):
        adt = ADT(num_experts=2)
        module = _LogModule()
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: {"gate": ADTStatsMonitor(adt)},
            metrics=("temperature",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertEqual(
            set(values),
            {
                "adt/gate/temperature_mean",
                "adt/gate/temperature_std",
                "adt/gate/temperature_min",
                "adt/gate/temperature_max",
            },
        )

    def test_infers_name_from_returned_adt_module_path(self):
        adt = ADT(num_experts=2)
        module = _NamedLogModule([("encoder.router.gate", adt)])
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: adt,
            metrics=("temperature",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertIn("adt/gate/temperature_mean", values)

    def test_infers_name_from_returned_adt_monitor_path(self):
        adt = ADT(num_experts=2)
        module = _NamedLogModule([("encoder.router.gate", adt)])
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: ADTStatsMonitor(adt),
            metrics=("temperature",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertIn("adt/gate/temperature_mean", values)

    def test_monitor_name_overrides_inferred_module_path(self):
        adt = ADT(num_experts=2)
        module = _NamedLogModule([("encoder.router.gate", adt)])
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: ADTStatsMonitor(adt, name="primary_gate"),
            metrics=("temperature",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertIn("adt/primary_gate/temperature_mean", values)
        self.assertNotIn("adt/gate/temperature_mean", values)

    def test_stats_without_explicit_name_uses_stable_fallback(self):
        stats = ADTStats(
            expert_utilization=torch.ones(1),
            expert_vars=torch.zeros(1),
            alpha=torch.ones(1),
            temperature=torch.ones(1),
        )
        module = _LogModule()
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: stats,
            metrics=("temperature",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertIn("adt/monitor_0/temperature_mean", values)

    def test_histogram_logging_uses_global_zero_writer(self):
        writer = _Writer()
        stats = ADTStats(
            expert_utilization=torch.tensor([0.75, 0.25]),
            expert_vars=torch.tensor([0.01, 0.02]),
            alpha=torch.tensor([3.0, 4.0]),
            temperature=torch.tensor([1.1, 1.2]),
        )
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: {"router": ADTStatsMonitor(stats)},
            metrics=("variance", "temperature"),
            every_n_steps=1,
            log_histograms=True,
        )

        callback.on_train_batch_end(
            SimpleNamespace(
                global_step=9,
                world_size=1,
                is_global_zero=True,
                logger=SimpleNamespace(experiment=writer),
            ),
            _LogModule(),
            None,
            None,
            0,
        )

        self.assertEqual(len(writer.histograms), 2)
        self.assertEqual(writer.histograms[0][0], "adt/router/variance")
        self.assertEqual(writer.histograms[0][2], 9)
        torch.testing.assert_close(writer.histograms[0][1], torch.tensor([0.01, 0.02]))
        self.assertEqual(writer.histograms[1][0], "adt/router/temperature")
        self.assertEqual(writer.histograms[1][2], 9)

    def test_every_n_steps_logs_once_per_global_step(self):
        stats = ADTStats(
            expert_utilization=torch.ones(1),
            expert_vars=torch.zeros(1),
            alpha=torch.ones(1),
            temperature=torch.ones(1),
        )
        module = _LogModule()
        trainer = SimpleNamespace(global_step=3, world_size=1)
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: {"router": ADTStatsMonitor(stats)},
            every_n_steps=3,
        )

        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_train_batch_end(trainer, module, None, None, 1)

        self.assertEqual(len(module.log_dict_calls), 1)

    def test_callback_rejects_invalid_monitor_objects(self):
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: {"router": object()},
            every_n_steps=1,
        )

        with self.assertRaisesRegex(TypeError, "ADTStatsMonitor"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_callback_rejects_gate_usage_metric(self):
        with self.assertRaisesRegex(ValueError, "usage"):
            ADTStatsLoggerCallback(metrics=("usage",))

    def test_callback_exports_from_lightning_root(self):
        from anytrain import lightning

        self.assertIs(lightning.ADTStatsLoggerCallback, ADTStatsLoggerCallback)
        self.assertIs(lightning.ADTStatsMonitor, ADTStatsMonitor)


if __name__ == "__main__":
    unittest.main()
