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


class ADTStatsLoggerCallbackTest(unittest.TestCase):
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
        self.assertEqual(values["adt/router/expert_0/usage"].item(), 0.75)
        torch.testing.assert_close(values["adt/router/expert_1/variance"], torch.tensor(0.02))
        self.assertEqual(values["adt/router/expert_0/alpha"].item(), 3.0)
        torch.testing.assert_close(values["adt/router/expert_1/temperature"], torch.tensor(1.2))
        self.assertIn("adt/router/effective_experts", values)

    def test_monitor_can_read_live_adt_module_stats(self):
        adt = ADT.from_kwargs(num_experts=2)
        module = _LogModule()
        callback = ADTStatsLoggerCallback(
            lambda outputs, batch, pl_module: {"gate": ADTStatsMonitor(adt)},
            metrics=("usage",),
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
                "adt/gate/effective_experts",
                "adt/gate/expert_0/usage",
                "adt/gate/expert_1/usage",
            },
        )

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

    def test_callback_requires_explicit_monitor_objects(self):
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

    def test_callback_exports_from_lightning_root(self):
        from anytrain import lightning

        self.assertIs(lightning.ADTStatsLoggerCallback, ADTStatsLoggerCallback)
        self.assertIs(lightning.ADTStatsMonitor, ADTStatsMonitor)


if __name__ == "__main__":
    unittest.main()
