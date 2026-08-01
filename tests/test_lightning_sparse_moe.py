from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from anytrain.lightning import SparseMoELoggerCallback, SparseMoEMonitor
from anytrain.module.moe import MoEOutput, RouterOutput, sparse_moe_aux_loss


class _LogModule:
    def __init__(self):
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class SparseMoELoggerCallbackTest(unittest.TestCase):
    def test_logs_sparse_moe_metrics_and_aux_loss(self):
        routing = RouterOutput(
            logits=torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
            probabilities=torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]),
            indices=torch.tensor([[0], [1], [0]]),
            weights=torch.tensor([[1.0], [1.0], [0.0]]),
        )
        output = MoEOutput(
            hidden_states=torch.zeros(3, 4),
            routing=routing,
            tokens_per_expert=torch.tensor([1, 1]),
            dropped_mask=torch.tensor([False, False, True]),
        )
        aux_loss = sparse_moe_aux_loss(output, load_balance_weight=0.1, router_z_weight=0.01)
        module = _LogModule()
        callback = SparseMoELoggerCallback(
            lambda outputs, batch, pl_module: {
                "router": SparseMoEMonitor(output, loss=aux_loss),
            },
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
        self.assertIn("moe/router/effective_experts", values)
        torch.testing.assert_close(values["moe/router/expert_0/tokens"], torch.tensor(1.0))
        torch.testing.assert_close(values["moe/router/expert_1/probability"], torch.tensor(0.5))
        torch.testing.assert_close(values["moe/router/dropped/ratio"], torch.tensor(1.0 / 3.0))
        self.assertIn("moe/router/loss/total", values)
        self.assertIn("moe/router/loss/router_z", values)

    def test_every_n_steps_logs_once_per_global_step(self):
        routing = RouterOutput(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        module = _LogModule()
        trainer = SimpleNamespace(global_step=3, world_size=1)
        callback = SparseMoELoggerCallback(
            lambda outputs, batch, pl_module: {"router": SparseMoEMonitor(routing)},
            every_n_steps=3,
        )

        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_train_batch_end(trainer, module, None, None, 1)

        self.assertEqual(len(module.log_dict_calls), 1)

    def test_loss_metric_without_loss_output_is_noop(self):
        routing = RouterOutput(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        module = _LogModule()
        callback = SparseMoELoggerCallback(
            lambda outputs, batch, pl_module: {"router": SparseMoEMonitor(routing)},
            metrics=("loss",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        self.assertEqual(module.log_dict_calls, [])

    def test_callback_requires_explicit_monitor_objects(self):
        callback = SparseMoELoggerCallback(
            lambda outputs, batch, pl_module: {"router": object()},
            every_n_steps=1,
        )

        with self.assertRaisesRegex(TypeError, "SparseMoEMonitor"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_callback_exports_from_lightning_root(self):
        from anytrain import lightning

        self.assertIs(lightning.SparseMoELoggerCallback, SparseMoELoggerCallback)
        self.assertIs(lightning.SparseMoEMonitor, SparseMoEMonitor)


if __name__ == "__main__":
    unittest.main()
