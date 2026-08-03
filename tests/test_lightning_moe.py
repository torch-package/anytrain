from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from anytrain.lightning import MoELoggerCallback, MoEMonitor
from anytrain.module.dirichlet_tempering import ADT
from anytrain.module.moe import (
    CallableExpert,
    ExpertRouting,
    LinearRouter,
    MoE,
    MoEOutput,
    moe_loss,
)


class _LogModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class _NamedLogModule(_LogModule):
    def __init__(self, named_modules):
        super().__init__()
        self._named_modules = named_modules

    def named_modules(self):
        yield from self._named_modules


class _Writer:
    def __init__(self):
        self.histograms = []

    def add_histogram(self, tag, values, global_step):
        self.histograms.append((tag, values, global_step))


def _scale_expert(input, weight, bias=None):
    del bias
    return input * weight


def _moe_with_adt_gate() -> MoE:
    return MoE(
        num_experts=2,
        router=LinearRouter(2, 2),
        gate=ADT(num_experts=2),
        expert=CallableExpert(_scale_expert),
        weight=nn.Parameter(torch.ones(2)),
        top_k=1,
    )


def _invalid_moe_monitors(outputs, batch, pl_module):
    del outputs, batch, pl_module
    return {"router": object()}


class MoELoggerCallbackTest(unittest.TestCase):
    def test_logs_topk_moe_metrics_and_aux_loss(self):
        routing = ExpertRouting(
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
        aux_loss = moe_loss(output, balance=0.1, z=0.01)
        module = _LogModule()
        callback = MoELoggerCallback(every_n_steps=1)

        callback.on_train_batch_end(
            SimpleNamespace(global_step=4, world_size=1),
            module,
            {"router": output, "router_loss": aux_loss},
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
        torch.testing.assert_close(values["moe/router/tokens_mean"], torch.tensor(1.0))
        torch.testing.assert_close(values["moe/router/gate_probability_mean"], torch.tensor(0.5))
        torch.testing.assert_close(values["moe/router/route_weight_mean"], torch.tensor(1.0 / 3.0))
        torch.testing.assert_close(values["moe/router/dropped/ratio"], torch.tensor(1.0 / 3.0))
        self.assertIn("moe/router/loss/total", values)
        self.assertIn("moe/router/loss/z", values)

    def test_logs_dense_moe_metrics(self):
        routing = ExpertRouting(
            logits=torch.zeros(2, 2),
            probabilities=torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
            top_k=None,
        )
        module = _LogModule()
        callback = MoELoggerCallback(
            metrics=("usage", "gate_probability", "route_weight", "tokens", "dropped"),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            {"router": routing},
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        torch.testing.assert_close(values["moe/router/usage_mean"], torch.tensor(0.5))
        torch.testing.assert_close(values["moe/router/gate_probability_mean"], torch.tensor(0.5))
        torch.testing.assert_close(values["moe/router/route_weight_mean"], torch.tensor(0.5))
        torch.testing.assert_close(values["moe/router/tokens_mean"], torch.tensor(2.0))
        torch.testing.assert_close(values["moe/router/dropped/ratio"], torch.tensor(0.0))

    def test_histogram_logging_uses_summary_metrics_without_scalar_expert_curves(self):
        routing = ExpertRouting(
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
        writer = _Writer()
        module = _LogModule()
        callback = MoELoggerCallback(
            metrics=("usage", "gate_probability", "route_weight", "tokens", "dropped"),
            every_n_steps=1,
            log_histograms=True,
        )

        callback.on_train_batch_end(
            SimpleNamespace(
                global_step=7,
                world_size=1,
                is_global_zero=True,
                logger=SimpleNamespace(experiment=writer),
            ),
            module,
            {"router": output},
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        per_expert_usage = "/".join(("moe", "router", "expert_0", "usage"))
        self.assertNotIn(per_expert_usage, values)
        self.assertEqual(
            [item[0] for item in writer.histograms],
            [
                "moe/router/usage",
                "moe/router/gate_probability",
                "moe/router/route_weight",
                "moe/router/tokens",
            ],
        )
        for _tag, _values, step in writer.histograms:
            self.assertEqual(step, 7)
        torch.testing.assert_close(writer.histograms[0][1], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(writer.histograms[3][1], torch.tensor([1.0, 1.0]))

    def test_auto_logs_adt_gate_stats_from_moe_suffix(self):
        routing = ExpertRouting(
            logits=torch.zeros(1, 2),
            probabilities=torch.full((1, 2), 0.5),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        moe = _moe_with_adt_gate()
        module = _NamedLogModule([("", nn.Module()), ("encoder.router", moe)])
        callback = MoELoggerCallback(
            metrics=("usage", "gate_probability"),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            {"router": MoEMonitor(routing)},
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertIn("moe/router/gate_probability_mean", values)
        self.assertIn("moe/router/adt/temperature_mean", values)
        duplicate_usage = "/".join(("moe", "router", "adt", "usage_mean"))
        duplicate_effective = "/".join(("moe", "router", "adt", "effective_experts"))
        self.assertNotIn(duplicate_usage, values)
        self.assertNotIn(duplicate_effective, values)

    def test_auto_logs_adt_gate_stats_from_explicit_moe_module(self):
        routing = ExpertRouting(
            logits=torch.zeros(1, 2),
            probabilities=torch.full((1, 2), 0.5),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        moe = _moe_with_adt_gate()
        module = _LogModule()
        callback = MoELoggerCallback(
            metrics=("usage",),
            adt_metrics=("temperature",),
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            {"custom": MoEMonitor(routing, module=moe)},
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertIn("moe/custom/adt/temperature_mean", values)

    def test_every_n_steps_logs_once_per_global_step(self):
        routing = ExpertRouting(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        module = _LogModule()
        trainer = SimpleNamespace(global_step=3, world_size=1)
        callback = MoELoggerCallback(every_n_steps=3)

        outputs = {"router": routing}
        callback.on_train_batch_end(trainer, module, outputs, None, 0)
        callback.on_train_batch_end(trainer, module, outputs, None, 1)

        self.assertEqual(len(module.log_dict_calls), 1)

    def test_loss_metric_without_loss_output_is_noop(self):
        routing = ExpertRouting(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        module = _LogModule()
        callback = MoELoggerCallback(metrics=("loss",), every_n_steps=1)

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            {"router": routing},
            None,
            0,
        )

        self.assertEqual(module.log_dict_calls, [])

    def test_callback_requires_explicit_monitor_objects(self):
        callback = MoELoggerCallback(
            _invalid_moe_monitors,
            every_n_steps=1,
        )

        with self.assertRaisesRegex(TypeError, "MoEMonitor"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_callback_rejects_monitor_names_that_normalize_to_same_tag(self):
        routing = ExpertRouting(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        callback = MoELoggerCallback(
            lambda outputs, batch, pl_module: {
                "router": MoEMonitor(routing),
                "/router/": MoEMonitor(routing),
            },
            every_n_steps=1,
        )

        with self.assertRaisesRegex(ValueError, "normalize to the same tag"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_callback_rejects_non_string_monitor_names(self):
        routing = ExpertRouting(
            logits=torch.zeros(1, 1),
            probabilities=torch.ones(1, 1),
            indices=torch.zeros(1, 1, dtype=torch.long),
            weights=torch.ones(1, 1),
        )
        monitors = cast(Any, {1: MoEMonitor(routing)})
        callback = MoELoggerCallback(
            lambda outputs, batch, pl_module: monitors,
            every_n_steps=1,
        )

        with self.assertRaisesRegex(TypeError, "monitor name must be a string"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_callback_rejects_adt_gate_usage_metric(self):
        with self.assertRaisesRegex(ValueError, "usage"):
            MoELoggerCallback(adt_metrics=("us" + "age",))

    def test_callback_exports_from_lightning_root(self):
        from anytrain import lightning

        self.assertIs(lightning.MoELoggerCallback, MoELoggerCallback)
        self.assertIs(lightning.MoEMonitor, MoEMonitor)


if __name__ == "__main__":
    unittest.main()
