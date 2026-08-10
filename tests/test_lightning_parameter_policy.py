from __future__ import annotations

import unittest
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast

import torch

from anytrain.lightning import ParameterPolicyCallback, apply_parameter_trainability
from anytrain.lightning import __all__ as lightning_exports
from anytrain.lightning.callback import ParameterPolicyCallback as CallbackExport
from anytrain.lightning.callback import __all__ as callback_exports
from anytrain.lightning.callback import apply_parameter_trainability as apply_export


def SimpleNamespace(**kwargs: object) -> Any:
    return _SimpleNamespace(**kwargs)


class ParameterPolicyTest(unittest.TestCase):
    def test_apply_parameter_trainability_freezes_and_summarizes(self):
        module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))

        summary = apply_parameter_trainability(
            module,
            lambda module, name, parameter: name.endswith("bias"),
        )

        self.assertEqual(summary.tensors, 4)
        self.assertEqual(summary.trainable_tensors, 2)
        self.assertEqual(summary.frozen_tensors, 2)
        self.assertEqual(summary.changed_tensors, 2)
        self.assertEqual(summary.parameters, 13)
        self.assertEqual(summary.trainable_parameters, 4)
        self.assertEqual(summary.frozen_parameters, 9)
        for name, parameter in module.named_parameters():
            self.assertEqual(parameter.requires_grad, name.endswith("bias"))

    def test_apply_parameter_trainability_rejects_non_bool_return(self):
        with self.assertRaisesRegex(TypeError, "boolean"):
            apply_parameter_trainability(
                torch.nn.Linear(1, 1),
                lambda module, name, parameter: cast(Any, 1),
            )

    def test_parameter_policy_callback_applies_once_on_fit_start(self):
        module: Any = torch.nn.Sequential(torch.nn.Linear(1, 1))
        calls = []

        def trainable(module, name, parameter):
            del module, parameter
            calls.append(name)
            return False

        callback = ParameterPolicyCallback(trainable)
        callback.on_fit_start(SimpleNamespace(), module)
        callback.on_fit_start(SimpleNamespace(), module)

        self.assertTrue(callback.applied)
        self.assertEqual(calls, ["0.weight", "0.bias"])
        self.assertIsNotNone(callback.summary)
        self.assertFalse(module[0].weight.requires_grad)
        self.assertFalse(module[0].bias.requires_grad)

    def test_parameter_policy_callback_setup_runs_before_optimizer(self):
        from lightning.pytorch import LightningModule, Trainer
        from torch.utils.data import DataLoader, TensorDataset

        class Tiny(LightningModule):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Linear(2, 2)
                self.net.bias.requires_grad_(False)
                self.opt_names: list[str] | None = None

            def training_step(self, batch, batch_idx):
                del batch_idx
                x, y = batch
                return torch.nn.functional.mse_loss(self.net(x), y)

            def configure_optimizers(self):
                trainable = [
                    (name, parameter)
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                ]
                self.opt_names = [name for name, _ in trainable]
                return torch.optim.SGD([parameter for _, parameter in trainable], lr=0.1)

        def trainable(module, name, parameter):
            del module, parameter
            return name.endswith("bias")

        model = Tiny()
        callback = ParameterPolicyCallback(trainable)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 2), torch.randn(8, 2)),
            batch_size=4,
        )
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[callback],
        )
        trainer.fit(model, loader)

        self.assertTrue(callback.applied)
        self.assertEqual(model.opt_names, ["net.bias"])
        self.assertFalse(model.net.weight.requires_grad)
        self.assertTrue(model.net.bias.requires_grad)
        optimizer = trainer.optimizers[0]
        opt_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        self.assertNotIn(id(model.net.weight), opt_ids)
        self.assertIn(id(model.net.bias), opt_ids)

    def test_lightning_exports_parameter_policy_api(self):
        self.assertIn("ParameterPolicyCallback", lightning_exports)
        self.assertIn("apply_parameter_trainability", lightning_exports)
        self.assertIn("ParameterPolicyCallback", callback_exports)
        self.assertIn("apply_parameter_trainability", callback_exports)
        self.assertIs(ParameterPolicyCallback, CallbackExport)
        self.assertIs(apply_parameter_trainability, apply_export)


if __name__ == "__main__":
    unittest.main()
