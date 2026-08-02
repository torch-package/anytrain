from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from anytrain.lightning import ParameterPolicyCallback, apply_parameter_trainability
from anytrain.lightning import __all__ as lightning_exports
from anytrain.lightning.callback import ParameterPolicyCallback as CallbackExport
from anytrain.lightning.callback import __all__ as callback_exports
from anytrain.lightning.callback import apply_parameter_trainability as apply_export


class ParameterPolicyTest(unittest.TestCase):
    def test_apply_parameter_trainability_freezes_and_summarizes(self):
        module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))

        summary = apply_parameter_trainability(
            module,
            lambda _module, name, _parameter: name.endswith("bias"),
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
                lambda _module, _name, _parameter: 1,
            )

    def test_parameter_policy_callback_applies_once_on_fit_start(self):
        module = torch.nn.Sequential(torch.nn.Linear(1, 1))
        calls = []

        def trainable(_module, name, _parameter):
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

    def test_lightning_exports_parameter_policy_api(self):
        self.assertIn("ParameterPolicyCallback", lightning_exports)
        self.assertIn("apply_parameter_trainability", lightning_exports)
        self.assertIn("ParameterPolicyCallback", callback_exports)
        self.assertIn("apply_parameter_trainability", callback_exports)
        self.assertIs(ParameterPolicyCallback, CallbackExport)
        self.assertIs(apply_parameter_trainability, apply_export)


if __name__ == "__main__":
    unittest.main()
