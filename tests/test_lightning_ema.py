from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from anytrain.lightning import EMACallback, ParameterEMA, find_ema_callback


class _LinearModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.bias = nn.Parameter(torch.zeros(2))
        self.register_buffer("not_trained", torch.ones(2))


class ParameterEMATests(unittest.TestCase):
    def test_update_and_copy_to_averaged_weights(self) -> None:
        module = _LinearModule()
        ema = ParameterEMA(module, decay=0.5)
        module.weight.data.fill_(3.0)
        ema.update(module)
        module.weight.data.fill_(5.0)
        ema.update(module)

        with ema.average_parameters(module):
            self.assertTrue(torch.allclose(module.weight, torch.full((2, 2), 4.0)))
        self.assertTrue(torch.allclose(module.weight, torch.full((2, 2), 5.0)))

    def test_state_dict_roundtrip(self) -> None:
        module = _LinearModule()
        ema = ParameterEMA(module, decay=0.9)
        module.weight.data.add_(1.0)
        ema.update(module)
        state = ema.state_dict()

        restored = ParameterEMA(module, decay=0.9)
        restored.load_state_dict(state)
        self.assertEqual(restored.num_updates, 1)
        self.assertTrue(torch.equal(restored.shadow["weight"], ema.shadow["weight"]))

    def test_rejects_nested_store(self) -> None:
        module = _LinearModule()
        ema = ParameterEMA(module, decay=0.9)
        ema.store(module)
        with self.assertRaisesRegex(RuntimeError, "backup is active"):
            ema.store(module)
        ema.restore(module)


class EMACallbackTests(unittest.TestCase):
    def test_updates_when_global_step_advances(self) -> None:
        module = _LinearModule()
        callback = EMACallback(decay=0.5)
        trainer = SimpleNamespace(global_step=0, callbacks=[callback])
        callback.on_fit_start(trainer, module)  # type: ignore[arg-type]
        assert callback.ema is not None

        module.weight.data.fill_(3.0)
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)  # type: ignore[arg-type]
        module.weight.data.fill_(5.0)
        trainer.global_step = 2
        callback.on_train_batch_end(trainer, module, None, None, 1)  # type: ignore[arg-type]

        callback.on_validation_start(trainer, module)  # type: ignore[arg-type]
        self.assertTrue(torch.allclose(module.weight, torch.full((2, 2), 4.0)))
        callback.on_validation_end(trainer, module)  # type: ignore[arg-type]
        self.assertTrue(torch.allclose(module.weight, torch.full((2, 2), 5.0)))

    def test_find_ema_callback(self) -> None:
        callback = EMACallback(decay=0.999)
        trainer = SimpleNamespace(callbacks=[object(), callback])
        self.assertIs(find_ema_callback(trainer), callback)  # type: ignore[arg-type]

    def test_average_parameters_context(self) -> None:
        module = _LinearModule()
        callback = EMACallback(decay=0.0)
        trainer = SimpleNamespace(global_step=0, callbacks=[callback])
        callback.on_fit_start(trainer, module)  # type: ignore[arg-type]
        module.weight.data.fill_(7.0)
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)  # type: ignore[arg-type]
        module.weight.data.fill_(1.0)

        with callback.average_parameters(module):
            self.assertTrue(torch.allclose(module.weight, torch.full((2, 2), 7.0)))
        self.assertTrue(torch.allclose(module.weight, torch.ones(2, 2)))


if __name__ == "__main__":
    unittest.main()
