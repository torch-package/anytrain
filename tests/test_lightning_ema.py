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
        expected = ema.shadow["weight"].detach().clone()
        state = ema.state_dict()
        saved_weight = state["shadow"]["weight"]

        # Match nn.Module.state_dict(): state entries are detached shallow
        # references, so collecting callback state does not clone the full EMA.
        self.assertFalse(saved_weight.requires_grad)
        self.assertEqual(saved_weight.data_ptr(), ema.shadow["weight"].data_ptr())

        restored = ParameterEMA(module, decay=0.9)
        restored_storage = restored.shadow["weight"].data_ptr()
        restored.load_state_dict(state)
        self.assertEqual(restored.num_updates, 1)
        self.assertEqual(restored.shadow["weight"].data_ptr(), restored_storage)
        self.assertTrue(torch.equal(restored.shadow["weight"], expected))

        ema.shadow["weight"].add_(10.0)
        self.assertTrue(torch.equal(restored.shadow["weight"], expected))

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

    def test_exception_restores_each_eval_stage_swap(self) -> None:
        for stage in ("validation", "test", "predict"):
            with self.subTest(stage=stage):
                module = _LinearModule()
                callback = EMACallback(decay=0.5)
                trainer = SimpleNamespace(global_step=0, callbacks=[callback])
                callback.on_fit_start(trainer, module)  # type: ignore[arg-type]
                assert callback.ema is not None
                callback.ema.shadow["weight"].fill_(2.0)
                module.weight.data.fill_(5.0)

                start = getattr(callback, f"on_{stage}_start")
                start(trainer, module)
                self.assertTrue(torch.equal(module.weight, torch.full((2, 2), 2.0)))

                callback.on_exception(
                    trainer,  # type: ignore[arg-type]
                    module,  # type: ignore[arg-type]
                    RuntimeError("evaluation failed"),
                )

                self.assertTrue(torch.equal(module.weight, torch.full((2, 2), 5.0)))
                self.assertFalse(callback._active)
                self.assertIsNone(callback.ema._backup)

    def test_enter_ema_restores_training_weights_when_copy_fails(self) -> None:
        from unittest.mock import patch

        module = _LinearModule()
        callback = EMACallback(decay=0.5)
        trainer = SimpleNamespace(global_step=0, callbacks=[callback])
        callback.on_fit_start(trainer, module)  # type: ignore[arg-type]
        assert callback.ema is not None
        module.weight.data.fill_(5.0)

        def fail_after_partial_copy(target: nn.Module) -> None:
            with torch.no_grad():
                target.get_parameter("weight").fill_(2.0)
            raise RuntimeError("copy failed")

        with (
            patch.object(callback.ema, "copy_to", side_effect=fail_after_partial_copy),
            self.assertRaisesRegex(RuntimeError, "copy failed"),
        ):
            callback.on_validation_start(trainer, module)  # type: ignore[arg-type]

        self.assertTrue(torch.equal(module.weight, torch.full((2, 2), 5.0)))
        self.assertFalse(callback._active)
        self.assertIsNone(callback.ema._backup)

    def test_load_state_dict_before_on_fit_start_defers_ema(self) -> None:
        module = _LinearModule()
        source = EMACallback(decay=0.5)
        trainer = SimpleNamespace(global_step=0, callbacks=[source])
        source.on_fit_start(trainer, module)  # type: ignore[arg-type]
        module.weight.data.fill_(3.0)
        trainer.global_step = 1
        source.on_train_batch_end(trainer, module, None, None, 0)  # type: ignore[arg-type]
        state = source.state_dict()
        assert source.ema is not None

        restored = EMACallback(decay=0.5)
        restored.load_state_dict(state)
        self.assertIsNone(restored.ema)
        self.assertEqual(restored._last_global_step, 1)

        trainer = SimpleNamespace(global_step=0, callbacks=[restored])
        restored.on_fit_start(trainer, module)  # type: ignore[arg-type]
        assert restored.ema is not None
        self.assertEqual(restored.ema.num_updates, source.ema.num_updates)
        self.assertTrue(
            torch.equal(restored.ema.shadow["weight"], source.ema.shadow["weight"])
        )
        self.assertEqual(restored._last_global_step, 1)
        self.assertIsNone(restored._pending_ema)

    def test_trainer_checkpoint_roundtrip_restores_ema(self) -> None:
        import tempfile
        from pathlib import Path

        from lightning.pytorch import LightningModule, Trainer
        from torch.utils.data import DataLoader, TensorDataset

        class Tiny(LightningModule):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Linear(2, 2)

            def training_step(self, batch, batch_idx):
                del batch_idx
                x, y = batch
                return nn.functional.mse_loss(self.net(x), y)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        dataset = TensorDataset(torch.randn(8, 2), torch.randn(8, 2))
        loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ema.ckpt"
            source_callback = EMACallback(decay=0.9)
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=1,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[source_callback],
            )
            trainer.fit(Tiny(), loader)
            trainer.save_checkpoint(ckpt)
            assert source_callback.ema is not None
            expected_updates = source_callback.ema.num_updates
            expected_weight = source_callback.ema.shadow["net.weight"].detach().cpu().clone()

            restored_callback = EMACallback(decay=0.9)
            restored = Trainer(
                accelerator="cpu",
                max_epochs=1,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[restored_callback],
            )
            restored.fit(Tiny(), loader, ckpt_path=str(ckpt))
            assert restored_callback.ema is not None
            self.assertEqual(restored_callback.ema.num_updates, expected_updates)
            self.assertTrue(
                torch.equal(
                    restored_callback.ema.shadow["net.weight"].detach().cpu(),
                    expected_weight,
                )
            )

    def test_validation_exception_restores_training_weights_before_checkpoint(self) -> None:
        import tempfile
        from pathlib import Path

        from lightning.pytorch import LightningModule, Trainer
        from torch.utils.data import DataLoader, TensorDataset

        from anytrain.lightning import ModelCheckpoint

        class FailingValidationModule(LightningModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.training_weight: torch.Tensor | None = None
                self.validation_weight: torch.Tensor | None = None

            def training_step(self, batch, batch_idx):
                del batch_idx
                (target,) = batch
                return (self.weight - target).square().mean()

            def on_train_batch_end(self, outputs, batch, batch_idx):
                del outputs, batch, batch_idx
                self.training_weight = self.weight.detach().clone()

            def validation_step(self, batch, batch_idx):
                del batch, batch_idx
                self.validation_weight = self.weight.detach().clone()
                raise RuntimeError("validation failed")

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        dataset = TensorDataset(torch.ones(2, 1))
        loader = DataLoader(dataset, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            ema = EMACallback(decay=0.5)
            checkpoint = ModelCheckpoint(
                dirpath=tmp,
                filename="exception",
                save_on_exception=True,
            )
            module = FailingValidationModule()
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=1,
                num_sanity_val_steps=0,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[checkpoint, ema],
            )
            self.assertLess(trainer.callbacks.index(ema), trainer.callbacks.index(checkpoint))

            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                trainer.fit(module, loader, loader)

            assert module.training_weight is not None
            assert module.validation_weight is not None
            assert ema.ema is not None
            self.assertFalse(torch.equal(module.training_weight, module.validation_weight))
            self.assertTrue(torch.equal(module.weight.detach(), module.training_weight))
            self.assertTrue(torch.equal(ema.ema.shadow["weight"], module.validation_weight))
            self.assertFalse(ema._active)
            self.assertIsNone(ema.ema._backup)

            checkpoint_path = Path(tmp) / "exception.ckpt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(saved["state_dict"]["weight"], module.training_weight))
            saved_ema = saved["callbacks"][ema.state_key]["ema"]
            self.assertEqual(saved_ema["num_updates"], ema.ema.num_updates)
            self.assertTrue(
                torch.equal(saved_ema["shadow"]["weight"], ema.ema.shadow["weight"])
            )


if __name__ == "__main__":
    unittest.main()
