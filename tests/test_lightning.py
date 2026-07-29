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
            "loss": torch.tensor(2.0),
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
                ("token/loss/asr", 3.0),
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

    def test_gradient_norm_logger_respects_step_interval(self):
        import torch

        from anytrain.lightning import GradientNormLoggerCallback

        class Logger:
            def __init__(self):
                self.log_calls = []

            def log(self, name, value, **kwargs):
                self.log_calls.append((name, float(value), kwargs))

        weight = torch.nn.Parameter(torch.zeros(2))
        bias = torch.nn.Parameter(torch.zeros(1))
        weight.grad = torch.tensor([3.0, 4.0])
        bias.grad = torch.tensor([12.0])
        logger = Logger()
        module = SimpleNamespace(
            parameters=lambda: iter((weight, bias)),
            log=logger.log,
        )
        trainer = SimpleNamespace(global_step=1)
        callback = GradientNormLoggerCallback(every_n_steps=2)
        optimizer = torch.optim.SGD([weight], lr=0.1)

        callback.on_before_optimizer_step(trainer, module, optimizer)

        self.assertEqual(logger.log_calls, [])

        trainer.global_step = 2
        callback.on_before_optimizer_step(trainer, module, optimizer)

        self.assertEqual(logger.log_calls[0][0], "train/grad_norm")
        self.assertEqual(logger.log_calls[0][1], 13.0)

    def test_gradient_logger_compares_loss_item_gradients(self):
        import math

        import torch

        from anytrain.lightning import GradientLoggerCallback
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
        callback = GradientLoggerCallback(("token", "flow"), "weight", every_n_steps=2)

        callback.on_before_backward(trainer, module, torch.tensor(0.0))
        self.assertEqual(module.log_calls, [])

        trainer.global_step = 2
        callback.on_before_backward(trainer, module, torch.tensor(0.0))

        values = {name: value for name, value, _ in module.log_calls}
        self.assertAlmostEqual(values["grad/token_flow/norm/token"], 4.0)
        self.assertAlmostEqual(values["grad/token_flow/norm/flow"], 36.0)
        self.assertAlmostEqual(values["grad/token_flow/log_ratio"], math.log(4.0 / 36.0))
        self.assertAlmostEqual(values["grad/token_flow/cosine"], 1.0)

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

        self.assertEqual(list(custom.parameters)[:-1], list(original.parameters))
        self.assertEqual(custom.parameters["async_save"].default, True)
        self.assertEqual(custom.parameters["async_save"].kind, inspect.Parameter.KEYWORD_ONLY)

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


if __name__ == "__main__":
    unittest.main()
