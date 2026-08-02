import unittest
from types import SimpleNamespace

import torch

from anytrain.lightning import (
    CodebookUsageLoggerCallback,
    CodebookUsageMonitor,
    QuantizationLoggerCallback,
    QuantizationMonitor,
)
from anytrain.module.quantization import QuantizationLoss, QuantizeOutput


class _LogModule:
    def __init__(self):
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class _Writer:
    def __init__(self):
        self.histograms = []

    def add_histogram(self, tag, values, global_step):
        self.histograms.append((tag, values, global_step))


class QuantizationLoggerCallbackTest(unittest.TestCase):
    def test_codebook_usage_callback_logs_raw_index_distribution(self):
        module = _LogModule()
        callback = CodebookUsageLoggerCallback(
            lambda outputs, batch, pl_module: {
                "target": CodebookUsageMonitor(
                    indices=torch.tensor([[0, 1], [1, -1]]),
                    codebook_sizes=(3, 3),
                    active_codebook_mask=torch.tensor([[True, True], [True, False]]),
                )
            },
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
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
        self.assertEqual(values["codebook/target/book_0/usage/observations"].item(), 2.0)
        self.assertEqual(values["codebook/target/book_1/usage/observations"].item(), 1.0)
        self.assertEqual(values["codebook/target/book_1/active/ratio"].item(), 0.5)

    def test_codebook_usage_callback_rejects_quantization_only_metrics(self):
        with self.assertRaisesRegex(ValueError, "unknown codebook usage metrics"):
            CodebookUsageLoggerCallback(
                lambda outputs, batch, pl_module: {},
                metrics=("loss",),
            )

    def test_codebook_usage_callback_requires_usage_monitors(self):
        callback = CodebookUsageLoggerCallback(
            lambda outputs, batch, pl_module: {"target": object()},
            every_n_steps=1,
        )

        with self.assertRaisesRegex(TypeError, "CodebookUsageMonitor"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_logs_loss_usage_and_active_metrics_from_explicit_monitors(self):
        output = QuantizeOutput(
            quantized_latents=torch.empty(2, 3),
            indices=torch.tensor([[0, 1], [2, -1]]),
            loss=QuantizationLoss(
                commitment=torch.tensor(0.25),
                codebook=torch.tensor(0.75),
            ),
            active_codebook_mask=torch.tensor([[True, True], [True, False]]),
        )
        module = _LogModule()
        callback = QuantizationLoggerCallback(
            lambda outputs, batch, pl_module: {
                "rvq": QuantizationMonitor(
                    output=output,
                    quantizer=SimpleNamespace(codebook_sizes=(3, 3)),
                )
            },
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=4, world_size=1, is_global_zero=True),
            module,
            {"rvq": output},
            None,
            0,
        )

        self.assertEqual(len(module.log_dict_calls), 1)
        values, kwargs = module.log_dict_calls[0]
        self.assertEqual(kwargs["on_step"], True)
        self.assertEqual(kwargs["on_epoch"], False)
        self.assertEqual(kwargs["logger"], True)
        self.assertEqual(kwargs["sync_dist"], False)
        self.assertEqual(values["quant/rvq/loss/total"].item(), 1.0)
        self.assertEqual(values["quant/rvq/book_0/usage/observations"].item(), 2.0)
        self.assertEqual(values["quant/rvq/book_1/usage/observations"].item(), 1.0)
        self.assertEqual(values["quant/rvq/book_1/active/ratio"].item(), 0.5)
        torch.testing.assert_close(
            values["quant/rvq/book_1/usage/dead_ratio"],
            torch.tensor(2.0 / 3.0),
        )

    def test_every_n_steps_logs_once_per_global_step(self):
        output = QuantizeOutput(
            quantized_latents=torch.empty(1),
            indices=torch.tensor([0]),
        )
        module = _LogModule()
        trainer = SimpleNamespace(global_step=3, world_size=1)
        callback = QuantizationLoggerCallback(
            lambda outputs, batch, pl_module: {
                "vq": QuantizationMonitor(output=output, codebook_sizes=(2,))
            },
            every_n_steps=3,
        )

        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_train_batch_end(trainer, module, None, None, 1)

        self.assertEqual(len(module.log_dict_calls), 1)

    def test_distributed_usage_reduces_counts_before_ratios(self):
        class Strategy:
            def __init__(self):
                self.ops = []

            def reduce(self, tensor, *, reduce_op):
                self.ops.append((tuple(tensor.shape), reduce_op))
                if reduce_op == "sum":
                    return tensor * 2
                return tensor + 1

        output = QuantizeOutput(
            quantized_latents=torch.empty(2),
            indices=torch.tensor([0, 1]),
            loss=QuantizationLoss(
                commitment=torch.tensor(0.25),
                codebook=torch.tensor(0.75),
            ),
        )
        module = _LogModule()
        strategy = Strategy()
        callback = QuantizationLoggerCallback(
            lambda outputs, batch, pl_module: {
                "vq": QuantizationMonitor(output=output, codebook_sizes=(4,))
            },
            every_n_steps=1,
        )

        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=2, strategy=strategy),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertEqual(values["quant/vq/loss/total"].item(), 2.0)
        self.assertEqual(values["quant/vq/book_0/usage/observations"].item(), 4.0)
        self.assertEqual(values["quant/vq/book_0/usage/unique_ratio"].item(), 0.5)
        self.assertEqual(
            strategy.ops,
            [
                ((), "mean"),
                ((), "mean"),
                ((), "mean"),
                ((1, 4), "sum"),
                ((1,), "sum"),
                ((1,), "sum"),
            ],
        )

    def test_monitor_supports_heterogeneous_codebook_sizes(self):
        output = QuantizeOutput(
            quantized_latents=torch.empty(1, 2),
            indices=torch.tensor([[1, 3]]),
        )
        callback = QuantizationLoggerCallback(
            lambda outputs, batch, pl_module: {
                "gvq": QuantizationMonitor(
                    output=output,
                    quantizer=SimpleNamespace(codebook_sizes=(2, 4)),
                )
            },
            every_n_steps=1,
        )

        module = _LogModule()
        callback.on_train_batch_end(
            SimpleNamespace(global_step=0, world_size=1),
            module,
            None,
            None,
            0,
        )

        values, _ = module.log_dict_calls[0]
        self.assertEqual(values["quant/gvq/book_0/usage/unique_ratio"].item(), 0.5)
        self.assertEqual(values["quant/gvq/book_1/usage/unique_ratio"].item(), 0.25)

    def test_histogram_logging_uses_global_zero_writer(self):
        writer = _Writer()
        output = QuantizeOutput(
            quantized_latents=torch.empty(2),
            indices=torch.tensor([0, 1]),
        )
        callback = QuantizationLoggerCallback(
            lambda outputs, batch, pl_module: {
                "vq": QuantizationMonitor(output=output, codebook_sizes=(4,))
            },
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

        self.assertEqual(len(writer.histograms), 1)
        self.assertEqual(writer.histograms[0][0], "quant/vq/book_0/indices")
        self.assertEqual(writer.histograms[0][2], 9)

    def test_callback_requires_explicit_monitor_objects(self):
        callback = QuantizationLoggerCallback(
            lambda outputs, batch, pl_module: {"vq": object()},
            every_n_steps=1,
        )

        with self.assertRaisesRegex(TypeError, "QuantizationMonitor"):
            callback.on_train_batch_end(
                SimpleNamespace(global_step=0, world_size=1),
                _LogModule(),
                None,
                None,
                0,
            )

    def test_callback_exports_from_lightning_root(self):
        from anytrain import lightning
        from anytrain.lightning import callback

        self.assertIs(lightning.CodebookUsageLoggerCallback, CodebookUsageLoggerCallback)
        self.assertIs(lightning.CodebookUsageMonitor, CodebookUsageMonitor)
        self.assertIs(lightning.QuantizationLoggerCallback, QuantizationLoggerCallback)
        self.assertIs(lightning.QuantizationMonitor, QuantizationMonitor)
        self.assertIs(callback.CodebookUsageLoggerCallback, CodebookUsageLoggerCallback)
        self.assertIs(callback.CodebookUsageMonitor, CodebookUsageMonitor)
        self.assertIs(callback.QuantizationLoggerCallback, QuantizationLoggerCallback)
        self.assertIs(callback.QuantizationMonitor, QuantizationMonitor)


if __name__ == "__main__":
    unittest.main()
