import unittest
from types import SimpleNamespace


class LightningCapabilityTest(unittest.TestCase):
    def test_logger_experiment_prefers_primary_logger_then_loggers(self):
        from anytrain.lightning import _capability

        primary = object()
        fallback = object()

        trainer = SimpleNamespace(
            logger=SimpleNamespace(experiment=primary),
            loggers=[SimpleNamespace(experiment=fallback)],
        )
        self.assertIs(_capability.logger_experiment(trainer), primary)

        trainer = SimpleNamespace(logger=None, loggers=[SimpleNamespace(experiment=fallback)])
        self.assertIs(_capability.logger_experiment(trainer), fallback)

        self.assertIsNone(_capability.logger_experiment(SimpleNamespace(loggers=[])))

    def test_strategy_reduce_has_explicit_missing_capability_error(self):
        import torch

        from anytrain.lightning import _capability

        with self.assertRaisesRegex(RuntimeError, "custom reduce error"):
            _capability.strategy_reduce(
                SimpleNamespace(strategy=object()),
                torch.tensor([1.0]),
                reduce_op="sum",
                error="custom reduce error",
            )
        self.assertIsNone(
            _capability.strategy_reduce_or_none(
                SimpleNamespace(strategy=object()),
                torch.tensor([1.0]),
                reduce_op="sum",
            )
        )

    def test_strategy_reduce_and_barrier_delegate_to_strategy(self):
        import torch

        from anytrain.lightning import _capability

        class Strategy:
            def __init__(self):
                self.barrier_count = 0
                self.reduce_ops = []

            def reduce(self, tensor, *, reduce_op):
                self.reduce_ops.append(reduce_op)
                return tensor + 1

            def barrier(self):
                self.barrier_count += 1

        strategy = Strategy()
        trainer = SimpleNamespace(strategy=strategy)

        result = _capability.strategy_reduce(trainer, torch.tensor([1.0]), reduce_op="sum")
        self.assertEqual(result.tolist(), [2.0])
        self.assertEqual(strategy.reduce_ops, ["sum"])

        _capability.strategy_barrier(trainer)
        self.assertEqual(strategy.barrier_count, 1)

    def test_log_hyperparams_skips_loggers_without_support(self):
        from anytrain.lightning import _capability

        class Logger:
            def __init__(self):
                self.values = []

            def log_hyperparams(self, params):
                self.values.append(params)

        logger = Logger()
        metadata = {"source": "test"}

        _capability.log_hyperparams(
            SimpleNamespace(loggers=[object(), logger]),
            metadata,
        )

        self.assertEqual(logger.values, [metadata])


if __name__ == "__main__":
    unittest.main()
