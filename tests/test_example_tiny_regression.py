import unittest
from typing import Any, cast


class TinyRegressionExampleTest(unittest.TestCase):
    def test_run_tiny_regression(self):
        from examples.tiny_regression import run_tiny_regression

        trainer, module = run_tiny_regression(
            num_samples=8,
            batch_size=4,
            enable_progress_bar=False,
        )

        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(str(trainer.default_root_dir), "outputs/anytrain/tiny_regression")
        self.assertEqual(module.model.in_features, 4)
        self.assertIsNotNone(module.optimizer)

    def test_example_controls_and_datamodule_lifecycle_are_explicit(self):
        from torch import nn

        from examples.tiny_regression import (
            TinyRegressionDataModule,
            TinyRegressionModule,
            run_tiny_regression,
        )

        with self.assertRaisesRegex(RuntimeError, "setup"):
            TinyRegressionDataModule().train_dataloader()

        invalid_calls = (
            lambda: TinyRegressionModule(nn.Linear(1, 1), lr=float("nan")),
            lambda: TinyRegressionDataModule(num_samples=cast(Any, True)),
            lambda: run_tiny_regression(max_epochs=cast(Any, True)),
            lambda: run_tiny_regression(default_root_dir=cast(Any, 1)),
            lambda: run_tiny_regression(default_root_dir="  "),
            lambda: run_tiny_regression(enable_progress_bar=cast(Any, 1)),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call), self.assertRaises((TypeError, ValueError)):
                invalid_call()

        data = TinyRegressionDataModule(num_samples=2, input_dim=1, batch_size=1)
        data.setup()
        self.assertEqual(len(data.train_dataloader()), 2)


if __name__ == "__main__":
    unittest.main()
