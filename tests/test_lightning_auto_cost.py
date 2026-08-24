from __future__ import annotations

import unittest

import torch
from torch import nn

from anytrain.lightning import (
    AutoCostHooks,
    CostProbe,
    fit_memory_calibration,
)


class _TinyModel(nn.Module):
    pass


class AutoCostTest(unittest.TestCase):
    def test_probe_preserves_project_batch_without_inspection(self) -> None:
        batch = {"tokens": [1, 2, 3]}
        probe = CostProbe(
            batch=batch,
            target_cost=8,
            actual_cost=6,
            padded_cost=9,
            sample_count=3,
        )

        self.assertIs(probe.batch, batch)
        self.assertEqual(probe.target_cost, 8)
        self.assertEqual(probe.actual_cost, 6)

    def test_hooks_are_batch_and_module_agnostic(self) -> None:
        hooks: AutoCostHooks[tuple[str, ...], _TinyModel, str] = AutoCostHooks(
            prepare=lambda module, batch: batch + (module,),
            move_to_device=lambda _module, batch, device: batch + (str(device),),
            run_forward_backward=lambda _module, _model, _batch: None,
        )

        prepared = hooks.prepare("pl", ("batch",))
        moved = hooks.move_to_device("pl", prepared, torch.device("cpu"))

        self.assertEqual(moved, ("batch", "pl", "cpu"))

    def test_fit_uses_arbitrary_cost_units(self) -> None:
        calibration = fit_memory_calibration(
            [
                (2, 10 * 1024**2),
                (4, 18 * 1024**2),
                (6, 26 * 1024**2),
            ]
        )

        self.assertAlmostEqual(calibration.mib_per_cost, 4.0)
        self.assertAlmostEqual(calibration.intercept_mib, 2.0)

    def test_probe_rejects_non_positive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_cost"):
            CostProbe(batch=None, target_cost=0)


if __name__ == "__main__":
    unittest.main()
