from __future__ import annotations

import unittest
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

from anytrain.lightning import (
    AutoCostHooks,
    CostProbe,
    fit_memory_calibration,
    resolve_auto_cost,
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

    def test_hooks_reject_non_callable_adapters(self) -> None:
        with self.assertRaisesRegex(TypeError, "prepare"):
            AutoCostHooks(
                prepare=cast(Any, None),
                move_to_device=lambda _module, batch, _device: batch,
                run_forward_backward=lambda _module, _model, _batch: None,
            )
        with self.assertRaisesRegex(TypeError, "actual_cost"):
            AutoCostHooks(
                prepare=lambda _module, batch: batch,
                move_to_device=lambda _module, batch, _device: batch,
                run_forward_backward=lambda _module, _model, _batch: None,
                actual_cost=cast(Any, 1),
            )

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

    def test_fit_floors_a_negative_intercept_for_conservative_resolution(self) -> None:
        calibration = fit_memory_calibration(
            [
                (2, 10 * 1024**2),
                (4, 22 * 1024**2),
            ]
        )

        self.assertAlmostEqual(calibration.mib_per_cost, 6.0)
        self.assertEqual(calibration.intercept_mib, 0.0)

    def test_probe_rejects_non_positive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_cost"):
            CostProbe(batch=None, target_cost=0)

    def test_probe_rejects_coercible_but_invalid_metadata(self) -> None:
        with self.assertRaisesRegex(TypeError, "actual_cost"):
            CostProbe(batch=None, target_cost=1, actual_cost=cast(Any, "2"))
        with self.assertRaisesRegex(TypeError, "sample_count"):
            CostProbe(batch=None, target_cost=1, sample_count=True)

    def test_fit_rejects_coercible_point_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "point 0 cost"):
            fit_memory_calibration(cast(Any, [(2.5, 10), (4, 18)]))
        with self.assertRaisesRegex(TypeError, "point 0 peak"):
            fit_memory_calibration(cast(Any, [(2, "10"), (4, 18)]))

    def test_resolve_budget_applies_memory_policy_and_rounding(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(10 * 1024**3, 12 * 1024**3)),
        ):
            budget = resolve_auto_cost(
                device=0,
                memory_fraction=0.5,
                baseline_gib=2.0,
                headroom_gib=1.0,
                mib_per_cost=1.0,
                min_cost=1000,
                concurrent_loaders=2,
            )

        self.assertEqual(budget.raw_cost, 1536)
        self.assertEqual(budget.cost, 1536)
        self.assertEqual(budget.usable_gib, 1.5)

    def test_resolve_rejects_boolean_device_index(self) -> None:
        with self.assertRaisesRegex(TypeError, "device"):
            resolve_auto_cost(device=True, mib_per_cost=1.0)


if __name__ == "__main__":
    unittest.main()
