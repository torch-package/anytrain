import unittest
from dataclasses import fields

import torch

from anytrain.module.dirichlet_tempering import (
    ADT,
    ADTConfig,
    ADTStats,
    AdaptiveDirichletTempering,
    adt_stats_metrics,
    ema_update,
)


class AdaptiveDirichletTemperingTest(unittest.TestCase):
    def test_config_is_plain_data(self):
        config = ADTConfig(num_experts=4)

        field_names = {field.name for field in fields(config)}

        self.assertNotIn("strategy_fn", field_names)
        self.assertNotIn("temperature_fn", field_names)
        self.assertNotIn("prior_mean", field_names)
        self.assertNotIn("strategy", field_names)
        self.assertNotIn("temperature", field_names)
        self.assertNotIn("temperature_transform", field_names)
        self.assertIn("dispersion_strategy", field_names)

    def test_forward_returns_simplex_weights(self):
        adt = AdaptiveDirichletTempering(ADTConfig(num_experts=4))
        logits = torch.randn(8, 4)

        weights = adt(logits)

        self.assertEqual(weights.shape, logits.shape)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-6))
        self.assertTrue(torch.isfinite(weights).all())

    def test_forward_accepts_sequence_logits(self):
        adt = AdaptiveDirichletTempering(ADTConfig(num_experts=4))
        logits = torch.randn(2, 5, 4)

        weights = adt(logits)

        self.assertEqual(weights.shape, logits.shape)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6))

    def test_training_updates_stats(self):
        adt = ADT.from_kwargs(num_experts=3, stat_ema_decays=(0.5, 0.5, 0.5))
        before = adt.expert_means.clone()

        adt(torch.tensor([[3.0, 0.0, -1.0], [2.0, -1.0, 0.5]]))

        self.assertEqual(adt.num_updates, 1)
        self.assertFalse(torch.allclose(adt.expert_means, before))
        self.assertTrue(torch.isfinite(adt.expert_vars).all())

    def test_large_fp16_uniform_logits_accumulate_finite_stats(self):
        self._assert_large_low_precision_stats(
            dtype=torch.float16,
            logits_per_token=torch.zeros(4),
        )

    def test_large_fp16_skewed_logits_accumulate_finite_stats(self):
        self._assert_large_low_precision_stats(
            dtype=torch.float16,
            logits_per_token=torch.tensor([8.0, 0.0, -2.0, -4.0]),
        )

    def test_large_bfloat16_logits_accumulate_finite_stats(self):
        self._assert_large_low_precision_stats(
            dtype=torch.bfloat16,
            logits_per_token=torch.tensor([4.0, 1.0, -1.0, -3.0]),
        )

    def test_eval_does_not_update_stats_by_default(self):
        adt = ADT.from_kwargs(num_experts=3)
        adt.eval()
        before = adt.expert_means.clone()

        adt(torch.randn(4, 3))

        self.assertEqual(adt.num_updates, 0)
        self.assertTrue(torch.allclose(adt.expert_means, before))

    def test_eval_can_update_stats_with_override(self):
        adt = ADT.from_kwargs(num_experts=3)
        adt.eval()

        adt(torch.randn(4, 3), collect_stats=True)

        self.assertEqual(adt.num_updates, 1)

    def test_disable_uses_unit_temperature_tensor(self):
        adt = ADT.from_kwargs(num_experts=3)
        logits = torch.randn(5, 3)

        adt.disable()
        weights = adt(logits)

        self.assertTrue(torch.equal(adt.temperature, torch.ones(3)))
        self.assertTrue(torch.allclose(weights, logits.softmax(dim=-1), atol=1e-6))

    def test_grouped_temperature_shape(self):
        adt = ADT.from_kwargs(num_experts=6, temperature_groups=3)

        temperature = adt.temperature

        self.assertIsInstance(temperature, torch.Tensor)
        self.assertEqual(temperature.shape, (6,))

    def test_one_expert_per_group_temperature_is_finite(self):
        adt = ADT.from_kwargs(num_experts=4, temperature_groups=4)

        temperature = adt.temperature

        self.assertIsInstance(temperature, torch.Tensor)
        self.assertTrue(torch.isfinite(temperature).all())

    def test_gumbel_softmax_accepts_grouped_temperature(self):
        adt = ADT.from_kwargs(
            num_experts=4,
            temperature_groups=2,
            use_gumbel_softmax=True,
        )
        adt.train()
        logits = torch.randn(3, 4)

        weights = adt(logits)

        self.assertEqual(weights.shape, logits.shape)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6))

    def test_temperature_bounds_are_applied(self):
        adt = ADT.from_kwargs(
            num_experts=4,
            min_temperature=0.75,
            max_temperature=1.0,
        )
        adt._expert_means.copy_(torch.tensor([0.97, 0.01, 0.01, 0.01]))
        adt._expert_means_square.copy_(adt._expert_means)

        temperature = adt.temperature

        self.assertTrue((temperature >= 0.75).all())
        self.assertTrue((temperature <= 1.0).all())

    def test_temperature_ema_updates_during_training_forward(self):
        adt = ADT.from_kwargs(
            num_experts=4,
            stat_ema_decays=(0.5, 0.5, 0.5),
            temperature_smoothing_decay=0.5,
        )
        before = adt.temperature.clone()

        adt(torch.tensor([[8.0, -2.0, -2.0, -2.0], [7.0, -1.0, -1.0, -1.0]]))

        self.assertFalse(torch.allclose(adt.temperature, before))
        self.assertTrue(torch.isfinite(adt.temperature).all())

    def test_warmup_uses_unit_temperature_but_collects_stats(self):
        adt = ADT.from_kwargs(
            num_experts=3,
            temperature_warmup_steps=2,
            stat_ema_decays=(0.5, 0.5, 0.5),
        )
        logits = torch.tensor([[4.0, 0.0, -1.0]])

        weights = adt(logits)

        self.assertEqual(adt.num_updates, 1)
        self.assertTrue(torch.allclose(weights, logits.softmax(dim=-1), atol=1e-6))

    def test_num_updates_restores_from_state_dict(self):
        adt = ADT.from_kwargs(
            num_experts=3,
            temperature_warmup_steps=1,
            stat_ema_decays=(0.0, 0.0, 0.0),
        )
        adt(torch.tensor([[8.0, -2.0, -2.0]]))

        restored = ADT.from_kwargs(
            num_experts=3,
            temperature_warmup_steps=1,
            stat_ema_decays=(0.0, 0.0, 0.0),
        )
        restored.load_state_dict(adt.state_dict())

        self.assertEqual(restored.num_updates, 1)
        weights = restored(torch.tensor([[4.0, 0.0, -1.0]]))
        self.assertFalse(torch.allclose(weights, torch.tensor([[4.0, 0.0, -1.0]]).softmax(dim=-1)))

    def test_mask_excludes_invalid_sequence_positions_from_stats(self):
        adt = ADT.from_kwargs(num_experts=3, stat_ema_decays=(0.0, 0.0, 0.0))
        logits = torch.tensor(
            [
                [
                    [8.0, -4.0, -4.0],
                    [-4.0, 8.0, -4.0],
                    [-4.0, -4.0, 8.0],
                ]
            ]
        )
        mask = torch.tensor([[True, False, False]])

        adt(logits, mask=mask)

        expected = logits[:, :1].reshape(-1, 3).softmax(dim=-1).mean(dim=0)
        self.assertTrue(torch.allclose(adt.expert_means, expected, atol=1e-6))

    def test_mask_must_be_bool(self):
        adt = ADT.from_kwargs(num_experts=3)

        with self.assertRaisesRegex(TypeError, "bool"):
            adt(torch.randn(2, 4, 3), mask=torch.ones(2, 4))

    def test_mask_can_broadcast_to_sequence_shape(self):
        adt = ADT.from_kwargs(num_experts=3, stat_ema_decays=(0.0, 0.0, 0.0))
        logits = torch.randn(2, 4, 3)

        adt(logits, mask=torch.tensor([[True, False, True, False]]))

        self.assertEqual(adt.num_updates, 1)

    def test_empty_mask_does_not_update_stats(self):
        adt = ADT.from_kwargs(num_experts=3)
        before = adt.expert_means.clone()

        adt(torch.randn(2, 4, 3), mask=torch.zeros(2, 4, dtype=torch.bool))

        self.assertEqual(adt.num_updates, 0)
        self.assertTrue(torch.allclose(adt.expert_means, before))

    def test_freeze_and_reset_stats(self):
        adt = ADT.from_kwargs(num_experts=3, stat_ema_decays=(0.5, 0.5, 0.5))

        adt.freeze_stats()
        adt(torch.randn(4, 3))
        self.assertEqual(adt.num_updates, 0)

        adt.unfreeze_stats()
        adt(torch.randn(4, 3))
        self.assertEqual(adt.num_updates, 1)

        adt.reset_stats()
        self.assertEqual(adt.num_updates, 0)
        self.assertTrue(torch.allclose(adt.expert_means, torch.full((3,), 1 / 3)))
        self.assertTrue(torch.equal(adt.temperature, torch.ones(3)))

    def test_sync_stats_is_noop_when_distributed_is_uninitialized(self):
        adt = ADT.from_kwargs(num_experts=3, sync_distributed_stats=True)

        adt(torch.randn(4, 3))

        self.assertEqual(adt.num_updates, 1)

    def test_minka_refinement_supports_groups(self):
        adt = ADT.from_kwargs(
            num_experts=4,
            temperature_groups=2,
            minka_refinement_iters=1,
        )

        adt(torch.randn(6, 4))

        self.assertEqual(adt.alpha.shape, (4,))
        self.assertTrue(torch.isfinite(adt.alpha).all())

    def test_diagnostics_contains_expected_tensors(self):
        adt = ADT.from_kwargs(num_experts=4)

        diagnostics = adt.diagnostics()

        self.assertEqual(
            set(diagnostics),
            {
                "effective_experts",
                "expert_means",
                "expert_utilization",
                "expert_vars",
                "alpha",
                "temperature",
            },
        )
        for value in diagnostics.values():
            self.assertIsInstance(value, torch.Tensor)
            self.assertTrue(torch.isfinite(value).all())

        self.assertTrue(torch.allclose(adt.expert_utilization, adt.expert_means))
        self.assertEqual(adt.effective_experts.ndim, 0)
        self.assertGreaterEqual(float(adt.effective_experts), 1.0)
        self.assertLessEqual(float(adt.effective_experts), 4.0)

    def test_stats_object_matches_diagnostics(self):
        adt = ADT.from_kwargs(num_experts=4)

        stats = adt.stats()
        metrics = adt_stats_metrics(stats)

        self.assertIsInstance(stats, ADTStats)
        self.assertEqual(set(metrics), set(adt.diagnostics()))
        for name, value in metrics.items():
            torch.testing.assert_close(value, adt.diagnostics()[name])
        torch.testing.assert_close(stats.expert_means, stats.expert_utilization)
        torch.testing.assert_close(stats.effective_experts, adt.effective_experts)

    def test_stats_metrics_can_select_groups(self):
        stats = ADT.from_kwargs(num_experts=3).stats()

        usage = adt_stats_metrics(stats, metrics=("usage",))
        variance_and_temperature = adt_stats_metrics(
            stats,
            metrics=("variance", "temperature"),
        )

        self.assertEqual(
            set(usage),
            {"effective_experts", "expert_means", "expert_utilization"},
        )
        self.assertEqual(set(variance_and_temperature), {"expert_vars", "temperature"})
        with self.assertRaisesRegex(ValueError, "unknown ADT metrics"):
            adt_stats_metrics(stats, metrics=("loss",))

    def test_stats_effective_experts_ignores_zero_utilization(self):
        stats = ADTStats(
            expert_utilization=torch.tensor([1.0, 0.0, 0.0]),
            expert_vars=torch.zeros(3),
            alpha=torch.ones(3),
            temperature=torch.ones(3),
        )

        torch.testing.assert_close(stats.effective_experts, torch.tensor(1.0))

    def test_stats_exports_from_module_root(self):
        from anytrain import module

        self.assertIs(module.ADTStats, ADTStats)
        self.assertIs(module.adt_stats_metrics, adt_stats_metrics)

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "num_experts"):
            ADTConfig(num_experts=0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ADTConfig(num_experts=5, temperature_groups=2)
        with self.assertRaisesRegex(ValueError, "stat_ema_decays"):
            ADTConfig(num_experts=4, stat_ema_decays=(0.9, 0.9))
        with self.assertRaisesRegex(ValueError, "temperature_warmup"):
            ADTConfig(num_experts=4, temperature_warmup_steps=-1)
        with self.assertRaisesRegex(ValueError, "min_temperature"):
            ADTConfig(num_experts=4, min_temperature=2.0, max_temperature=1.0)
        with self.assertRaisesRegex(ValueError, "temperature_smoothing_decay"):
            ADTConfig(num_experts=4, temperature_smoothing_decay=1.0)

    def test_invalid_logits_shape_fails_clearly(self):
        adt = ADT.from_kwargs(num_experts=4)

        with self.assertRaisesRegex(ValueError, "last logits dimension"):
            adt(torch.randn(2, 3))

    def test_ema_update_requires_one_mode(self):
        x = torch.tensor([1.0])
        y = torch.tensor([3.0])

        self.assertTrue(torch.equal(ema_update(x, y, momentum=0.5), torch.tensor([2.0])))
        self.assertTrue(torch.equal(ema_update(x, y, decay=0.5), torch.tensor([2.0])))
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            ema_update(x, y)
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            ema_update(x, y, momentum=0.5, decay=0.5)

    def _assert_large_low_precision_stats(
        self,
        *,
        dtype: torch.dtype,
        logits_per_token: torch.Tensor,
    ) -> None:
        num_tokens = 70_001
        adt = ADT.from_kwargs(
            num_experts=4,
            stat_ema_decays=(0.0, 0.0, 0.0),
            minka_refinement_iters=1,
        )
        logits = logits_per_token.to(dtype=dtype).expand(num_tokens, -1)

        sums, square_sums, log_sums, count, local_count = adt._router_stat_sums(
            logits,
            mask=None,
        )
        adt.update_stats(logits)

        self.assertEqual(local_count, num_tokens)
        self.assertEqual(count.item(), num_tokens)
        for value in (sums, square_sums, log_sums, count):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())

        expected_means = logits_per_token.softmax(dim=-1)
        self.assertTrue(torch.allclose(adt.expert_means, expected_means, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.all(adt.expert_vars < 2e-6))
        self.assertTrue(torch.isfinite(adt.expert_vars).all())
        self.assertTrue(torch.isfinite(adt.expert_logs).all())
        self.assertTrue(torch.isfinite(adt.temperature).all())


if __name__ == "__main__":
    unittest.main()
