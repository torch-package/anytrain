import math
import sys
import unittest
from importlib import import_module, util
from typing import Any, cast

import torch
from torch import nn

FLOW_MATCHING_AVAILABLE = util.find_spec("flow_matching") is not None


class TensorStatsTest(unittest.TestCase):
    def test_time_bucketed_mean_returns_ddp_friendly_sum_and_count(self):
        from anytrain.stats import time_bucketed_mean

        bucketed = time_bucketed_mean(
            torch.tensor([1.0, 3.0, 5.0, 7.0]),
            torch.tensor([0.0, 0.2, 0.6, 1.0]),
            bucket_count=2,
        )

        torch.testing.assert_close(bucketed.total, torch.tensor([4.0, 12.0]))
        torch.testing.assert_close(bucketed.count, torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(bucketed.mean, torch.tensor([2.0, 6.0]))
        self.assertTrue(torch.equal(bucketed.populated, torch.tensor([True, True])))

    def test_time_bucketed_mean_validates_public_inputs(self):
        from anytrain.stats import time_bucketed_mean

        with self.assertRaisesRegex(ValueError, "same shape"):
            time_bucketed_mean(torch.ones(2), torch.ones(2, 1), bucket_count=2)
        with self.assertRaisesRegex(ValueError, "within"):
            time_bucketed_mean(
                torch.ones(1),
                torch.tensor([1.1]),
                bucket_count=2,
            )
        invalid_calls = (
            lambda: time_bucketed_mean(torch.ones(1), torch.zeros(1), bucket_count=True),
            lambda: time_bucketed_mean(torch.ones(1), torch.zeros(1), bucket_count=1, t_min=True),
            lambda: time_bucketed_mean(
                torch.ones(1),
                torch.zeros(1),
                bucket_count=1,
                t_max=float("inf"),
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call), self.assertRaises((TypeError, ValueError)):
                invalid_call()


class FlowMatchingImportTest(unittest.TestCase):
    def test_framework_import_does_not_require_flow_matching_extra(self):
        framework = import_module("anytrain.framework")

        self.assertEqual(framework.__all__, [])

    def test_flow_matching_import_requires_optional_dependency_when_missing(self):
        if FLOW_MATCHING_AVAILABLE:
            self.skipTest("flow_matching is installed")

        sys.modules.pop("anytrain.framework.flow_matching", None)
        with self.assertRaisesRegex(ImportError, r"anytrain\[flow\]"):
            import_module("anytrain.framework.flow_matching")


@unittest.skipUnless(FLOW_MATCHING_AVAILABLE, "flow_matching extra is not installed")
class FlowMatchingComponentTest(unittest.TestCase):
    def test_flow_matching_does_not_export_tensor_stats(self):
        flow_matching = import_module("anytrain.framework.flow_matching")

        self.assertNotIn("time_bucketed_mean", flow_matching.__all__)
        self.assertNotIn("TimeBucketedMean", flow_matching.__all__)
        self.assertFalse(hasattr(flow_matching, "time_bucketed_mean"))
        self.assertFalse(hasattr(flow_matching, "TimeBucketedMean"))

    def test_objectives_register_scalar_loss_observations(self):
        from anytrain import observation
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            ContinuousVelocityObjective,
            DiscreteFlowRuntime,
            DiscreteGeneralizedKLObjective,
        )

        objectives = (
            ContinuousVelocityObjective(ContinuousFlowRuntime()),
            DiscreteGeneralizedKLObjective(DiscreteFlowRuntime(8)),
        )
        for objective in objectives:
            specs = observation.registry.specs(objective)

            self.assertEqual(len(specs), 1)
            self.assertIsInstance(specs[0], observation.OutputObservation)
            self.assertEqual(specs[0].name, "loss")
            self.assertTrue(specs[0].recommended)

    def test_sources_and_time_sampler(self):
        from anytrain.framework.flow_matching import (
            DiscreteEulerSampler,
            GaussianSource,
            LogitNormalTimeSampler,
            MaskTokenSource,
            ODESampler,
            UniformSource,
            UniformTimeSampler,
            UniformTokenSource,
        )

        x = torch.zeros(2, 3)

        self.assertEqual(GaussianSource().sample_like(x).shape, x.shape)
        self.assertEqual(UniformSource(low=-1, high=1).sample_like(x).shape, x.shape)
        self.assertTrue((UniformTokenSource(5).sample_like(x) < 5).all())
        self.assertTrue(
            torch.equal(MaskTokenSource(7).sample_like(x), torch.full_like(x, 7).long())
        )

        logit = LogitNormalTimeSampler()
        t = logit.sample(1024, x.device)
        self.assertEqual(t.shape, (1024,))
        self.assertTrue((t >= logit.t_min).all())
        self.assertTrue((t <= logit.t_max).all())

        t = UniformTimeSampler(t_min=0.2, t_max=0.3).sample(4, x.device)
        self.assertEqual(t.shape, (4,))
        self.assertTrue((t >= 0.2).all())
        self.assertTrue((t <= 0.3).all())

        invalid_calls = (
            lambda: UniformSource(low=float("nan")),
            lambda: UniformTokenSource(True),
            lambda: MaskTokenSource(False),
            lambda: LogitNormalTimeSampler(std=float("inf")),
            lambda: UniformTimeSampler(t_min=cast(Any, "0")),
            lambda: ODESampler(nfe=True),
            lambda: ODESampler(num_steps=cast(Any, 2.5)),
            lambda: ODESampler(return_intermediates=cast(Any, 1)),
            lambda: DiscreteEulerSampler(eps=float("nan")),
            lambda: DiscreteEulerSampler(verbose=cast(Any, 0)),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call), self.assertRaises((TypeError, ValueError)):
                invalid_call()

        with self.assertRaises(TypeError):
            UniformTimeSampler().sample(True, x.device)

    def test_discrete_runtime_rejects_mask_id_outside_vocab(self):
        from anytrain.framework.flow_matching import DiscreteFlowRuntime, MaskTokenSource

        with self.assertRaisesRegex(ValueError, "mask_id must be in"):
            DiscreteFlowRuntime(vocab_size=8, source=MaskTokenSource(mask_id=8))

    def test_flow_matching_install_hint_mentions_extra(self):
        from anytrain.framework.flow_matching._deps import INSTALL_HINT

        self.assertRegex(INSTALL_HINT, r"anytrain\[flow\]")

    def test_discrete_mask_token_objective_and_sample(self):
        from anytrain.framework.flow_matching import (
            DiscreteFlowRuntime,
            DiscreteGeneralizedKLObjective,
            MaskTokenSource,
        )

        class ToyTokenFlow(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 8)
                self.head = nn.Linear(8, vocab_size)

            def forward(self, x_t, t, condition=None):
                del condition
                scale = t.view(t.shape[0], *([1] * (x_t.ndim - 1)), 1)
                return self.head(self.embedding(x_t.long()) + scale)

        vocab_size = 8
        mask_id = 7
        model = ToyTokenFlow(vocab_size)
        runtime = DiscreteFlowRuntime(
            vocab_size,
            source=MaskTokenSource(mask_id=mask_id),
        )
        objective = DiscreteGeneralizedKLObjective(runtime)
        x_1 = torch.randint(0, mask_id, (4, 5))

        loss = objective(model, x_1)
        loss.backward()
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

        output = runtime.sample(model, runtime.source_like(x_1))
        self.assertEqual(output.final.shape, x_1.shape)

    def test_default_time_sampler_is_uniform(self):
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            DiscreteFlowRuntime,
            DiscreteGeneralizedKLObjective,
            UniformTimeSampler,
        )

        discrete = DiscreteFlowRuntime(6)
        self.assertIsInstance(ContinuousFlowRuntime().time_sampler, UniformTimeSampler)
        self.assertIsInstance(discrete.time_sampler, UniformTimeSampler)
        self.assertIs(DiscreteGeneralizedKLObjective(discrete).runtime, discrete)
        continuous_sampler: Any = ContinuousFlowRuntime().time_sampler
        discrete_sampler: Any = discrete.time_sampler
        self.assertEqual(continuous_sampler.t_min, 0.0)
        self.assertEqual(continuous_sampler.t_max, 1.0)
        self.assertEqual(discrete_sampler.t_min, 0.0)
        self.assertLess(discrete_sampler.t_max, 1.0)

    def test_continuous_loss_backward_and_sample_shape(self):
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            ContinuousVelocityObjective,
            ODESampler,
        )

        class ToyVelocity(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 2)

            def forward(self, x_t, t, condition=None):
                del condition
                scale = t.view(t.shape[0], *([1] * (x_t.ndim - 1)))
                return self.proj(x_t) + scale

        model = ToyVelocity()
        sampler = ODESampler()
        runtime = ContinuousFlowRuntime(sampler=sampler)
        objective = ContinuousVelocityObjective(runtime=runtime)
        x_1 = torch.randn(4, 3, 2)

        loss = objective(model, x_1, condition=torch.zeros(4, 1))
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.proj.weight.grad)

        self.assertIs(runtime.sampler, sampler)
        output = runtime.sample(
            model,
            torch.randn(4, 3, 2),
            condition=torch.zeros(4, 1),
        )
        self.assertEqual(output.final.shape, x_1.shape)
        self.assertIsNone(output.states)
        self.assertIsNotNone(output.time_grid)

    def test_continuous_loss_accepts_custom_masked_loss(self):
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            ContinuousVelocityObjective,
            masked_mse_velocity_loss,
        )

        class ZeroVelocity(nn.Module):
            def forward(self, x_t, t, mask=None):
                del t, mask
                return torch.zeros_like(x_t)

        objective = ContinuousVelocityObjective(
            ContinuousFlowRuntime(),
            loss_fn=masked_mse_velocity_loss,
        )
        x_1 = torch.tensor(
            [
                [[1.0, 2.0], [10.0, 20.0]],
                [[3.0, 4.0], [30.0, 40.0]],
            ]
        )
        x_0 = torch.zeros_like(x_1)
        mask = torch.tensor([[True, False], [True, False]])

        loss = objective(ZeroVelocity(), x_1, x_0=x_0, mask=mask)

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).square().mean()
        self.assertTrue(torch.equal(loss, expected))

    def test_continuous_training_sample_preserves_input_dtype(self):
        from anytrain.framework.flow_matching import ContinuousFlowRuntime

        runtime = ContinuousFlowRuntime()
        x_1 = torch.randn(2, 3, dtype=torch.bfloat16)

        sample = runtime.training_sample(x_1)

        self.assertEqual(sample.x_t.dtype, x_1.dtype)
        self.assertEqual(sample.velocity.dtype, x_1.dtype)
        self.assertEqual(sample.t.dtype, x_1.dtype)

        with self.assertRaisesRegex(TypeError, "same dtype"):
            runtime.training_sample(x_1, x_0=torch.zeros_like(x_1, dtype=torch.float32))

        with self.assertRaisesRegex(TypeError, "x_1 must be a tensor"):
            runtime.training_sample(cast(Any, [1, 2]))
        with self.assertRaisesRegex(TypeError, "x_0 and x_1 must be tensors"):
            runtime.training_sample(x_1, x_0=cast(Any, [1, 2]))

    def test_continuous_custom_loss_must_return_scalar(self):
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            ContinuousVelocityObjective,
        )

        class ZeroVelocity(nn.Module):
            def forward(self, x_t, t):
                del t
                return torch.zeros_like(x_t)

        def vector_loss(prediction, target, extras):
            del extras
            return (prediction - target).square().mean(dim=-1)

        objective = ContinuousVelocityObjective(
            ContinuousFlowRuntime(),
            loss_fn=vector_loss,
        )

        with self.assertRaisesRegex(ValueError, "scalar"):
            objective(ZeroVelocity(), torch.randn(2, 3, 4))

    def test_continuous_runtime_uses_model_caller_and_expands_scalar_time(self):
        from anytrain.framework.flow_matching import ContinuousFlowRuntime, ODESampler

        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.t_shape = None

            def forward(self, x_t, t):
                self.t_shape = tuple(t.shape)
                return x_t

        class FakeSolver:
            def __init__(self, model):
                self.model = model

            def sample(self, x_init, **kwargs):
                del kwargs
                return self.model(x=x_init, t=torch.tensor(0.5, device=x_init.device))

        calls = []

        def call_model(model, x_t, t, extras):
            calls.append(extras)
            return model(x_t, t)

        model = RecordingModel()
        sampler = ODESampler(solver_factory=cast(Any, FakeSolver))
        runtime = ContinuousFlowRuntime(sampler=sampler, call_model=call_model)
        output = runtime.sample(model, torch.zeros(4, 3), condition=torch.ones(4, 1))

        self.assertEqual(model.t_shape, (4,))
        self.assertEqual(len(calls), 1)
        self.assertIn("condition", calls[0])
        self.assertEqual(output.final.shape, (4, 3))

    def test_ode_sampler_default_time_grid_matches_state_dtype(self):
        from anytrain.framework.flow_matching import ODESampler

        def run(dtype):
            model_dtypes = []
            solver_dtypes = []
            solvers = []

            class RecordingModel(nn.Module):
                def forward(self, x_t, t):
                    model_dtypes.append(t.dtype)
                    return torch.zeros_like(x_t)

            class RecordingSolver:
                def __init__(self, model):
                    self.model = model

                def sample(self, x_init, **kwargs):
                    solver_dtypes.append(kwargs["time_grid"].dtype)
                    return self.model(x=x_init, t=kwargs["time_grid"][0])

            def factory(model):
                solver = RecordingSolver(model)
                solvers.append(solver)
                return solver

            x_0 = torch.zeros(2, 3, dtype=dtype)
            output = ODESampler(solver_factory=cast(Any, factory)).sample(
                RecordingModel(),
                x_0,
            )

            assert output.time_grid is not None
            self.assertEqual(output.time_grid.dtype, dtype)
            self.assertEqual(solver_dtypes, [dtype])
            self.assertEqual(model_dtypes, [dtype])
            self.assertEqual(len(solvers), 1)

        for dtype in (torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                run(dtype)

    def test_discrete_loss_backward_and_sample_shape(self):
        from anytrain.framework.flow_matching import (
            DiscreteFlowRuntime,
            DiscreteGeneralizedKLObjective,
        )

        class ToyTokenFlow(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 8)
                self.head = nn.Linear(8, vocab_size)

            def forward(self, x_t, t, condition=None):
                del condition
                scale = t.view(t.shape[0], *([1] * (x_t.ndim - 1)), 1)
                return self.head(self.embedding(x_t.long()) + scale)

        vocab_size = 6
        model = ToyTokenFlow(vocab_size)
        runtime = DiscreteFlowRuntime(vocab_size)
        objective = DiscreteGeneralizedKLObjective(runtime)
        x_1 = torch.randint(0, vocab_size, (4, 5))

        loss = objective(model, x_1, condition=torch.zeros(4, 1))
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.head.weight.grad)

        output = runtime.sample(model, torch.randint(0, vocab_size, (4, 5)))
        self.assertEqual(output.final.shape, x_1.shape)
        self.assertIsNone(output.states)
        self.assertIsNotNone(output.time_grid)

    def test_discrete_inputs_must_be_long(self):
        from anytrain.framework.flow_matching import (
            DiscreteFlowRuntime,
            DiscreteGeneralizedKLObjective,
        )

        runtime = DiscreteFlowRuntime(6)
        objective = DiscreteGeneralizedKLObjective(runtime)

        with self.assertRaisesRegex(TypeError, "x_1"):
            objective(nn.Identity(), torch.randn(2, 3))
        with self.assertRaisesRegex(TypeError, "x_0"):
            objective(
                nn.Identity(),
                torch.zeros(2, 3, dtype=torch.long),
                x_0=torch.randn(2, 3),
            )
        with self.assertRaisesRegex(TypeError, "x_0"):
            runtime.sample(nn.Identity(), torch.randn(2, 3))

    def test_gaussian_log_prob_reduces_over_non_batch_axes(self):
        from anytrain.framework.flow_matching import gaussian_log_prob

        value = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 1.5], [2.5, 4.0]]])
        mean = torch.zeros_like(value)
        std = torch.full_like(value, 2.0)

        elementwise = (
            -0.5 * ((value - mean) / std).square()
            - torch.log(std)
            - 0.5 * value.new_tensor(math.log(2.0 * math.pi))
        )

        torch.testing.assert_close(
            gaussian_log_prob(value, mean, std, reduction="mean"),
            elementwise.mean(dim=(1, 2)),
        )
        torch.testing.assert_close(
            gaussian_log_prob(value, mean, std, reduction="sum"),
            elementwise.sum(dim=(1, 2)),
        )

    def test_gaussian_kl_from_means_handles_equal_variance(self):
        from anytrain.framework.flow_matching import gaussian_kl_from_means

        policy_mean = torch.tensor([[[0.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
        reference_mean = torch.tensor([[[0.0, 0.0]], [[1.0, 4.0]]])
        std = torch.full_like(policy_mean, 2.0)

        kl = gaussian_kl_from_means(policy_mean, reference_mean, std, reduction="mean")
        expected = ((policy_mean - reference_mean).square() / (2.0 * std.square())).mean(dim=(1, 2))

        torch.testing.assert_close(kl, expected)
        torch.testing.assert_close(
            gaussian_kl_from_means(reference_mean, reference_mean, std),
            torch.zeros(2),
        )
        kl.sum().backward()
        self.assertIsNotNone(policy_mean.grad)

    def test_rectified_flow_sde_step_matches_fixed_next_x_log_prob_and_kl(self):
        from anytrain.framework.flow_matching import (
            RectifiedFlowSDE,
            gaussian_kl_from_means,
            gaussian_log_prob,
        )

        sde = RectifiedFlowSDE(noise_level=0.7, log_prob_reduction="mean")
        x_t = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        velocity = torch.tensor([[0.2, -0.1], [0.0, 0.3]], requires_grad=True)
        t = torch.tensor([0.8, 0.6])
        next_t = torch.tensor([0.7, 0.4])
        next_x = torch.tensor([[0.95, -1.05], [0.55, 0.2]])

        transition = sde.step_with_log_prob(
            x_t,
            t,
            next_t,
            velocity,
            next_x=next_x,
        )

        self.assertEqual(transition.next_x.shape, x_t.shape)
        self.assertEqual(transition.mean.shape, x_t.shape)
        self.assertEqual(transition.step_std.shape, (2, 1))
        self.assertEqual(transition.log_prob.shape, (2,))
        torch.testing.assert_close(transition.next_x, next_x)
        torch.testing.assert_close(
            transition.log_prob,
            gaussian_log_prob(
                next_x,
                transition.mean,
                transition.step_std,
                reduction="mean",
            ),
        )
        kl = gaussian_kl_from_means(
            transition.mean,
            transition.mean.detach(),
            transition.step_std,
            reduction="mean",
        )
        self.assertEqual(kl.shape, (2,))
        torch.testing.assert_close(kl, torch.zeros(2))
        transition.log_prob.sum().backward()
        self.assertIsNotNone(velocity.grad)

    def test_rectified_flow_sde_requires_reverse_time(self):
        from anytrain.framework.flow_matching import RectifiedFlowSDE

        sde = RectifiedFlowSDE()
        x_t = torch.zeros(2, 3)

        with self.assertRaisesRegex(ValueError, "strictly less"):
            sde.step_with_log_prob(
                x_t,
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.4, 0.6]),
                torch.zeros_like(x_t),
            )

    def test_rectified_flow_sde_rejects_invalid_noise_and_eps(self):
        from anytrain.framework.flow_matching import RectifiedFlowSDE

        for noise_level in (0.0, float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(noise_level=noise_level),
                self.assertRaisesRegex(
                    ValueError,
                    "noise_level",
                ),
            ):
                RectifiedFlowSDE(noise_level=noise_level)

        for eps in (0.0, 0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "eps"):
                RectifiedFlowSDE(eps=eps)

    def test_rectified_flow_sde_detaches_next_x_but_keeps_velocity_gradient(self):
        from anytrain.framework.flow_matching import RectifiedFlowSDE

        sde = RectifiedFlowSDE()
        x_t = torch.tensor([[1.0, -1.0]], requires_grad=True)
        velocity = torch.tensor([[0.2, -0.1]], requires_grad=True)
        next_x = torch.tensor([[0.9, -1.1]], requires_grad=True)

        transition = sde.step_with_log_prob(
            x_t,
            torch.tensor([0.8]),
            torch.tensor([0.6]),
            velocity,
            next_x=next_x,
        )

        transition.log_prob.sum().backward()

        self.assertIsNotNone(velocity.grad)
        self.assertIsNone(next_x.grad)


if __name__ == "__main__":
    unittest.main()


def test_masked_velocity_loss_excludes_padding_before_arithmetic():
    from anytrain.framework.flow_matching import masked_mse_velocity_loss

    prediction = torch.tensor([[[1.0], [float("nan")], [float("inf")]]], requires_grad=True)
    target = torch.tensor([[[0.0], [float("inf")], [float("nan")]]], requires_grad=True)
    mask = torch.tensor([[True, False, False]])
    result = masked_mse_velocity_loss(prediction, target, {"mask": mask})
    torch.testing.assert_close(result, torch.tensor(1.0))
    result.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[2.0], [0.0], [0.0]]]))
    torch.testing.assert_close(target.grad, torch.tensor([[[-2.0], [0.0], [0.0]]]))
