import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from lightning import pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import anytrain.lightning.callback.observation as observation_callback
from anytrain import observation
from anytrain.lightning import ObservationCallback
from anytrain.module import AdaptiveDirichletTempering
from anytrain.module.moe import CallableExpert, LinearRouter, MoE
from anytrain.module.quantization import (
    EmbeddingVectorQuantizer,
    ResidualVectorQuantizer,
    RVQConfig,
    VQConfig,
)


class _Writer:
    def __init__(self):
        self.histograms = []

    def add_histogram_raw(self, tag, **kwargs):
        self.histograms.append((tag, kwargs))


class _Observed(nn.Module):
    def __init__(self, *, sizes=(2,)):
        super().__init__()
        self.sizes = sizes
        observation.registry.recommend(self)

    def forward(self, values):
        return values

    @staticmethod
    def _collect(module, inputs, kwargs, output):
        del inputs, kwargs
        if not isinstance(module, _Observed):
            raise TypeError("test observation requires _Observed.")
        return {
            "codes": observation.Bins(
                output,
                sizes=module.sizes,
                group_dim=None if len(module.sizes) == 1 else -1,
                group_names=tuple(
                    f"book_{index}" for index in range(len(module.sizes))
                ),
            ),
            "mean": observation.Curve(output.float().mean()),
        }


observation.registry.register(
    _Observed,
    (
        observation.ForwardObservation(
            "stats",
            _Observed._collect,
            history=observation.EMAHistory(half_life_steps=1),
            recommended=True,
            covers_descendants=True,
        ),
    ),
)


class _Module(pl.LightningModule):
    def __init__(self, observed=None):
        super().__init__()
        self.observed = _Observed() if observed is None else observed
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


def _trainer(
    writer,
    *,
    step=0,
    world_size=1,
    strategy=None,
    is_global_zero=True,
):
    return SimpleNamespace(
        global_step=step,
        world_size=world_size,
        is_global_zero=is_global_zero,
        strategy=strategy,
        logger=SimpleNamespace(experiment=writer),
    )


def _run_step(callback, trainer, module, values, *, batch_idx):
    callback.on_train_batch_start(trainer, module, None, batch_idx)
    module.observed(values)
    trainer.global_step += 1
    callback.on_train_batch_end(trainer, module, None, None, batch_idx)


def _start(callback, trainer, module):
    callback.on_fit_start(trainer, module)
    callback.on_train_start(trainer, module)


class ObservationCallbackTest(unittest.TestCase):
    def test_cadence_only_controls_render_and_ema_includes_skipped_step(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module()
        callback = ObservationCallback(every_n_steps=2, half_life_steps=1)
        _start(callback, trainer, module)

        _run_step(callback, trainer, module, torch.tensor([0, 0]), batch_idx=0)
        self.assertEqual(writer.histograms, [])
        _run_step(callback, trainer, module, torch.tensor([1, 1, 1, 1]), batch_idx=1)

        ema = next(
            kwargs
            for tag, kwargs in writer.histograms
            if tag == "module/observed/stats/codes/book_0/ema"
        )
        self.assertEqual(ema["bucket_counts"], [1.0, 2.0])
        values, kwargs = module.log_dict_calls[-1]
        self.assertEqual(kwargs["sync_dist"], False)
        self.assertEqual(values["module/observed/stats/mean"].item(), 1.0)
        self.assertEqual(
            values[
                "module/observed/stats/codes/book_0/step/occupied_count"
            ].item(),
            1.0,
        )
        self.assertEqual(
            values[
                "module/observed/stats/codes/book_0/ema/occupied_count"
            ].item(),
            2.0,
        )

    def test_observation_time_is_amortized_over_the_render_interval(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module()
        callback = ObservationCallback(every_n_steps=2)
        _start(callback, trainer, module)

        with mock.patch.object(
            observation_callback.time,
            "perf_counter",
            side_effect=(0.0, 0.1, 1.0, 1.2, 2.0, 2.3, 3.0, 3.4),
        ):
            _run_step(callback, trainer, module, torch.tensor([0]), batch_idx=0)
            self.assertEqual(module.log_dict_calls, [])
            _run_step(callback, trainer, module, torch.tensor([1]), batch_idx=1)

        values, _kwargs = module.log_dict_calls[-1]
        self.assertAlmostEqual(values["module/observation_time"].item(), 0.5)

    def test_disabled_render_does_not_measure_observation_time(self):
        trainer = _trainer(_Writer())
        module = _Module()
        callback = ObservationCallback(every_n_steps=None)
        _start(callback, trainer, module)

        with mock.patch.object(
            observation_callback.time,
            "perf_counter",
            side_effect=AssertionError("disabled observation timing must not run"),
        ):
            _run_step(callback, trainer, module, torch.tensor([0]), batch_idx=0)
        self.assertEqual(module.log_dict_calls, [])

    def test_curve_collectors_only_run_for_rendered_steps(self):
        class Curved(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del inputs, kwargs
            if not isinstance(module, Curved):
                raise TypeError("test observation requires Curved.")
            module.calls += 1
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Curved,
            (observation.ForwardObservation("curve", collect),),
        )
        trainer = _trainer(_Writer())
        module = _Module(Curved())
        callback = ObservationCallback(every_n_steps=2)
        _start(callback, trainer, module)

        _run_step(callback, trainer, module, torch.tensor([1.0]), batch_idx=0)
        self.assertEqual(module.observed.calls, 0)
        _run_step(callback, trainer, module, torch.tensor([2.0]), batch_idx=1)
        self.assertEqual(module.observed.calls, 1)

    def test_gradient_accumulation_finalizes_once_when_global_step_advances(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)

        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0, 0]))
        callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(writer.histograms, [])

        callback.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.tensor([1, 1]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 1)
        current = next(
            kwargs
            for tag, kwargs in writer.histograms
            if tag == "module/observed/stats/codes/book_0/step"
        )
        self.assertEqual(current["bucket_counts"], [2.0, 2.0])

    def test_observation_time_counts_gradient_accumulation_as_one_optimizer_step(self):
        trainer = _trainer(_Writer())
        module = _Module()
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)

        with mock.patch.object(
            observation_callback.time,
            "perf_counter",
            side_effect=(0.0, 0.1, 1.0, 1.2, 2.0, 2.2, 3.0, 3.3),
        ):
            callback.on_train_batch_start(trainer, module, None, 0)
            module.observed(torch.tensor([0]))
            callback.on_train_batch_end(trainer, module, None, None, 0)

            callback.on_train_batch_start(trainer, module, None, 1)
            module.observed(torch.tensor([1]))
            trainer.global_step = 1
            callback.on_train_batch_end(trainer, module, None, None, 1)

        values, _kwargs = module.log_dict_calls[-1]
        self.assertAlmostEqual(values["module/observation_time"].item(), 0.8)

    def test_checkpoint_roundtrip_restores_ema(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, half_life_steps=1)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.tensor([0, 0]), batch_idx=0)
        state = callback.state_dict()
        callback.on_fit_end(trainer, module)

        restored_writer = _Writer()
        restored_trainer = _trainer(restored_writer, step=1)
        restored_module = _Module()
        restored = ObservationCallback(every_n_steps=1, half_life_steps=1)
        restored.load_state_dict(state)
        _start(restored, restored_trainer, restored_module)
        _run_step(
            restored,
            restored_trainer,
            restored_module,
            torch.tensor([1, 1, 1, 1]),
            batch_idx=1,
        )
        ema = next(
            kwargs
            for tag, kwargs in restored_writer.histograms
            if tag == "module/observed/stats/codes/book_0/ema"
        )
        self.assertEqual(ema["bucket_counts"], [1.0, 2.0])

    def test_callback_suppresses_nested_quantizer_observers(self):
        rvq = ResidualVectorQuantizer(
            RVQConfig(
                (
                    VQConfig(input_dim=2, codebook_size=4),
                    VQConfig(input_dim=2, codebook_size=4),
                )
            )
        )
        module = _Module(rvq)
        callback = ObservationCallback()
        trainer = _trainer(_Writer())
        _start(callback, trainer, module)
        self.assertEqual(callback.observed_modules, ("observed",))
        callback.on_fit_end(trainer, module)

    def test_registry_discovery_does_not_call_collectors(self):
        class Provider(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value):
                return value

        def collect(module):
            if not isinstance(module, Provider):
                raise TypeError("test observation requires Provider.")
            module.calls += 1
            return observation.Curve(torch.tensor(1.0))

        observation.registry.register(
            Provider,
            (observation.PollObservation("state", collect),),
        )
        provider = Provider()
        module = _Module(provider)
        callback = ObservationCallback()
        _start(callback, _trainer(_Writer()), module)
        self.assertEqual(provider.calls, 0)

    def test_quantizer_adapter_logs_codes_and_loss_curves(self):
        quantizer = EmbeddingVectorQuantizer(VQConfig(input_dim=2, codebook_size=4))
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(quantizer)
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.randn(3, 2), batch_idx=0)

        values, _ = module.log_dict_calls[-1]
        self.assertIn("module/observed/quantization/loss/commitment", values)
        self.assertIn("module/observed/quantization/loss/codebook", values)
        self.assertTrue(
            any(
                tag == "module/observed/quantization/codes/book_0/ema"
                for tag, _kwargs in writer.histograms
            )
        )

    def test_adt_registry_logs_poll_summaries(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(AdaptiveDirichletTempering(num_experts=3))
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        _run_step(
            callback,
            trainer,
            module,
            torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, -2.0]]),
            batch_idx=0,
        )

        values, _ = module.log_dict_calls[-1]
        self.assertIn("module/observed/adt/variance_mean", values)
        self.assertIn("module/observed/adt/alpha_std", values)
        self.assertIn("module/observed/adt/temperature_max", values)

    def test_moe_registry_logs_forward_routing_summaries(self):
        def scale(input, weight, bias=None):
            del bias
            return input * weight

        moe = MoE(
            num_experts=2,
            router=LinearRouter(2, 2),
            expert=CallableExpert(scale),
            weight=nn.Parameter(torch.tensor([1.0, 2.0])),
            top_k=1,
        )
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(moe)
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        _run_step(
            callback,
            trainer,
            module,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            batch_idx=0,
        )

        values, _ = module.log_dict_calls[-1]
        self.assertIn("module/observed/routing/effective_experts", values)
        self.assertIn("module/observed/routing/usage_mean", values)
        self.assertIn("module/observed/routing/tokens/total", values)
        self.assertIn("module/observed/routing/dropped/ratio", values)

    def test_dense_bin_limit_fails_before_allocation(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(_Observed(sizes=(100,)))
        callback = ObservationCallback(max_dense_bins=10)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "max_dense_bins=10"):
            module.observed(torch.tensor([0]))

    def test_distributed_reduces_counts_before_derived_metrics(self):
        class Strategy:
            def __init__(self):
                self.shapes = []

            def all_gather(self, tensor):
                return torch.stack((tensor, tensor))

            def reduce(self, tensor, *, reduce_op):
                self.shapes.append((tuple(tensor.shape), reduce_op))
                return tensor * 2

        writer = _Writer()
        strategy = Strategy()
        trainer = _trainer(writer, world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.tensor([0, 1]), batch_idx=0)
        values, _ = module.log_dict_calls[-1]
        self.assertEqual(
            values["module/observed/stats/codes/book_0/step/observations"].item(),
            4.0,
        )
        self.assertIn(((1, 2), "sum"), strategy.shapes)

    def test_distributed_schema_mismatch_fails_before_reduce(self):
        class Strategy:
            def __init__(self):
                self.reduce_calls = 0

            def all_gather(self, tensor):
                other = tensor.clone()
                other[0] += 1
                return torch.stack((tensor, other))

            def reduce(self, tensor, *, reduce_op):
                del reduce_op
                self.reduce_calls += 1
                return tensor

        strategy = Strategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0]))
        trainer.global_step = 1
        with self.assertRaisesRegex(RuntimeError, "same observation keys and schemas"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distributed_schema_detects_key_missing_from_current_step(self):
        class Strategy:
            def __init__(self):
                self.missing_current_key = False
                self.reduce_calls = 0

            def all_gather(self, tensor):
                other = tensor.clone()
                if self.missing_current_key:
                    other[1] = 0
                return torch.stack((tensor, other))

            def reduce(self, tensor, *, reduce_op):
                del reduce_op
                self.reduce_calls += 1
                return tensor

        strategy = Strategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(sync_distributed=True)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.tensor([0]), batch_idx=0)

        strategy.reduce_calls = 0
        strategy.missing_current_key = True
        callback.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.tensor([1]))
        trainer.global_step = 2
        with self.assertRaisesRegex(RuntimeError, "same observation keys and schemas"):
            callback.on_train_batch_end(trainer, module, None, None, 1)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distributed_schema_includes_stable_source_identity(self):
        class SourceA(nn.Module):
            def forward(self, value):
                return value

        class SourceB(nn.Module):
            def forward(self, value):
                return value

        def collect_a(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"b/c": observation.Bins(output, sizes=(2,))}

        def collect_b(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"c": observation.Bins(output, sizes=(2,))}

        observation.registry.register(
            SourceA,
            (
                observation.ForwardObservation(
                    "a",
                    collect_a,
                    history=observation.EMAHistory(),
                ),
            ),
        )
        observation.registry.register(
            SourceB,
            (
                observation.ForwardObservation(
                    "a/b",
                    collect_b,
                    history=observation.EMAHistory(),
                ),
            ),
        )

        first_trainer = _trainer(_Writer(), world_size=2)
        first_module = _Module(SourceA())
        first = ObservationCallback(sync_distributed=True)
        _start(first, first_trainer, first_module)
        first.on_train_batch_start(first_trainer, first_module, None, 0)
        first_module.observed(torch.tensor([0]))

        second_trainer = _trainer(_Writer(), world_size=2)
        second_module = _Module(SourceB())
        second = ObservationCallback(sync_distributed=True)
        _start(second, second_trainer, second_module)
        second.on_train_batch_start(second_trainer, second_module, None, 0)
        second_module.observed(torch.tensor([0]))

        class Strategy:
            def __init__(self):
                self.reduce_calls = 0

            def all_gather(self, tensor):
                other = observation_callback._schema_signature(
                    second._states,
                    emitted_keys=second._emitted_keys,
                    device=tensor.device,
                )
                return torch.stack((tensor, other))

            def reduce(self, tensor, *, reduce_op):
                del reduce_op
                self.reduce_calls += 1
                return tensor

        strategy = Strategy()
        first_trainer.strategy = strategy
        first_trainer.global_step = 1
        with self.assertRaisesRegex(RuntimeError, "same observation keys and schemas"):
            first.on_train_batch_end(first_trainer, first_module, None, None, 0)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distribution_writes_step_and_ema_raw_histograms(self):
        class Continuous(nn.Module):
            def forward(self, values):
                return values

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {
                "values": observation.Distribution(
                    output,
                    edges=(0.0, 1.0, 2.0),
                )
            }

        observation.registry.register(
            Continuous,
            (
                observation.ForwardObservation(
                    "stats",
                    collect,
                    history=observation.EMAHistory(half_life_steps=1),
                ),
            ),
        )

        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(Continuous())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        _run_step(
            callback,
            trainer,
            module,
            torch.tensor([0.25, 1.25]),
            batch_idx=0,
        )
        step = next(
            kwargs
            for tag, kwargs in writer.histograms
            if tag == "module/observed/stats/values/step"
        )
        ema = next(
            kwargs
            for tag, kwargs in writer.histograms
            if tag == "module/observed/stats/values/ema"
        )
        self.assertEqual(step["bucket_counts"], [1.0, 1.0])
        self.assertEqual(ema["bucket_counts"], [1.0, 1.0])

    def test_invalid_bins_fail_at_step_boundary_even_without_render(self):
        trainer = _trainer(_Writer())
        module = _Module()
        callback = ObservationCallback(every_n_steps=None)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([-2]))
        trainer.global_step = 1
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(module.log_dict_calls, [])

    def test_invalid_grouped_bins_only_allow_minus_one_as_inactive(self):
        trainer = _trainer(_Writer())
        module = _Module(_Observed(sizes=(2, 2)))
        callback = ObservationCallback(every_n_steps=None)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([[0, -2]]))
        trainer.global_step = 1
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            callback.on_train_batch_end(trainer, module, None, None, 0)

    def test_poll_observation_runs_at_optimizer_step_boundary(self):
        class Polled(nn.Module):
            def __init__(self):
                super().__init__()
                self.current = torch.tensor(0.0)

            def forward(self, value):
                self.current = value.float().mean()
                return value

        observation.registry.register(
            Polled,
            (
                observation.PollObservation(
                    "state",
                    lambda module: observation.Curve(module.current),
                ),
            ),
        )
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(Polled())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.tensor([2, 4]), batch_idx=0)
        values, _ = module.log_dict_calls[-1]
        self.assertEqual(values["module/observed/state/value"].item(), 3.0)

    def test_backward_observation_warns_once_and_is_not_attached(self):
        class BackwardObserved(nn.Module):
            def forward(self, value):
                return value * 2

        def collect(module, grad_outputs):
            del module, grad_outputs
            raise AssertionError("backward observation must not be collected")

        observation.registry.register(
            BackwardObserved,
            (
                observation.BackwardObservation("gradient", collect),
                observation.BackwardObservation("other_gradient", collect),
            ),
        )

        trainer = _trainer(_Writer())
        module = _Module(BackwardObserved())
        callback = ObservationCallback()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _start(callback, trainer, module)
        backward_warnings = [
            item
            for item in caught
            if issubclass(item.category, observation.BackwardObservationWarning)
        ]
        self.assertEqual(len(backward_warnings), 1)
        self.assertIn("GradientProbeLoggerCallback", str(backward_warnings[0].message))
        self.assertEqual(callback.observed_modules, ())
        self.assertEqual(len(module.observed._forward_hooks), 0)
        self.assertEqual(len(module.observed._backward_hooks), 0)
        value = torch.tensor(1.0, requires_grad=True)
        module.observed(value).backward()
        self.assertEqual(value.grad.item(), 2.0)

    def test_default_observation_hooks_only_attach_on_global_zero(self):
        trainer = _trainer(
            _Writer(),
            world_size=2,
            is_global_zero=False,
        )
        module = _Module()
        callback = ObservationCallback()
        _start(callback, trainer, module)
        self.assertEqual(callback.observed_modules, ())
        self.assertEqual(len(module.observed._forward_hooks), 0)

    def test_distributed_observation_attaches_on_nonzero_rank(self):
        trainer = _trainer(
            _Writer(),
            world_size=2,
            is_global_zero=False,
        )
        module = _Module()
        callback = ObservationCallback(sync_distributed=True)
        _start(callback, trainer, module)
        self.assertEqual(callback.observed_modules, ("observed",))
        self.assertEqual(len(module.observed._forward_hooks), 1)
        callback.on_fit_end(trainer, module)

    def test_mapping_value_names_reject_normalized_collisions(self):
        class Colliding(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            value = observation.Curve(output.float().mean())
            return {"value": value, "/value/": value}

        observation.registry.register(
            Colliding,
            (observation.ForwardObservation("stats", collect),),
        )

        trainer = _trainer(_Writer())
        module = _Module(Colliding())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "same tag"):
            module.observed(torch.tensor([1]))

    def test_exception_and_teardown_remove_hooks_idempotently(self):
        trainer = _trainer(_Writer())
        module = _Module()
        callback = ObservationCallback()
        _start(callback, trainer, module)
        self.assertEqual(len(module.observed._forward_hooks), 1)
        callback.on_exception(trainer, module, RuntimeError("boom"))
        self.assertEqual(len(module.observed._forward_hooks), 0)
        callback.teardown(trainer, module, "fit")
        self.assertEqual(len(module.observed._forward_hooks), 0)

    def test_deepcopied_observer_hook_cannot_write_to_original_callback(self):
        trainer = _trainer(_Writer())
        module = _Module()
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        copied = copy.deepcopy(module)
        callback.on_train_batch_start(trainer, module, None, 0)
        copied.observed(torch.tensor([1]))
        self.assertEqual(len(copied.observed._forward_hooks), 0)
        module.observed(torch.tensor([0]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)
        step = next(
            kwargs
            for tag, kwargs in trainer.logger.experiment.histograms
            if tag == "module/observed/stats/codes/book_0/step"
        )
        self.assertEqual(step["bucket_counts"], [1.0, 0.0])

    def test_root_and_module_named_child_have_distinct_tags(self):
        class Root(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.module = _Observed()
                self.log_dict_calls = []

            def forward(self, value):
                self.module(value)
                return value

            def log_dict(self, values, **kwargs):
                self.log_dict_calls.append((values, kwargs))

        def collect_root(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Root,
            (observation.ForwardObservation("root", collect_root),),
        )
        trainer = _trainer(_Writer())
        module = Root()
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module(torch.tensor([0]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)
        values, _ = module.log_dict_calls[-1]
        self.assertIn("module/root/value", values)
        self.assertIn("module/module/stats/mean", values)

    def test_root_and_child_logger_tag_collision_fails_fast(self):
        class Child(nn.Module):
            def forward(self, value):
                return value

        class Root(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.module = Child()

            def forward(self, value):
                return self.module(value)

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Child,
            (observation.ForwardObservation("stats", collect),),
        )
        observation.registry.register(
            Root,
            (observation.ForwardObservation("module/stats", collect),),
        )
        trainer = _trainer(_Writer())
        module = Root()
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "same logger tag"):
            module(torch.tensor([1]))

    def test_histogram_derived_logger_tag_collision_fails_fast(self):
        class Colliding(nn.Module):
            def forward(self, value):
                return value

        def histogram(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"codes": observation.Bins(output, sizes=(2,))}

        def curve(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"entropy": observation.Curve(output.float().mean())}

        observation.registry.register(
            Colliding,
            (
                observation.ForwardObservation(
                    "histogram",
                    histogram,
                    history=observation.EMAHistory(),
                ),
                observation.ForwardObservation(
                    "histogram/codes/step",
                    curve,
                ),
            ),
        )

        trainer = _trainer(_Writer())
        module = _Module(Colliding())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "same logger tag"):
            module.observed(torch.tensor([1]))

    def test_observation_state_key_collision_fails_fast(self):
        class Colliding(nn.Module):
            def forward(self, value):
                return value

        def first(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"b/c": observation.Curve(output.float().mean())}

        def second(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"c": observation.Curve(output.float().mean())}

        observation.registry.register(
            Colliding,
            (
                observation.ForwardObservation("a", first),
                observation.ForwardObservation("a/b", second),
            ),
        )

        trainer = _trainer(_Writer())
        module = _Module(Colliding())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "same state key"):
            module.observed(torch.tensor([1]))

    def test_render_cadence_can_change_when_loading_state_directly(self):
        source = ObservationCallback(every_n_steps=1, tag="observed")
        restored = ObservationCallback(every_n_steps=10, tag="observed")
        self.assertEqual(source.state_key, restored.state_key)
        restored.load_state_dict(source.state_dict())

    def test_checkpoint_restores_hierarchical_spec_source_identity(self):
        class Hierarchical(nn.Module):
            def forward(self, value):
                return value

        def first(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"x": observation.Bins(output, sizes=(2,))}

        def second(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"c": observation.Bins(output, sizes=(2,))}

        observation.registry.register(
            Hierarchical,
            (
                observation.ForwardObservation(
                    "a",
                    first,
                    history=observation.EMAHistory(),
                ),
                observation.ForwardObservation(
                    "a/b",
                    second,
                    history=observation.EMAHistory(),
                ),
            ),
        )

        trainer = _trainer(_Writer())
        module = _Module(Hierarchical())
        callback = ObservationCallback(every_n_steps=None)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.tensor([0]), batch_idx=0)
        state = callback.state_dict()
        callback.on_fit_end(trainer, module)

        restored_trainer = _trainer(_Writer(), step=1)
        restored_module = _Module(Hierarchical())
        restored = ObservationCallback(every_n_steps=None)
        restored.load_state_dict(state)
        _start(restored, restored_trainer, restored_module)
        self.assertEqual(
            set(restored.state_dict()["owners"]),
            {"observed/a/x", "observed/a/b/c"},
        )

    def test_real_trainer_checkpoint_resume_restores_observation_ema(self):
        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.codes = _Observed()

            def training_step(self, batch: Any, batch_idx: int):
                del batch, batch_idx
                value = min(int(self.global_step), 1)
                self.codes(torch.full((2,), value, dtype=torch.long, device=self.device))
                return self.weight.square()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "observation.ckpt"
            source = ObservationCallback(
                every_n_steps=None,
                half_life_steps=1,
            )
            trainer = pl.Trainer(
                accelerator="cpu",
                max_epochs=1,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                logger=False,
                callbacks=[source],
            )
            trainer.fit(Tiny(), loader)
            trainer.save_checkpoint(checkpoint)

            restored = ObservationCallback(
                every_n_steps=None,
                half_life_steps=1,
            )
            resumed = pl.Trainer(
                accelerator="cpu",
                max_epochs=2,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                logger=False,
                callbacks=[restored],
            )
            resumed.fit(Tiny(), loader, ckpt_path=str(checkpoint))
            histogram = restored.state_dict()["histograms"]["codes/stats/codes"]
            self.assertTrue(
                torch.equal(histogram["ema_mass"], torch.tensor([[1.0, 1.0]]))
            )

    def test_only_general_callback_is_exported_from_lightning_root(self):
        from anytrain import lightning
        from anytrain.lightning import callback

        self.assertIs(lightning.ObservationCallback, ObservationCallback)
        self.assertFalse(hasattr(lightning, "ModuleObservationCallback"))
        self.assertFalse(hasattr(lightning, "ADTStatsLoggerCallback"))
        self.assertFalse(hasattr(lightning, "CodebookUsageLoggerCallback"))
        self.assertFalse(hasattr(lightning, "QuantizationLoggerCallback"))
        self.assertFalse(hasattr(lightning, "MoELoggerCallback"))
        self.assertFalse(hasattr(callback, "ObservationCallback"))
        self.assertFalse(hasattr(callback, "ModuleObservationCallback"))


if __name__ == "__main__":
    unittest.main()
