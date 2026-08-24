import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast
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


def SimpleNamespace(**kwargs: object) -> Any:
    return _SimpleNamespace(**kwargs)


class _Writer:
    def __init__(self):
        self.histograms = []

    def add_histogram_raw(self, tag, **kwargs):
        self.histograms.append((tag, kwargs))


class _ErrorStrategy:
    def __init__(self, *, remote_error=False):
        self.remote_error = remote_error
        self.all_gather_calls = 0
        self.reduce_calls = 0

    def all_gather(self, value):
        self.all_gather_calls += 1
        other = value.clone()
        if self.remote_error:
            other[observation_callback._SCHEMA_ERROR_INDEX] = 1
        return torch.stack((value, other))

    def reduce(self, tensor, *, reduce_op):
        del reduce_op
        self.reduce_calls += 1
        return tensor


class _Observed(nn.Module):
    def __init__(self, *, sizes=(2,)):
        super().__init__()
        self.sizes = sizes
        observation.registry.recommend(self)

    def forward(self, values):
        return values

    @staticmethod
    def _collect_codes(module, inputs, kwargs, output):
        del inputs, kwargs
        if not isinstance(module, _Observed):
            raise TypeError("test observation requires _Observed.")
        return {
            "codes": observation.Bins(
                output,
                sizes=module.sizes,
                group_dim=None if len(module.sizes) == 1 else -1,
                group_names=tuple(f"book_{index}" for index in range(len(module.sizes))),
            ),
        }


observation.registry.register(
    _Observed,
    (
        observation.ForwardObservation(
            "stats",
            _Observed._collect_codes,
            reduction=observation.Reduction.Merge,
            history=observation.EMAHistory(half_life_steps=1),
            recommended=True,
            covers_descendants=True,
        ),
    ),
)


class _ProfileObserved(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, values):
        return values

    @staticmethod
    def _collect(module, inputs, kwargs, output):
        del module, inputs, kwargs
        if not isinstance(output, torch.Tensor):
            raise TypeError("test profile observation requires a tensor.")
        coordinates = torch.tensor(
            [0.25, 1.25],
            dtype=torch.float32,
            device=output.device,
        )
        return {
            "value": observation.Profile(
                output.float(),
                coordinates,
                edges=(0.0, 1.0, 2.0),
            )
        }


observation.registry.register(
    _ProfileObserved,
    (
        observation.ForwardObservation(
            "profile",
            _ProfileObserved._collect,
            reduction=observation.Reduction.Merge,
            history=observation.EMAHistory(half_life_steps=1),
        ),
    ),
)


class _RankFailObserved(nn.Module):
    def __init__(self, *, fail: bool):
        super().__init__()
        self.fail = fail

    def forward(self, values):
        return values

    @staticmethod
    def _collect(module, inputs, kwargs, output):
        del inputs, kwargs
        if not isinstance(module, _RankFailObserved):
            raise TypeError("test observation requires _RankFailObserved.")
        if module.fail:
            raise ValueError("rank-local collector exploded")
        return observation.Curve(output.float().mean())


observation.registry.register(
    _RankFailObserved,
    (
        observation.OutputObservation(
            "stats",
            _RankFailObserved._collect,
            reduction=observation.Reduction.Mean,
        ),
    ),
)


class _Module(pl.LightningModule):
    def __init__(self, observed=None):
        super().__init__()
        self.observed = _Observed() if observed is None else observed
        self.log_dict_calls = []

    def log_dict(self, dictionary, *args, **kwargs):
        del args
        self.log_dict_calls.append((dictionary, kwargs))


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


def _distributed_error_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    import json
    from datetime import timedelta

    from lightning.pytorch.strategies import DDPStrategy

    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    payload: dict[str, str] = {}
    try:
        collector_trainer = _trainer(
            _Writer(),
            world_size=world_size,
            strategy=DDPStrategy(),
            is_global_zero=rank == 0,
        )
        collector_module = _Module(_RankFailObserved(fail=rank == 1))
        collector_callback = ObservationCallback(
            every_n_steps=1,
            sync_distributed=True,
        )
        _start(collector_callback, collector_trainer, collector_module)
        collector_callback.on_train_batch_start(
            collector_trainer,
            collector_module,
            None,
            0,
        )
        collector_module.observed(torch.tensor([1.0]))
        collector_trainer.global_step = 1
        try:
            collector_callback.on_train_batch_end(
                collector_trainer,
                collector_module,
                None,
                None,
                0,
            )
        except RuntimeError as error:
            payload["collector"] = str(error)
        finally:
            collector_callback.on_exception(
                collector_trainer,
                collector_module,
                RuntimeError("test cleanup"),
            )

        schema_trainer = _trainer(
            _Writer(),
            world_size=world_size,
            strategy=DDPStrategy(),
            is_global_zero=rank == 0,
        )
        schema_module = _Module()
        schema_callback = ObservationCallback(
            every_n_steps=1,
            sync_distributed=True,
        )
        _start(schema_callback, schema_trainer, schema_module)
        schema_callback.on_train_batch_start(
            schema_trainer,
            schema_module,
            None,
            0,
        )
        schema_module.observed(torch.tensor([0, 1]))
        if rank == 1:
            schema_callback._emitted_keys.add("rank-local-missing-state")
        schema_trainer.global_step = 1
        try:
            schema_callback.on_train_batch_end(
                schema_trainer,
                schema_module,
                None,
                None,
                0,
            )
        except RuntimeError as error:
            payload["schema"] = str(error)
        finally:
            schema_callback.on_exception(
                schema_trainer,
                schema_module,
                RuntimeError("test cleanup"),
            )

        class RankWriter(_Writer):
            def add_histogram_raw(self, tag, **kwargs):
                if rank == 0:
                    raise RuntimeError("rank-local histogram writer exploded")
                super().add_histogram_raw(tag, **kwargs)

        histogram_trainer = _trainer(
            RankWriter(),
            world_size=world_size,
            strategy=DDPStrategy(),
            is_global_zero=rank == 0,
        )
        histogram_module = _Module()
        histogram_callback = ObservationCallback(
            every_n_steps=1,
            sync_distributed=True,
        )
        _start(histogram_callback, histogram_trainer, histogram_module)
        histogram_callback.on_train_batch_start(
            histogram_trainer,
            histogram_module,
            None,
            0,
        )
        histogram_module.observed(torch.tensor([0, 1]))
        histogram_trainer.global_step = 1
        try:
            histogram_callback.on_train_batch_end(
                histogram_trainer,
                histogram_module,
                None,
                None,
                0,
            )
        except RuntimeError as error:
            payload["histogram"] = str(error)
        finally:
            histogram_callback.on_exception(
                histogram_trainer,
                histogram_module,
                RuntimeError("test cleanup"),
            )
        (Path(output_dir) / f"rank-{rank}.json").write_text(json.dumps(payload))
    finally:
        torch.distributed.destroy_process_group()


class ObservationCallbackTest(unittest.TestCase):
    def test_curve_reduction_folds_one_train_commit_window(self):
        class Curves(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Curves,
            (
                observation.OutputObservation(
                    "last",
                    collect,
                    reduction=observation.Reduction.Last,
                ),
                observation.OutputObservation(
                    "mean",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(Curves())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)

        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.tensor([3.0]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 1)

        values, _kwargs = module.log_dict_calls[-1]
        self.assertEqual(values["observation/module/observed/last/value"].item(), 3.0)
        self.assertEqual(values["observation/module/observed/mean/value"].item(), 2.0)

    def test_curve_reduction_folds_one_validation_epoch(self):
        class Curves(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Curves,
            (
                observation.OutputObservation(
                    "last",
                    collect,
                    reduction=observation.Reduction.Last,
                ),
                observation.OutputObservation(
                    "mean",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(Curves())
        callback = ObservationCallback(every_n_steps=None)
        _start(callback, trainer, module)

        callback.on_validation_epoch_start(trainer, module)
        module.observed(torch.tensor([1.0]))
        module.observed(torch.tensor([3.0]))
        callback.on_validation_epoch_end(trainer, module)

        values, _kwargs = module.log_dict_calls[-1]
        self.assertEqual(values["observation/val/module/observed/last/value"].item(), 3.0)
        self.assertEqual(values["observation/val/module/observed/mean/value"].item(), 2.0)

    def test_context_prefix_is_shared_by_parent_and_child_observations(self):
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = _Observed()

            def forward(self, values):
                return self.child(values)

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Parent,
            (
                observation.OutputObservation(
                    "total",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(Parent())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with observation.context("train/reconstruction"):
            module.observed(torch.tensor([0, 1]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)

        values, _kwargs = module.log_dict_calls[-1]
        prefix = "observation/train/reconstruction/module/observed"
        self.assertEqual(values[f"{prefix}/total/value"].item(), 0.5)
        self.assertEqual(
            values[
                "observation/train/reconstruction/module/observed.child/stats/codes/book_0/step/observations"
            ].item(),
            2.0,
        )

    def test_validation_observations_use_an_independent_context(self):
        trainer = _trainer(_Writer())
        module = _Module(_Observed(sizes=(4,)))
        callback = ObservationCallback(every_n_steps=10)
        _start(callback, trainer, module)

        callback.on_validation_epoch_start(trainer, module)
        module.observed(torch.tensor([1]))
        module.observed(torch.tensor([3]))
        callback.on_validation_epoch_end(trainer, module)

        values, kwargs = module.log_dict_calls[-1]
        self.assertEqual(
            values["observation/val/module/observed/stats/codes/book_0/step/observations"].item(),
            2.0,
        )
        self.assertFalse(kwargs["on_step"])
        self.assertTrue(kwargs["on_epoch"])

        state = callback.state_dict()
        self.assertEqual(state["validation_step"], 1)
        self.assertIn("val/observed/stats/codes", state["histograms"])

        restored = ObservationCallback(every_n_steps=10)
        restored.load_state_dict(state)
        restored_module = _Module(_Observed(sizes=(4,)))
        _start(restored, trainer, restored_module)
        restored.on_validation_epoch_start(trainer, restored_module)
        restored_module.observed(torch.tensor([2]))
        restored.on_validation_epoch_end(trainer, restored_module)
        self.assertEqual(restored.state_dict()["validation_step"], 2)

    def test_real_lightning_validation_logs_contextual_observations(self):
        class CurveObserved(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            CurveObserved,
            (
                observation.OutputObservation(
                    "value",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )

        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.observed = CurveObserved()

            def training_step(self, batch, batch_idx):
                del batch, batch_idx
                return self.weight.square()

            def validation_step(self, batch, batch_idx):
                del batch_idx
                with observation.context("reconstruction"):
                    self.observed(batch[0])

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        loader = DataLoader(TensorDataset(torch.tensor([[1.0], [3.0]])), batch_size=2)
        callback = ObservationCallback(every_n_steps=None)
        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[callback],
            num_sanity_val_steps=0,
        )
        trainer.fit(Tiny(), loader, loader)

        key = "observation/val/reconstruction/module/observed/value/value"
        self.assertEqual(trainer.callback_metrics[key].item(), 2.0)

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
            if tag == "observation/module/observed/stats/codes/book_0/ema"
        )
        self.assertEqual(ema["bucket_counts"], [1.0, 2.0])
        values, kwargs = module.log_dict_calls[-1]
        self.assertEqual(kwargs["sync_dist"], False)
        self.assertEqual(
            values["observation/module/observed/stats/codes/book_0/step/observations"].item(),
            4.0,
        )
        self.assertEqual(
            values["observation/module/observed/stats/codes/book_0/step/occupied_count"].item(),
            1.0,
        )
        self.assertEqual(
            values["observation/module/observed/stats/codes/book_0/ema/occupied_count"].item(),
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
        self.assertAlmostEqual(values["observation/time"].item(), 0.5)

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
            (
                observation.ForwardObservation(
                    "curve",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
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
            if tag == "observation/module/observed/stats/codes/book_0/step"
        )
        self.assertEqual(current["bucket_counts"], [2.0, 2.0])

    def test_manual_optimization_allows_multiple_optimizer_steps_per_batch(self):
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module()
        module.automatic_optimization = False
        callback = ObservationCallback(every_n_steps=2)
        _start(callback, trainer, module)

        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0]))
        module.observed(torch.tensor([1]))
        trainer.global_step = 2
        callback.on_train_batch_end(trainer, module, None, None, 0)

        current = next(
            kwargs
            for tag, kwargs in writer.histograms
            if tag == "observation/module/observed/stats/codes/book_0/step"
        )
        self.assertEqual(current["bucket_counts"], [1.0, 1.0])

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
        self.assertAlmostEqual(values["observation/time"].item(), 0.8)

    def test_backward_window_ignores_checkpoint_replay_then_resumes_forward(self):
        class ReplayObserved(nn.Module):
            def forward(self, value):
                return value

        def collect_histograms(module, inputs, kwargs, output):
            del module, inputs, kwargs
            values = output.float()
            return {
                "distribution": observation.Distribution(
                    values,
                    edges=(-0.5, 0.5, 1.5),
                ),
                "bins": observation.Bins(output, sizes=(2,)),
                "profile": observation.Profile(
                    values,
                    torch.tensor([0.25, 1.25]),
                    edges=(0.0, 1.0, 2.0),
                ),
            }

        def collect_scalars(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {"curve": observation.Curve(output.float().mean())}

        observation.registry.register(
            ReplayObserved,
            (
                observation.OutputObservation(
                    "replay",
                    collect_histograms,
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(),
                ),
                observation.OutputObservation(
                    "replay/scalars",
                    collect_scalars,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        writer = _Writer()
        trainer = _trainer(writer)
        module = _Module(ReplayObserved())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0, 1]))
        callback.on_before_backward(trainer, module, torch.tensor(0.0))
        module.observed(torch.tensor([1, 1]))
        callback.on_after_backward(trainer, module)
        module.observed(torch.tensor([0, 0]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)

        metrics, _ = module.log_dict_calls[-1]
        prefix = "observation/module/observed/replay"
        self.assertEqual(metrics[f"{prefix}/scalars/curve"].item(), 0.25)
        self.assertEqual(
            metrics[f"{prefix}/distribution/step/observations"].item(),
            4.0,
        )
        self.assertEqual(metrics[f"{prefix}/bins/step/observations"].item(), 4.0)
        self.assertEqual(metrics[f"{prefix}/profile/step/bin_0/count"].item(), 2.0)
        self.assertEqual(metrics[f"{prefix}/profile/step/bin_1/count"].item(), 2.0)
        distribution = next(
            kwargs for tag, kwargs in writer.histograms if tag == f"{prefix}/distribution/step"
        )
        bins = next(kwargs for tag, kwargs in writer.histograms if tag == f"{prefix}/bins/step")
        self.assertEqual(distribution["bucket_counts"], [3.0, 1.0])
        self.assertEqual(bins["bucket_counts"], [3.0, 1.0])

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
            if tag == "observation/module/observed/stats/codes/book_0/ema"
        )
        self.assertEqual(ema["bucket_counts"], [1.0, 2.0])

    def test_profile_checkpoint_roundtrip_uses_v4_state(self):
        trainer = _trainer(_Writer())
        module = _Module(_ProfileObserved())
        source = ObservationCallback(every_n_steps=1, half_life_steps=1)
        _start(source, trainer, module)
        _run_step(source, trainer, module, torch.tensor([1.0, 3.0]), batch_idx=0)
        state = source.state_dict()
        self.assertEqual(state["version"], 4)
        self.assertIn("profiles", state)
        source.on_fit_end(trainer, module)

        restored_trainer = _trainer(_Writer(), step=1)
        restored_module = _Module(_ProfileObserved())
        restored = ObservationCallback(every_n_steps=1, half_life_steps=1)
        restored.load_state_dict(state)
        _start(restored, restored_trainer, restored_module)
        _run_step(
            restored,
            restored_trainer,
            restored_module,
            torch.tensor([3.0, 5.0]),
            batch_idx=1,
        )
        values, _ = restored_module.log_dict_calls[-1]
        prefix = "observation/module/observed/profile/value/ema"
        self.assertAlmostEqual(values[f"{prefix}/bin_0/mean"].item(), 2.0)
        self.assertAlmostEqual(values[f"{prefix}/bin_1/mean"].item(), 4.0)

    def test_v2_checkpoint_without_profiles_remains_compatible(self):
        trainer = _trainer(_Writer())
        module = _Module()
        source = ObservationCallback(every_n_steps=None, half_life_steps=1)
        _start(source, trainer, module)
        _run_step(source, trainer, module, torch.tensor([0, 1]), batch_idx=0)
        state = copy.deepcopy(source.state_dict())
        state["version"] = 2
        state.pop("profiles")
        source.on_fit_end(trainer, module)

        restored = ObservationCallback(every_n_steps=None, half_life_steps=1)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict()["version"], 4)
        self.assertIn("histograms", restored.state_dict())

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
        self.assertIn("observation/module/observed/quantization/loss/commitment", values)
        self.assertIn("observation/module/observed/quantization/loss/codebook", values)
        self.assertTrue(
            any(
                tag == "observation/module/observed/quantization/codes/book_0/ema"
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
        self.assertIn("observation/module/observed/adt/variance_mean", values)
        self.assertIn("observation/module/observed/adt/alpha_std", values)
        self.assertIn("observation/module/observed/adt/temperature_max", values)

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
        self.assertIn("observation/module/observed/routing/effective_experts", values)
        self.assertIn("observation/module/observed/routing/usage_mean", values)
        self.assertIn("observation/module/observed/routing/tokens/total", values)
        self.assertIn("observation/module/observed/routing/dropped/ratio", values)

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
            values["observation/module/observed/stats/codes/book_0/step/observations"].item(),
            4.0,
        )
        self.assertIn(((1, 2), "sum"), strategy.shapes)

    def test_distributed_profile_reduces_counts_and_moments(self):
        class Strategy:
            def all_gather(self, tensor):
                return torch.stack((tensor, tensor))

            def reduce(self, tensor, *, reduce_op):
                self.last_reduce_op = reduce_op
                return tensor * 2

        strategy = Strategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module(_ProfileObserved())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.tensor([1.0, 3.0]), batch_idx=0)

        values, _ = module.log_dict_calls[-1]
        prefix = "observation/module/observed/profile/value/step"
        self.assertEqual(values[f"{prefix}/observations"].item(), 4.0)
        self.assertEqual(values[f"{prefix}/bin_0/count"].item(), 2.0)
        self.assertEqual(values[f"{prefix}/bin_1/count"].item(), 2.0)
        self.assertEqual(values[f"{prefix}/bin_0/mean"].item(), 1.0)
        self.assertEqual(values[f"{prefix}/bin_1/mean"].item(), 3.0)
        self.assertEqual(values[f"{prefix}/bin_0/std"].item(), 0.0)

    def test_distributed_profile_schema_mismatch_fails_before_reduce(self):
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
        module = _Module(_ProfileObserved())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0, 3.0]))
        trainer.global_step = 1
        with self.assertRaisesRegex(RuntimeError, "same observation keys and schemas"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(strategy.reduce_calls, 0)

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

    def test_distributed_schema_invariant_error_is_synchronized_before_reduce(self):
        strategy = _ErrorStrategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0]))
        callback._emitted_keys.add("missing/runtime/state")
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed module observation") as error:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("schema materialization failed", str(error.exception))
        self.assertIn("missing runtime state", str(error.exception))
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_non_distributed_collector_error_still_raises_in_forward(self):
        class BrokenOutput(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs, output
            raise ValueError("output collector exploded")

        observation.registry.register(
            BrokenOutput,
            (
                observation.OutputObservation(
                    "stats",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(BrokenOutput())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "output collector exploded"):
            module.observed(torch.tensor([1.0]))

    def test_distributed_output_collector_error_is_synchronized_before_reduce(self):
        class BrokenOutput(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs, output
            raise ValueError("output collector exploded")

        observation.registry.register(
            BrokenOutput,
            (
                observation.OutputObservation(
                    "stats",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        strategy = _ErrorStrategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module(BrokenOutput())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed module observation") as error:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("output collector exploded", str(error.exception))
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distributed_forward_event_record_error_is_synchronized(self):
        class BrokenEvent(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "distribution",
                    observation.Distribution(value, edges=(0.0, 1.0)),
                )
                return value

        observation.registry.register(
            BrokenEvent,
            (
                observation.ForwardEvent(
                    "distribution",
                    reduction=observation.Reduction.Last,
                ),
            ),
        )
        strategy = _ErrorStrategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module(BrokenEvent())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0.5]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed module observation") as error:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("does not support Distribution", str(error.exception))
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distributed_state_collector_error_is_synchronized(self):
        class BrokenState(nn.Module):
            def forward(self, value):
                return value

        def collect(module):
            del module
            raise RuntimeError("state collector exploded")

        observation.registry.register(
            BrokenState,
            (observation.StateObservation("state", collect),),
        )
        strategy = _ErrorStrategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module(BrokenState())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed module observation") as error:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("state collector exploded", str(error.exception))
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distributed_observation_error_on_other_rank_is_explicit(self):
        strategy = _ErrorStrategy(remote_error=True)
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "failed on another rank"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(strategy.reduce_calls, 0)

    def test_distributed_last_averages_each_rank_local_last_value(self):
        class LastCurve(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            LastCurve,
            (
                observation.OutputObservation(
                    "curve",
                    collect,
                    reduction=observation.Reduction.Last,
                ),
            ),
        )

        class Strategy:
            def __init__(self):
                self.reduce_calls = 0

            def all_gather(self, tensor):
                return torch.stack((tensor, tensor))

            def reduce(self, tensor, *, reduce_op):
                del reduce_op
                self.reduce_calls += 1
                remote = 5.0 if self.reduce_calls == 1 else 1.0
                return tensor + tensor.new_tensor(remote)

        strategy = Strategy()
        trainer = _trainer(_Writer(), world_size=2, strategy=strategy)
        module = _Module(LastCurve())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.tensor([3.0]))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 1)

        values, _kwargs = module.log_dict_calls[-1]
        self.assertEqual(values["observation/module/observed/curve/value"].item(), 4.0)
        self.assertEqual(strategy.reduce_calls, 2)

    def test_distributed_schema_includes_curve_reduction(self):
        class LastCurve(nn.Module):
            def forward(self, value):
                return value

        class MeanCurve(nn.Module):
            def forward(self, value):
                return value

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            LastCurve,
            (
                observation.OutputObservation(
                    "curve",
                    collect,
                    reduction=observation.Reduction.Last,
                ),
            ),
        )
        observation.registry.register(
            MeanCurve,
            (
                observation.OutputObservation(
                    "curve",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        first_trainer = _trainer(_Writer(), world_size=2)
        first_module = _Module(LastCurve())
        first = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(first, first_trainer, first_module)
        first.on_train_batch_start(first_trainer, first_module, None, 0)
        first_module.observed(torch.tensor([1.0]))

        second_trainer = _trainer(_Writer(), world_size=2)
        second_module = _Module(MeanCurve())
        second = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(second, second_trainer, second_module)
        second.on_train_batch_start(second_trainer, second_module, None, 0)
        second_module.observed(torch.tensor([1.0]))

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

    def test_real_gloo_synchronizes_observation_and_writer_errors(self):
        import json

        if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
            self.skipTest("Gloo distributed backend is unavailable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            init_method = (Path(tmp_dir) / "init").as_uri()
            multiprocessing: Any = torch.multiprocessing
            multiprocessing.spawn(
                _distributed_error_worker,
                args=(2, init_method, tmp_dir),
                nprocs=2,
                join=True,
            )
            payloads = [
                json.loads((Path(tmp_dir) / f"rank-{rank}.json").read_text()) for rank in range(2)
            ]

        for payload in payloads:
            self.assertIn("collector", payload)
            self.assertIn("schema", payload)
            self.assertIn("histogram", payload)
        self.assertIn("another rank", payloads[0]["collector"])
        self.assertIn("rank-local collector exploded", payloads[1]["collector"])
        self.assertIn("another rank", payloads[0]["schema"])
        self.assertIn("rank-local-missing-state", payloads[1]["schema"])
        self.assertIn("rank-local histogram writer exploded", payloads[0]["histogram"])
        self.assertIn("another rank", payloads[1]["histogram"])

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
                    reduction=observation.Reduction.Merge,
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
                    reduction=observation.Reduction.Merge,
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
                    reduction=observation.Reduction.Merge,
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
            if tag == "observation/module/observed/stats/values/step"
        )
        ema = next(
            kwargs for tag, kwargs in writer.histograms if tag == "observation/module/observed/stats/values/ema"
        )
        self.assertEqual(step["bucket_counts"], [1.0, 1.0])
        self.assertEqual(ema["bucket_counts"], [1.0, 1.0])

    def test_distributed_raw_histogram_discovery_error_is_synchronized(self):
        class DiscoveryWriter:
            @property
            def add_histogram_raw(self):
                raise RuntimeError("histogram discovery exploded")

        strategy = _ErrorStrategy()
        trainer = _trainer(DiscoveryWriter(), world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0, 1]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed raw histogram") as error:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("histogram discovery exploded", str(error.exception))
        self.assertEqual(strategy.all_gather_calls, 2)

    def test_distributed_raw_histogram_write_error_is_synchronized(self):
        class FailingWriter(_Writer):
            def __init__(self):
                super().__init__()
                self.write_calls = 0

            def add_histogram_raw(self, tag, **kwargs):
                del tag, kwargs
                self.write_calls += 1
                raise RuntimeError("histogram writer exploded")

        writer = FailingWriter()
        strategy = _ErrorStrategy()
        trainer = _trainer(writer, world_size=2, strategy=strategy)
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0, 1]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed raw histogram") as error:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("histogram writer exploded", str(error.exception))
        self.assertEqual(writer.write_calls, 1)
        self.assertEqual(strategy.all_gather_calls, 2)

    def test_distributed_raw_histogram_error_on_other_rank_is_explicit(self):
        class Strategy:
            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, value):
                self.all_gather_calls += 1
                other = value.clone()
                if self.all_gather_calls == 2:
                    other.fill_(1)
                return torch.stack((value, other))

            def reduce(self, tensor, *, reduce_op):
                del reduce_op
                return tensor

        strategy = Strategy()
        trainer = _trainer(
            _Writer(),
            world_size=2,
            strategy=strategy,
            is_global_zero=False,
        )
        module = _Module()
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([0, 1]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "failed on another rank"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(strategy.all_gather_calls, 2)

    def test_profile_logs_step_and_ema_count_mean_std(self):
        trainer = _trainer(_Writer())
        module = _Module(_ProfileObserved())
        callback = ObservationCallback(every_n_steps=1, half_life_steps=1)
        _start(callback, trainer, module)

        _run_step(callback, trainer, module, torch.tensor([1.0, 3.0]), batch_idx=0)
        _run_step(callback, trainer, module, torch.tensor([3.0, 5.0]), batch_idx=1)

        values, _ = module.log_dict_calls[-1]
        prefix = "observation/module/observed/profile/value"
        self.assertEqual(values[f"{prefix}/step/observations"].item(), 2.0)
        self.assertEqual(values[f"{prefix}/step/active_ratio"].item(), 1.0)
        self.assertEqual(values[f"{prefix}/step/bin_0/count"].item(), 1.0)
        self.assertEqual(values[f"{prefix}/step/bin_1/count"].item(), 1.0)
        self.assertEqual(values[f"{prefix}/step/bin_0/mean"].item(), 3.0)
        self.assertEqual(values[f"{prefix}/step/bin_1/mean"].item(), 5.0)
        self.assertEqual(values[f"{prefix}/step/bin_0/std"].item(), 0.0)
        self.assertEqual(values[f"{prefix}/step/bin_1/std"].item(), 0.0)
        self.assertEqual(values[f"{prefix}/ema/observations"].item(), 2.0)
        self.assertEqual(values[f"{prefix}/ema/bin_0/count"].item(), 1.0)
        self.assertEqual(values[f"{prefix}/ema/bin_1/count"].item(), 1.0)
        self.assertAlmostEqual(values[f"{prefix}/ema/bin_0/mean"].item(), 2.0)
        self.assertAlmostEqual(values[f"{prefix}/ema/bin_1/mean"].item(), 4.0)
        self.assertAlmostEqual(values[f"{prefix}/ema/bin_0/std"].item(), 1.0)
        self.assertAlmostEqual(values[f"{prefix}/ema/bin_1/std"].item(), 1.0)

    def test_invalid_profile_pairs_fail_at_step_boundary_without_render(self):
        class InvalidProfile(nn.Module):
            def forward(self, values):
                return values

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Profile(
                output,
                torch.tensor([0.5, float("nan")]),
                edges=(0.0, 1.0, 2.0),
            )

        observation.registry.register(
            InvalidProfile,
            (
                observation.ForwardObservation(
                    "profile",
                    collect,
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(),
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(InvalidProfile())
        callback = ObservationCallback(every_n_steps=None)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0, 2.0]))
        trainer.global_step = 1
        with self.assertRaisesRegex(ValueError, "out-of-range profile pairs"):
            callback.on_train_batch_end(trainer, module, None, None, 0)

    def test_output_reduction_must_support_every_mapping_value(self):
        class Mixed(nn.Module):
            def forward(self, values):
                return values

        def collect(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return {
                "curve": observation.Curve(output.float().mean()),
                "codes": observation.Bins(output, sizes=(2,)),
            }

        observation.registry.register(
            Mixed,
            (
                observation.OutputObservation(
                    "mixed",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(Mixed())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)

        with self.assertRaisesRegex(ValueError, "does not support Bins"):
            module.observed(torch.tensor([0, 1]))

    def test_state_observation_validates_implicit_reduction_against_history(self):
        class State(nn.Module):
            def forward(self, value):
                return value

        observation.registry.register(
            State,
            (
                observation.StateObservation(
                    "curve",
                    lambda module: observation.Curve(torch.tensor(1.0)),
                    history=observation.EMAHistory(),
                ),
                observation.StateObservation(
                    "distribution",
                    lambda module: observation.Distribution(
                        torch.tensor([0.5]),
                        edges=(0.0, 1.0),
                    ),
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(State())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        trainer.global_step = 1

        with self.assertRaisesRegex(ValueError, "only Merge"):
            callback.on_train_batch_end(trainer, module, None, None, 0)

        class HistogramState(nn.Module):
            def forward(self, value):
                return value

        observation.registry.register(
            HistogramState,
            (
                observation.StateObservation(
                    "distribution",
                    lambda module: observation.Distribution(
                        torch.tensor([0.5]),
                        edges=(0.0, 1.0),
                    ),
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _Module(HistogramState())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        trainer.global_step = 1

        with self.assertRaisesRegex(ValueError, "requires EMAHistory"):
            callback.on_train_batch_end(trainer, module, None, None, 0)

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
                    lambda module: observation.Curve(cast(Any, module).current),
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
        self.assertEqual(values["observation/module/observed/state/value"].item(), 3.0)

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
        assert value.grad is not None
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
            (
                observation.ForwardObservation(
                    "stats",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
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
            if tag == "observation/module/observed/stats/codes/book_0/step"
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

            def log_dict(self, dictionary, *args, **kwargs):
                del args
                self.log_dict_calls.append((dictionary, kwargs))

        def collect_root(module, inputs, kwargs, output):
            del module, inputs, kwargs
            return observation.Curve(output.float().mean())

        observation.registry.register(
            Root,
            (
                observation.ForwardObservation(
                    "root",
                    collect_root,
                    reduction=observation.Reduction.Mean,
                ),
            ),
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
        self.assertIn("observation/module/root/value", values)
        self.assertIn(
            "observation/module/module/stats/codes/book_0/step/observations",
            values,
        )

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
            (
                observation.ForwardObservation(
                    "stats",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        observation.registry.register(
            Root,
            (
                observation.ForwardObservation(
                    "module/stats",
                    collect,
                    reduction=observation.Reduction.Mean,
                ),
            ),
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
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(),
                ),
                observation.ForwardObservation(
                    "histogram/codes/step",
                    curve,
                    reduction=observation.Reduction.Mean,
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
                observation.ForwardObservation(
                    "a",
                    first,
                    reduction=observation.Reduction.Mean,
                ),
                observation.ForwardObservation(
                    "a/b",
                    second,
                    reduction=observation.Reduction.Mean,
                ),
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
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(),
                ),
                observation.ForwardObservation(
                    "a/b",
                    second,
                    reduction=observation.Reduction.Merge,
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
            self.assertTrue(torch.equal(histogram["ema_mass"], torch.tensor([[1.0, 1.0]])))

    def test_only_general_callback_is_exported_from_lightning_root(self):
        from anytrain import lightning
        from anytrain.lightning import callback

        self.assertIs(lightning.ObservationCallback, ObservationCallback)
        self.assertFalse(hasattr(lightning, "ModuleObservationCallback"))
        self.assertFalse(hasattr(lightning, "ADTStatsLoggerCallback"))
        self.assertFalse(hasattr(lightning, "CodebookUsageLoggerCallback"))
        self.assertFalse(hasattr(lightning, "QuantizationLoggerCallback"))
        self.assertFalse(hasattr(lightning, "MoELoggerCallback"))
        self.assertIs(callback.ObservationCallback, ObservationCallback)
        self.assertIn("ObservationCallback", callback.__all__)
        self.assertFalse(hasattr(callback, "ModuleObservationCallback"))


if __name__ == "__main__":
    unittest.main()
