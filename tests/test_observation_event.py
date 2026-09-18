import gc
import unittest
import weakref
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast
from unittest import mock

import torch
from lightning import pytorch as pl
from torch import nn

import anytrain.lightning.callback._observation_state as observation_state
import anytrain.lightning.callback.observation as observation_callback
from anytrain import observation
from anytrain.lightning import ObservationCallback
from anytrain.observation import _binding as observation_binding


def SimpleNamespace(**kwargs: object) -> Any:
    return _SimpleNamespace(**kwargs)


class _Writer:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram_raw(self, tag, **kwargs):
        self.histograms.append((tag, kwargs))

    def add_image(self, tag, image, global_step, *, dataformats):
        self.images.append((tag, image, global_step, dataformats))


class _PLModule(pl.LightningModule):
    def __init__(self, observed):
        super().__init__()
        self.observed = observed
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


def _start(callback, trainer, module):
    callback.on_fit_start(trainer, module)
    callback.on_train_start(trainer, module)


def _run_step(callback, trainer, module, value, *, batch_idx, direct=False):
    callback.on_train_batch_start(trainer, module, None, batch_idx)
    if direct:
        module.observed.forward(value)
    else:
        module.observed(value)
    trainer.global_step += 1
    callback.on_train_batch_end(trainer, module, None, None, batch_idx)


class ObservationEventTest(unittest.TestCase):
    def test_canonical_names_keep_explicit_compatibility_aliases(self):
        self.assertIs(observation.ForwardObservation, observation.OutputObservation)
        self.assertIs(observation.PollObservation, observation.StateObservation)
        self.assertNotIn("ForwardObservation", observation.__all__)
        self.assertNotIn("PollObservation", observation.__all__)

    def test_heatmap_requires_a_nonempty_2d_float_view_and_valid_range(self):
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            observation.Heatmap(torch.zeros(1, 2, 3))
        with self.assertRaisesRegex(TypeError, "floating dtype"):
            observation.Heatmap(torch.zeros(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            observation.Heatmap(torch.empty(0, 3))
        with self.assertRaisesRegex(TypeError, "pair"):
            observation.Heatmap(torch.zeros(2, 3), value_range=cast(Any, [0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            observation.Heatmap(torch.zeros(2, 3), value_range=(1.0, 1.0))

    def test_emit_without_an_observer_is_a_noop_and_retains_no_tensor(self):
        module = nn.Identity()
        source = torch.tensor([[1.0, 2.0]], requires_grad=True)
        local = source.square()
        reference = weakref.ref(local)

        with mock.patch.object(
            torch,
            "isfinite",
            side_effect=AssertionError("no observer must not scan Heatmap values"),
        ):
            observation.emit(
                module,
                "undeclared",
                observation.Heatmap(local, value_range=(0.0, 4.0)),
            )

        local.sum().backward()
        assert source.grad is not None
        self.assertTrue(torch.equal(source.grad, 2 * source.detach()))
        del local
        gc.collect()
        self.assertIsNone(reference())

    def test_active_observer_rejects_an_undeclared_event(self):
        class EventModule(nn.Module):
            def forward(self, value):
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("known", reduction=observation.Reduction.Mean),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)

        with self.assertRaisesRegex(ValueError, "does not declare ForwardEvent"):
            observation.emit(
                module.observed,
                "unknown",
                observation.Curve(torch.tensor(1.0)),
            )

    def test_forward_event_curve_respects_render_cadence_and_direct_forward(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "stats",
                    observation.Curve(value.float().mean()),
                )
                return value * 2

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("stats", reduction=observation.Reduction.Mean),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=2)
        _start(callback, trainer, module)

        _run_step(callback, trainer, module, torch.tensor([1.0]), batch_idx=0)
        self.assertEqual(module.log_dict_calls, [])
        _run_step(
            callback,
            trainer,
            module,
            torch.tensor([2.0]),
            batch_idx=1,
            direct=True,
        )

        values, _kwargs = module.log_dict_calls[-1]
        self.assertEqual(values["observation/module/observed/stats/value"].item(), 2.0)

    def test_forward_event_history_includes_steps_between_renders(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "codes",
                    observation.Bins(value, sizes=(2,)),
                )
                return value

        observation.registry.register(
            EventModule,
            (
                observation.ForwardEvent(
                    "codes",
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(half_life_steps=1),
                ),
            ),
        )
        writer = _Writer()
        trainer = _trainer(writer)
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=2)
        _start(callback, trainer, module)

        _run_step(callback, trainer, module, torch.tensor([0, 0]), batch_idx=0)
        _run_step(callback, trainer, module, torch.tensor([1, 1, 1, 1]), batch_idx=1)

        ema = next(
            kwargs for tag, kwargs in writer.histograms if tag == "observation/module/observed/codes/value/ema"
        )
        self.assertEqual(ema["bucket_counts"], [1.0, 2.0])

    def test_forward_event_reduction_must_match_the_emitted_value_type(self):
        class MeanHeatmap(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            MeanHeatmap,
            (
                observation.ForwardEvent(
                    "map",
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(MeanHeatmap())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "does not support Heatmap"):
            module.observed(torch.zeros(2, 2))

        class MergeCurve(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "curve",
                    observation.Curve(value.float().mean()),
                )
                return value

        observation.registry.register(
            MergeCurve,
            (
                observation.ForwardEvent(
                    "curve",
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(),
                ),
            ),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(MergeCurve())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        with self.assertRaisesRegex(ValueError, "does not support Curve"):
            module.observed(torch.tensor([1.0]))

    def test_heatmap_keeps_last_emit_and_materializes_a_standalone_cpu_image(self):
        class EventModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.emitted_pointer = 0

            def forward(self, value):
                local = value.clone()
                self.emitted_pointer = local.data_ptr()
                observation.emit(
                    self,
                    "attention",
                    observation.Heatmap(local, value_range=(0.0, 3.0)),
                )
                local.fill_(99.0)
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("attention", reduction=observation.Reduction.Last),),
        )
        writer = _Writer()
        trainer = _trainer(writer)
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)

        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.full((2, 2), 3.0))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, 1)

        tag, image, step, dataformats = writer.images[-1]
        self.assertEqual(tag, "observation/module/observed/attention/value")
        self.assertEqual(step, 1)
        self.assertEqual(dataformats, "HW")
        self.assertEqual(image.device.type, "cpu")
        self.assertEqual(image.dtype, torch.float32)
        self.assertNotEqual(image.data_ptr(), module.observed.emitted_pointer)
        self.assertTrue(torch.equal(image, torch.ones(2, 2)))

    def test_heatmap_rejects_history_schema_changes_and_oversized_views(self):
        class HistoryModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            HistoryModule,
            (
                observation.ForwardEvent(
                    "map",
                    reduction=observation.Reduction.Merge,
                    history=observation.EMAHistory(),
                ),
            ),
        )
        history_trainer = _trainer(_Writer())
        history_module = _PLModule(HistoryModule())
        history_callback = ObservationCallback(every_n_steps=1)
        _start(history_callback, history_trainer, history_module)
        history_callback.on_train_batch_start(
            history_trainer,
            history_module,
            None,
            0,
        )
        with self.assertRaisesRegex(ValueError, "do not support EMAHistory"):
            history_module.observed(torch.zeros(2, 2))

        class MapModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            MapModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(MapModule())
        callback = ObservationCallback(every_n_steps=1, max_heatmap_elements=4)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.zeros(2, 3))
        trainer.global_step = 1
        with self.assertRaisesRegex(ValueError, "max_heatmap_elements=4"):
            callback.on_train_batch_end(trainer, module, None, None, 0)

        trainer = _trainer(_Writer())
        module = _PLModule(MapModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        _run_step(callback, trainer, module, torch.zeros(2, 2), batch_idx=0)
        callback.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.zeros(2, 3))
        trainer.global_step = 2
        with self.assertRaisesRegex(ValueError, "schema changed"):
            callback.on_train_batch_end(trainer, module, None, None, 1)

    def test_pending_heatmap_requires_image_backend(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )
        trainer = _trainer(object())
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.zeros(2, 2))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "provide add_image"):
            callback.on_train_batch_end(trainer, module, None, None, 0)

    def test_nonzero_rank_does_not_materialize_heatmap_pixels(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )

        class Strategy:
            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, value):
                self.all_gather_calls += 1
                return torch.stack((value, value))

        strategy = Strategy()
        trainer = _trainer(
            _Writer(),
            world_size=2,
            strategy=strategy,
            is_global_zero=False,
        )
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        self.assertEqual(callback.observed_modules, ("observed",))

        with mock.patch.object(
            observation_state,
            "_heatmap_image",
            side_effect=AssertionError("nonzero rank must not copy heatmap pixels"),
        ):
            _run_step(callback, trainer, module, torch.zeros(2, 2), batch_idx=0)
        self.assertEqual(strategy.all_gather_calls, 3)
        self.assertIn("observed/map/value", callback._states)
        self.assertEqual(trainer.logger.experiment.images, [])

    def test_distributed_heatmap_errors_are_synchronized_after_schema_preflight(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )

        class Strategy:
            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, value):
                self.all_gather_calls += 1
                return torch.stack((value, value))

        strategy = Strategy()
        writer = _Writer()
        trainer = _trainer(writer, world_size=2, strategy=strategy)
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([[float("nan")]]))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed Heatmap observation failed"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(strategy.all_gather_calls, 2)
        self.assertEqual(writer.images, [])

    def test_distributed_heatmap_image_discovery_failure_is_synchronized(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )

        class Strategy:
            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, value):
                self.all_gather_calls += 1
                return torch.stack((value, value))

        class DiscoveryWriter:
            def __init__(self):
                self.discovery_calls = 0
                self.add_image_calls = 0

            @property
            def add_image(self):
                self.discovery_calls += 1
                raise RuntimeError("image discovery exploded")

        strategy = Strategy()
        writer = DiscoveryWriter()
        trainer = _trainer(writer, world_size=2, strategy=strategy)
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.zeros(2, 2))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed Heatmap") as caught:
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertIn("image discovery exploded", str(caught.exception))
        self.assertEqual(strategy.all_gather_calls, 2)
        self.assertEqual(writer.discovery_calls, 1)
        self.assertEqual(writer.add_image_calls, 0)

    def test_distributed_heatmap_writer_failure_is_synchronized_across_ranks(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )

        class Strategy:
            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, value):
                self.all_gather_calls += 1
                if self.all_gather_calls == 3:
                    return torch.stack(
                        (
                            torch.ones_like(value),
                            torch.zeros_like(value),
                        )
                    )
                return torch.stack((value, value))

        class FailingWriter(_Writer):
            def __init__(self):
                super().__init__()
                self.add_image_calls = 0

            def add_image(self, tag, image, global_step, *, dataformats):
                del tag, image, global_step, dataformats
                self.add_image_calls += 1
                raise RuntimeError("writer exploded")

        owner_strategy = Strategy()
        owner_writer = FailingWriter()
        owner_trainer = _trainer(
            owner_writer,
            world_size=2,
            strategy=owner_strategy,
        )
        owner_module = _PLModule(EventModule())
        owner_callback = ObservationCallback(
            every_n_steps=1,
            sync_distributed=True,
        )
        _start(owner_callback, owner_trainer, owner_module)
        owner_callback.on_train_batch_start(owner_trainer, owner_module, None, 0)
        owner_module.observed(torch.zeros(2, 2))
        owner_trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed Heatmap") as owner_error:
            owner_callback.on_train_batch_end(
                owner_trainer,
                owner_module,
                None,
                None,
                0,
            )
        self.assertIn("writer exploded", str(owner_error.exception))
        self.assertEqual(owner_writer.add_image_calls, 1)
        self.assertEqual(owner_strategy.all_gather_calls, 3)

        other_strategy = Strategy()
        other_writer = FailingWriter()
        other_trainer = _trainer(
            other_writer,
            world_size=2,
            strategy=other_strategy,
            is_global_zero=False,
        )
        other_module = _PLModule(EventModule())
        other_callback = ObservationCallback(
            every_n_steps=1,
            sync_distributed=True,
        )
        _start(other_callback, other_trainer, other_module)
        other_callback.on_train_batch_start(other_trainer, other_module, None, 0)
        other_module.observed(torch.zeros(2, 2))
        other_trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "distributed Heatmap") as other_error:
            other_callback.on_train_batch_end(
                other_trainer,
                other_module,
                None,
                None,
                0,
            )
        self.assertIn("another rank", str(other_error.exception))
        self.assertEqual(other_writer.add_image_calls, 0)
        self.assertEqual(other_strategy.all_gather_calls, 3)

    def test_distributed_heatmap_schema_mismatch_precedes_image_logging(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "map",
                    observation.Heatmap(value, value_range=(0, 1)),
                )
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )

        class Strategy:
            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, value):
                self.all_gather_calls += 1
                other = value.clone()
                other[-1] += 1
                return torch.stack((value, other))

        strategy = Strategy()
        writer = _Writer()
        trainer = _trainer(writer, world_size=2, strategy=strategy)
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1, sync_distributed=True)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.zeros(2, 2))
        trainer.global_step = 1

        with self.assertRaisesRegex(RuntimeError, "same observation keys and schemas"):
            callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(strategy.all_gather_calls, 1)
        self.assertEqual(writer.images, [])
        self.assertEqual(
            observation_state._heatmap_schema(
                observation.Heatmap(torch.zeros(1, 1), value_range=(0, 1))
            ),
            observation_state._heatmap_schema(
                observation.Heatmap(torch.zeros(1, 1), value_range=(0.0, 1.0))
            ),
        )

    def test_event_and_heatmap_writer_time_are_in_observation_time(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)

        with mock.patch.object(
            observation_callback.time,
            "perf_counter",
            side_effect=(0.0, 0.1, 1.0, 1.2),
        ):
            _run_step(callback, trainer, module, torch.zeros(2, 2), batch_idx=0)

        values, _kwargs = module.log_dict_calls[-1]
        self.assertAlmostEqual(values["observation/time"].item(), 0.3)

    def test_compile_capture_and_detached_callbacks_leave_emit_as_a_noop(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(self, "map", observation.Heatmap(value))
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("map", reduction=observation.Reduction.Last),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)

        with (
            mock.patch.object(torch.compiler, "is_compiling", return_value=True),
            mock.patch.object(
                observation_state,
                "_heatmap_image",
                side_effect=AssertionError("compile capture must not observe events"),
            ),
        ):
            module.observed(torch.zeros(2, 2))
        self.assertEqual(callback._states, {})

        callback.on_fit_end(trainer, module)
        with mock.patch.object(
            observation_state,
            "_heatmap_image",
            side_effect=AssertionError("detached callback must not observe events"),
        ):
            module.observed(torch.zeros(2, 2))

    def test_multiple_callbacks_subscribe_and_release_independently(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "stats",
                    observation.Curve(value.float().mean()),
                )
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("stats", reduction=observation.Reduction.Mean),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(EventModule())
        first = ObservationCallback(every_n_steps=1, tag="first")
        second = ObservationCallback(every_n_steps=1, tag="second")
        _start(first, trainer, module)
        _start(second, trainer, module)

        first.on_train_batch_start(trainer, module, None, 0)
        second.on_train_batch_start(trainer, module, None, 0)
        module.observed(torch.tensor([1.0]))
        trainer.global_step = 1
        first.on_train_batch_end(trainer, module, None, None, 0)
        second.on_train_batch_end(trainer, module, None, None, 0)
        logged = {tag for values, _kwargs in module.log_dict_calls for tag in values}
        self.assertIn("observation/first/observed/stats/value", logged)
        self.assertIn("observation/second/observed/stats/value", logged)

        first.on_fit_end(trainer, module)
        second.on_train_batch_start(trainer, module, None, 1)
        module.observed(torch.tensor([2.0]))
        trainer.global_step = 2
        second.on_train_batch_end(trainer, module, None, None, 1)
        self.assertEqual(
            module.log_dict_calls[-1][0]["observation/second/observed/stats/value"].item(),
            2.0,
        )

    def test_dead_callback_event_sink_removes_itself(self):
        class EventModule(nn.Module):
            def forward(self, value):
                observation.emit(
                    self,
                    "stats",
                    observation.Curve(value.float().mean()),
                )
                return value

        observation.registry.register(
            EventModule,
            (observation.ForwardEvent("stats", reduction=observation.Reduction.Mean),),
        )
        trainer = _trainer(_Writer())
        module = _PLModule(EventModule())
        callback = ObservationCallback(every_n_steps=1)
        _start(callback, trainer, module)
        callback_reference = weakref.ref(callback)

        del callback
        gc.collect()
        self.assertIsNone(callback_reference())
        module.observed(torch.tensor([1.0]))

        runtime = observation_binding._runtimes[module.observed]
        self.assertEqual(runtime.event_sinks["stats"], {})


if __name__ == "__main__":
    unittest.main()
