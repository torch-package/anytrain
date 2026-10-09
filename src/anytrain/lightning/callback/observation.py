from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, cast
from weakref import ReferenceType, ref

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from anytrain import observation
from anytrain.observation import (
    BackwardObservation,
    BackwardObservationWarning,
    Bins,
    Curve,
    Distribution,
    ForwardEvent,
    Heatmap,
    Observation,
    ObservationResult,
    ObservationSpec,
    OutputObservation,
    Profile,
    Reduction,
    StateObservation,
)
from anytrain.observation._binding import claim, release, subscribe_event

from .. import _capability, experiment
from ._naming import join_tag, unique_tag_items, validate_tag_part
from ._observation_state import (
    _bins_delta,
    _CurveState,
    _decode_history,
    _distribution_delta,
    _encode_history,
    _HeatmapState,
    _HistogramState,
    _non_negative_int,
    _positive_int,
    _positive_real,
    _profile_delta,
    _ProfileState,
    _schema_signature,
    _StateEntry,
    _StateOwner,
)
from ._observation_writer import (
    _histogram_metrics,
    _merge_metrics,
    _profile_metrics,
    _write_heatmap,
    _write_histogram,
)

_SCHEMA_ERROR_INDEX = 2
_ObservationError = tuple[str, str]


@dataclass(frozen=True)
class _Collector:
    identifier: int
    module_path: str
    module: nn.Module
    spec: ObservationSpec

    @property
    def state_prefix(self) -> str:
        return join_tag(self.module_path or "<root>", self.spec.name)

    @property
    def tag_prefix(self) -> str:
        if not self.module_path:
            return self.spec.name
        return join_tag(self.module_path, self.spec.name)


class _ForwardCollectorHook:
    def __init__(
        self,
        callback: ObservationCallback,
        module: nn.Module,
        collector_id: int,
    ) -> None:
        self.callback: ReferenceType[ObservationCallback] = ref(callback)
        self.module_id = id(module)
        self.collector_id = collector_id

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: object,
    ) -> None:
        callback = self.callback()
        if callback is None or id(module) != self.module_id:
            _remove_observer_hooks(module, self.callback, self.collector_id)
            return
        collector = callback._collectors_by_id.get(self.collector_id)
        if collector is None:
            _remove_observer_hooks(module, self.callback, self.collector_id)
            return
        callback._forward(collector, module, inputs, kwargs, output)


_OBSERVATION_DEBUG = os.environ.get("ANYTRAIN_OBSERVATION_DEBUG") == "1"


def _observation_debug(event: str, **fields: object) -> None:
    """Env-gated trace of the collection pipeline (default off)."""

    if not _OBSERVATION_DEBUG:
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"observation.collector.{event} {payload}", file=sys.stderr, flush=True)


class _ForwardEventSink:
    def __init__(
        self,
        callback: ObservationCallback,
        module: nn.Module,
        collector_id: int,
    ) -> None:
        self.callback: ReferenceType[ObservationCallback] = ref(callback)
        self.module_id = id(module)
        self.collector_id = collector_id

    def __call__(self, module: nn.Module, value: ObservationResult) -> bool:
        callback = self.callback()
        if callback is None or id(module) != self.module_id:
            return False
        collector = callback._collectors_by_id.get(self.collector_id)
        if collector is None:
            return False
        callback._event(collector, value)
        return True


class ObservationCallback(pl.Callback):
    """Discover registered module observations and log current/EMA statistics."""

    def __init__(
        self,
        *,
        every_n_steps: int | None = 100,
        tag: str = "module",
        half_life_steps: float | None = None,
        sync_distributed: bool = False,
        max_dense_bins: int = 1_000_000,
        max_heatmap_elements: int = 65_536,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        self.every_n_steps = every_n_steps
        self.tag = validate_tag_part(tag)
        self.half_life_steps = (
            None
            if half_life_steps is None
            else _positive_real(half_life_steps, "half_life_steps")
        )
        if not isinstance(sync_distributed, bool):
            raise TypeError("sync_distributed must be bool.")
        self.sync_distributed = sync_distributed
        self.max_dense_bins = _positive_int(max_dense_bins, "max_dense_bins")
        self.max_heatmap_elements = _positive_int(
            max_heatmap_elements,
            "max_heatmap_elements",
        )
        self._collectors: list[_Collector] = []
        self._collectors_by_id: dict[int, _Collector] = {}
        self._handles: list[RemovableHandle] = []
        self._claimed: list[nn.Module] = []
        self._states: dict[str, _StateEntry] = {}
        self._output_tag_owners: dict[str, str] = {}
        self._emitted_keys: set[str] = set()
        self._enabled = False
        self._warning_rank = False
        self._defer_observation_errors = False
        self._pending_observation_error: _ObservationError | None = None
        self._backward_warning_emitted = False
        self._forward_active = False
        self._render_active = False
        self._pending_observation_time = 0.0
        self._observation_time_sum = 0.0
        self._observation_time_steps = 0
        self._last_global_step: int | None = None
        self._train_optimizer_steps = 0
        self._validation_step = 0
        self._restored_last_global_step: int | None = None
        self._active_stage: Literal["train", "validation"] | None = None
        self.observed_modules: tuple[str, ...] = ()

    @property
    def state_key(self) -> str:
        return f"{type(self).__qualname__}[tag={self.tag}]"

    def _require_active_stage(self) -> Literal["train", "validation"]:
        if self._active_stage is None:
            raise RuntimeError("observation forward has no active Trainer stage.")
        return self._active_stage

    def _context(self, stage: Literal["train", "validation"]) -> str:
        prefix = join_tag(*observation.context_prefix())
        stage_prefix = "train" if stage == "train" else "val"
        if not prefix:
            return "" if stage == "train" else stage_prefix
        if prefix == stage_prefix or prefix.startswith(f"{stage_prefix}/"):
            return prefix
        return join_tag(stage_prefix, prefix)

    def _tag_context(self, stage: Literal["train", "validation"]) -> str:
        context = self._context(stage)
        return join_tag("observation", context) if context else "observation"

    def _stage_keys(self, stage: Literal["train", "validation"]) -> tuple[str, ...]:
        return tuple(
            key for key, entry in self._states.items() if entry.owner.stage == stage
        )

    def _pending_heatmaps(
        self,
        stage: Literal["train", "validation"],
    ) -> tuple[_HeatmapState, ...]:
        pending: list[_HeatmapState] = []
        for key in self._stage_keys(stage):
            state = self._states[key].state
            if isinstance(state, _HeatmapState) and state.pending:
                pending.append(state)
        return tuple(pending)

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._enabled or self._collectors or self._claimed:
            return
        self._warning_rank = _capability.is_global_zero(trainer)
        self._defer_observation_errors = (
            self.sync_distributed and _capability.is_distributed(trainer)
        )
        self._enabled = self.sync_distributed or self._warning_rank
        if not self._enabled:
            return
        try:
            self._attach(pl_module)
            self._bind_restored_tags()
        except BaseException:
            self._detach()
            raise

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self._enabled:
            return
        step = _capability.global_step(trainer)
        restored = self._restored_last_global_step
        # A checkpoint dumped by an earlier-ordered callback (e.g. checkpointing)
        # captures this callback's state before its on_train_batch_end ran for the
        # saved step, so the stored step legitimately lags by one.
        if restored is not None and restored not in (step, step - 1):
            raise ValueError(
                "ObservationCallback checkpoint global step does not match Trainer: "
                f"got {restored} and {step}."
            )
        if restored is None:
            ema_steps: list[int] = []
            for entry in self._states.values():
                state = entry.state
                if (
                    entry.owner.stage == "train"
                    and isinstance(state, (_HistogramState, _ProfileState))
                    and state.ema_step is not None
                ):
                    ema_steps.append(state.ema_step)
            if ema_steps and step < max(ema_steps):
                raise ValueError(
                    "ObservationCallback cannot reuse history after Trainer global_step "
                    "was reset; create a fresh callback for a new training run."
                )
        self._last_global_step = step
        self._restored_last_global_step = None
        for entry in self._states.values():
            if not isinstance(entry.state, _HeatmapState):
                entry.state._to(pl_module.device)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        if not self._enabled:
            return
        self._active_stage = "train"
        self._forward_active = True
        step = _capability.global_step(trainer)
        self._train_optimizer_steps = 0
        self._render_active = self._should_render_step(step + 1)

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del optimizer
        if not self._enabled or self._active_stage != "train":
            return
        if self._last_global_step is None:
            self._last_global_step = _capability.global_step(trainer)
        started_at = self._start_observation_timer()
        step = self._last_global_step + 1
        self._finalize_train_step(
            trainer,
            pl_module,
            step=step,
            started_at=started_at,
        )
        self._train_optimizer_steps += 1
        self._render_active = self._should_render_step(step + 1)

    def _should_render_step(self, step: int) -> bool:
        return self.every_n_steps is not None and step % self.every_n_steps == 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not self._enabled:
            return
        self._forward_active = False
        step = _capability.global_step(trainer)
        if self._train_optimizer_steps:
            if self._last_global_step != step:
                raise RuntimeError(
                    "ObservationCallback optimizer-step hook disagrees with "
                    f"trainer.global_step: callback={self._last_global_step}, "
                    f"trainer={step}."
                )
            self._discard_pending()
            return
        started_at = self._start_observation_timer()
        previous = self._last_global_step
        if previous is None:
            self._last_global_step = step
            self._active_stage = None
            return
        optimizer_steps = step - previous
        if optimizer_steps < 0:
            raise RuntimeError("trainer.global_step moved backwards during observation.")
        if optimizer_steps == 0:
            self._add_observation_time(started_at)
            self._active_stage = None
            return
        if optimizer_steps != 1:
            raise RuntimeError(
                "ObservationCallback requires on_before_optimizer_step to partition "
                "one train batch across multiple optimizer steps."
            )
        self._finalize_train_step(
            trainer,
            pl_module,
            step=step,
            started_at=started_at,
        )
        self._active_stage = None

    def _finalize_train_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        step: int,
        started_at: float | None,
    ) -> None:
        self._poll("train")
        self._validate_distributed_schema(trainer, pl_module)
        for key in sorted(self._stage_keys("train")):
            state = self._states[key].state
            if isinstance(state, _CurveState):
                state.finalize(trainer, sync_distributed=self.sync_distributed)
            elif isinstance(state, (_HistogramState, _ProfileState)):
                state.finalize(
                    trainer,
                    step=step,
                    sync_distributed=self.sync_distributed,
                )
        self._last_global_step = step
        self._emitted_keys.clear()
        self._render_active = False
        _validate_histograms(self._states, device=pl_module.device)
        _validate_profiles(self._states, device=pl_module.device)
        if not self._should_render_step(step):
            self._finish_observation_step(started_at)
            return
        self._log(
            trainer,
            pl_module,
            stage="train",
            step=step,
            started_at=started_at,
            on_step=True,
            on_epoch=False,
        )

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if not self._enabled or bool(getattr(trainer, "sanity_checking", False)):
            return
        self._active_stage = "validation"
        self._forward_active = True
        self._render_active = True

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self._enabled or self._active_stage != "validation":
            return
        self._forward_active = False
        self._poll("validation")
        self._validate_distributed_schema(trainer, pl_module)
        self._validation_step += 1
        for key in sorted(self._stage_keys("validation")):
            state = self._states[key].state
            if isinstance(state, _CurveState):
                state.finalize(trainer, sync_distributed=self.sync_distributed)
            elif isinstance(state, (_HistogramState, _ProfileState)):
                state.finalize(
                    trainer,
                    step=self._validation_step,
                    sync_distributed=self.sync_distributed,
                )
        self._emitted_keys.clear()
        self._render_active = False
        self._active_stage = None
        _validate_histograms(self._states, device=pl_module.device)
        _validate_profiles(self._states, device=pl_module.device)
        self._log(
            trainer,
            pl_module,
            stage="validation",
            step=_capability.global_step(trainer),
            started_at=None,
            on_step=False,
            on_epoch=True,
        )

    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: Tensor,
    ) -> None:
        del trainer, pl_module, loss
        if not self._enabled:
            return
        # Activation checkpointing may replay module forward during backward.
        self._forward_active = False

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not self._enabled:
            return
        self._forward_active = self._active_stage == "train"

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not self._enabled:
            return
        self._discard_pending()

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not self._enabled:
            return
        self._discard_pending()
        self._detach()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        if not self._enabled:
            return
        self._discard_pending()
        self._detach()

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        del trainer, pl_module, stage
        if not self._enabled:
            return
        self._discard_pending()
        self._detach()

    def state_dict(self) -> dict[str, Any]:
        histograms, profiles, owners = _encode_history(self._states)
        return {
            "version": 4,
            "tag": self.tag,
            "every_n_steps": self.every_n_steps,
            "half_life_steps": self.half_life_steps,
            "sync_distributed": self.sync_distributed,
            "max_dense_bins": self.max_dense_bins,
            "max_heatmap_elements": self.max_heatmap_elements,
            "last_global_step": self._last_global_step,
            "validation_step": self._validation_step,
            "histograms": histograms,
            "profiles": profiles,
            "owners": owners,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        version = state_dict.get("version")
        if version not in {2, 3, 4}:
            raise ValueError("unsupported ObservationCallback state version.")
        if state_dict.get("tag") != self.tag:
            raise ValueError("ObservationCallback state tag mismatch.")
        expected = {
            "half_life_steps": self.half_life_steps,
            "sync_distributed": self.sync_distributed,
        }
        for name, value in expected.items():
            if state_dict.get(name) != value:
                raise ValueError(f"ObservationCallback state {name} mismatch.")
        last_global_step = state_dict.get("last_global_step")
        if last_global_step is not None:
            last_global_step = _non_negative_int(
                last_global_step,
                "ObservationCallback last_global_step",
            )
        validation_step = (
            0
            if version < 4
            else _non_negative_int(
                state_dict.get("validation_step"),
                "ObservationCallback validation_step",
            )
        )
        loaded = _decode_history(
            state_dict, version=version, max_dense_bins=self.max_dense_bins,
            last_global_step=last_global_step, validation_step=validation_step,
        )
        self._states = loaded
        self._validation_step = validation_step
        self._output_tag_owners.clear()
        self._emitted_keys.clear()
        self._restored_last_global_step = last_global_step
        if self._collectors:
            self._bind_restored_tags()

    def _attach(self, pl_module: pl.LightningModule) -> None:
        registered: list[tuple[str, nn.Module, tuple[ObservationSpec, ...]]] = []
        for name, module in pl_module.named_modules():
            declared = observation.registry.specs(module)
            if declared:
                registered.append((name, module, declared))
        for _name, module, declared in registered:
            claim(module, declared)
            self._claimed.append(module)

        covered: list[str] = []
        observed: list[str] = []
        for module_path, module, specs in registered:
            if any(_is_descendant(module_path, parent) for parent in covered):
                continue
            supported: list[ObservationSpec] = []
            for spec in specs:
                if isinstance(spec, BackwardObservation):
                    self._warn_backward_observation()
                else:
                    supported.append(spec)
            if not supported:
                continue
            observed.append(module_path or "module")
            for spec in supported:
                collector = _Collector(len(self._collectors), module_path, module, spec)
                self._collectors.append(collector)
                self._collectors_by_id[collector.identifier] = collector
                self._attach_collector(collector)
            if any(spec.covers_descendants for spec in supported):
                covered.append(module_path)
        self.observed_modules = tuple(observed)

    def _warn_backward_observation(self) -> None:
        if not self._warning_rank or self._backward_warning_emitted:
            return
        self._backward_warning_emitted = True
        warnings.warn(
            "BackwardObservation is reserved but is not implemented by "
            "ObservationCallback yet. Use "
            "anytrain.lightning.GradientProbeLoggerCallback to observe loss "
            "gradients with respect to selected parameters.",
            BackwardObservationWarning,
            stacklevel=5,
        )

    def _attach_collector(self, collector: _Collector) -> None:
        spec = collector.spec
        if isinstance(spec, OutputObservation):
            handle = collector.module.register_forward_hook(
                _ForwardCollectorHook(self, collector.module, collector.identifier),
                with_kwargs=True,
            )
            self._handles.append(handle)
        elif isinstance(spec, ForwardEvent):
            self._handles.append(
                subscribe_event(
                    collector.module,
                    spec,
                    _ForwardEventSink(
                        self,
                        collector.module,
                        collector.identifier,
                    ),
                )
            )

    def _forward(
        self,
        collector: _Collector,
        module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: object,
    ) -> None:
        if not self._forward_active:
            return
        stage = self._require_active_stage()
        spec = cast(OutputObservation, collector.spec)
        if spec.history is None and not self._render_active:
            return
        started_at = self._start_observation_timer()
        try:
            self._collect_and_record(
                collector,
                lambda: spec.collect(module, inputs, kwargs, output),
                stage=stage,
            )
        finally:
            self._add_observation_time(started_at)

    def _event(
        self,
        collector: _Collector,
        value: ObservationResult,
    ) -> None:
        if not self._forward_active:
            _observation_debug(
                "event.skip.forward_inactive", prefix=collector.state_prefix
            )
            return
        stage = self._require_active_stage()
        spec = cast(ForwardEvent, collector.spec)
        if spec.history is None and not self._render_active:
            _observation_debug(
                "event.skip.render_inactive",
                prefix=collector.state_prefix,
                stage=stage,
            )
            return
        started_at = self._start_observation_timer()
        try:
            self._collect_and_record(collector, lambda: value, stage=stage)
        finally:
            self._add_observation_time(started_at)

    def _poll(self, stage: Literal["train", "validation"]) -> None:
        for collector in self._collectors:
            spec = collector.spec
            if isinstance(spec, StateObservation):
                if spec.history is None and not self._render_active:
                    continue
                self._collect_and_record(
                    collector,
                    lambda spec=spec, module=collector.module: spec.collect(module),
                    stage=stage,
                )

    def _collect_and_record(
        self,
        collector: _Collector,
        collect: Callable[[], ObservationResult],
        *,
        stage: Literal["train", "validation"],
    ) -> None:
        if self._pending_observation_error is not None:
            _observation_debug(
                "collector.skip.pending_error",
                prefix=collector.state_prefix,
                stage=stage,
            )
            return
        try:
            self._record(collector, collect(), stage=stage)
        except Exception as error:
            if not self._defer_observation_errors:
                raise
            _observation_debug(
                "collector.collect_failed",
                prefix=collector.state_prefix,
                stage=stage,
                error=f"{type(error).__name__}: {error}",
            )
            self._pending_observation_error = (
                "runtime",
                f"observation collector {collector.state_prefix!r} failed: "
                f"{type(error).__name__}: {error}",
            )

    def _record(
        self,
        collector: _Collector,
        result: ObservationResult,
        *,
        stage: Literal["train", "validation"],
    ) -> None:
        if result is None:
            _observation_debug(
                "record.skip.none", prefix=collector.state_prefix, stage=stage
            )
            return
        if isinstance(result, (Curve, Distribution, Profile, Bins, Heatmap)):
            values: Mapping[str, Observation] = {"value": result}
        elif isinstance(result, Mapping):
            values = result
        else:
            raise TypeError("observation collector must return an Observation, mapping, or None.")
        for item, value in unique_tag_items(
            values.items(),
            name="observation value name",
        ):
            if not isinstance(value, (Curve, Distribution, Profile, Bins, Heatmap)):
                raise TypeError("observation mapping values must be Observation objects.")
            if isinstance(value, Heatmap):
                if collector.spec.history is not None:
                    raise ValueError(
                        "Heatmap observations do not support EMAHistory; declare a "
                        "separate no-history observation."
                    )
                if not self._render_active:
                    continue
            context = self._context(stage)
            tag_context = self._tag_context(stage)
            key = join_tag(context, collector.state_prefix, item)
            tag = join_tag(tag_context, self.tag, collector.tag_prefix, item)
            _observation_debug(
                "record.write",
                prefix=collector.state_prefix,
                stage=stage,
                context=context,
                tag=tag,
                item=item,
            )
            self._update_state(
                key,
                tag,
                value,
                collector.spec,
                owner=_collector_owner(
                    collector,
                    item,
                    stage=stage,
                    context=context,
                ),
            )

    def _update_state(
        self,
        key: str,
        tag: str,
        value: Observation,
        spec: ObservationSpec,
        *,
        owner: _StateOwner,
    ) -> None:
        entry = self._states.get(key)
        reduction = _observation_reduction(spec, value)
        if isinstance(value, Curve):
            if entry is None:
                state = _CurveState(value, reduction=reduction)
                self._bind_state_identity(key, tag, state, owner)
                self._states[key] = _StateEntry(state, tag, owner)
                self._emitted_keys.add(key)
                return
            if not isinstance(entry.state, _CurveState):
                raise TypeError(f"observation {key!r} changed value type.")
            self._bind_state_identity(key, tag, entry.state, owner)
            entry.state.update(value, reduction=reduction)
            self._emitted_keys.add(key)
            return

        if isinstance(value, Heatmap):
            if entry is None:
                state = _HeatmapState(
                    value,
                    keep_image=self._warning_rank,
                    max_elements=self.max_heatmap_elements,
                )
                self._bind_state_identity(key, tag, state, owner)
                self._states[key] = _StateEntry(state, tag, owner)
                self._emitted_keys.add(key)
                return
            if not isinstance(entry.state, _HeatmapState):
                raise TypeError(f"observation {key!r} changed value type.")
            self._bind_state_identity(key, tag, entry.state, owner)
            entry.state.update(
                value,
                keep_image=self._warning_rank,
                max_elements=self.max_heatmap_elements,
            )
            self._emitted_keys.add(key)
            return

        history = spec.history
        if history is None:
            raise ValueError(
                f"observation {key!r} requires EMAHistory for Distribution, Profile, or Bins."
            )
        half_life_steps = (
            float(history.half_life_steps)
            if self.half_life_steps is None
            else self.half_life_steps
        )
        dense_bins = (
            len(value.sizes) * max(value.sizes)
            if isinstance(value, Bins)
            else len(value.edges) - 1
        )
        if dense_bins > self.max_dense_bins:
            raise ValueError(
                f"observation {key!r} requires {dense_bins} dense bins, exceeding "
                f"max_dense_bins={self.max_dense_bins}; use a factorized observation or raise "
                "the limit explicitly."
            )
        if isinstance(value, Profile):
            delta = _profile_delta(value)
            if entry is None:
                state = _ProfileState(delta, half_life_steps=half_life_steps)
                self._bind_state_identity(key, tag, state, owner)
                self._states[key] = _StateEntry(state, tag, owner)
                self._emitted_keys.add(key)
                return
            if not isinstance(entry.state, _ProfileState):
                raise TypeError(f"observation {key!r} changed value type.")
            self._bind_state_identity(key, tag, entry.state, owner)
            entry.state.update(delta, half_life_steps=half_life_steps)
            self._emitted_keys.add(key)
            return

        delta = (
            _distribution_delta(value)
            if isinstance(value, Distribution)
            else _bins_delta(value)
        )
        if entry is None:
            state = _HistogramState(delta, half_life_steps=half_life_steps)
            self._bind_state_identity(key, tag, state, owner)
            self._states[key] = _StateEntry(
                state,
                tag,
                owner,
            )
            self._emitted_keys.add(key)
            return
        if not isinstance(entry.state, _HistogramState):
            raise TypeError(f"observation {key!r} changed value type.")
        self._bind_state_identity(key, tag, entry.state, owner)
        entry.state.update(delta, half_life_steps=half_life_steps)
        self._emitted_keys.add(key)

    def _bind_state_identity(
        self,
        key: str,
        tag: str,
        state: _CurveState | _HeatmapState | _HistogramState | _ProfileState,
        owner: _StateOwner,
    ) -> None:
        entry = self._states.get(key)
        if entry is not None and entry.owner != owner:
            raise ValueError(
                f"observation sources {entry.owner!r} and {owner!r} resolve to the "
                f"same state key {key!r}."
            )
        if entry is not None and entry.tag and entry.tag != tag:
            if tag != join_tag("observation", entry.tag):
                raise ValueError(
                    f"observation {key!r} changed logger tag from {entry.tag!r} to {tag!r}."
                )
            entry.tag = tag
        output_tags = _state_output_tags(tag, state)
        for output_tag in output_tags:
            previous_key = self._output_tag_owners.get(output_tag)
            if previous_key is not None and previous_key != key:
                raise ValueError(
                    f"observations {previous_key!r} and {key!r} resolve to the same "
                    f"logger tag {output_tag!r}."
                )
        for output_tag in output_tags:
            self._output_tag_owners[output_tag] = key

    def _log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        stage: Literal["train", "validation"],
        step: int,
        started_at: float | None,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        metrics: dict[str, Tensor] = {}
        stage_keys = self._stage_keys(stage)
        has_pending_heatmap = bool(self._pending_heatmaps(stage))
        image_writer: experiment.Image | None = None
        image_error: tuple[Literal["runtime"], str] | None = None
        if has_pending_heatmap and _capability.is_global_zero(trainer):
            try:
                image_writer = experiment.image(trainer)
            except Exception as error:
                image_error = (
                    "runtime",
                    "heatmap image writer discovery failed: "
                    f"{type(error).__name__}: {error}",
                )
        self._validate_heatmap_logging(
            trainer,
            pl_module,
            stage=stage,
            image_writer=image_writer,
            image_error=image_error,
        )
        write_error = self._write_heatmaps(
            trainer,
            stage=stage,
            image_writer=image_writer,
            step=step,
        )
        self._validate_heatmap_write(
            trainer,
            pl_module,
            pending=has_pending_heatmap,
            error=write_error,
        )
        raw_writer: experiment.HistogramRaw | None = None
        raw_error: _ObservationError | None = None
        has_raw_histogram = any(
            isinstance(self._states[key].state, _HistogramState)
            and bool(self._states[key].tag)
            for key in stage_keys
        )
        if has_raw_histogram and _capability.is_global_zero(trainer):
            try:
                raw_writer = experiment.histogram_raw(trainer)
            except Exception as error:
                raw_error = (
                    "runtime",
                    "raw histogram writer discovery failed: "
                    f"{type(error).__name__}: {error}",
                )
        write_error = raw_error
        for key in sorted(stage_keys):
            entry = self._states[key]
            if not entry.tag:
                continue
            state = entry.state
            if isinstance(state, _CurveState):
                value = state.value()
                if value is not None:
                    _merge_metrics(metrics, {entry.tag: value})
                continue
            if isinstance(state, _HeatmapState):
                continue
            if isinstance(state, _ProfileState):
                _merge_metrics(
                    metrics,
                    _profile_metrics(entry.tag, "step", state.current()),
                )
                _merge_metrics(
                    metrics,
                    _profile_metrics(entry.tag, "ema", state.ema()),
                )
                continue
            current = state.current()
            ema = state.ema()
            _merge_metrics(metrics, _histogram_metrics(entry.tag, "step", current))
            _merge_metrics(metrics, _histogram_metrics(entry.tag, "ema", ema))
            if raw_writer is not None and write_error is None:
                try:
                    _write_histogram(raw_writer, entry.tag, "step", current, step=step)
                    _write_histogram(raw_writer, entry.tag, "ema", ema, step=step)
                except Exception as error:
                    write_error = (
                        "runtime",
                        f"raw histogram write failed for {entry.tag!r}: "
                        f"{type(error).__name__}: {error}",
                    )
        self._synchronize_runtime_error(
            trainer,
            pl_module,
            pending=has_raw_histogram,
            error=write_error,
            context="raw histogram logging",
            collective_error=(
                "Distributed raw histogram logging requires "
                "trainer.strategy.all_gather; set sync_distributed=False for "
                "rank-local diagnostics."
            ),
        )
        if stage == "train":
            self._finish_observation_step(started_at)
        if stage == "train" and self._observation_time_steps > 0:
            _merge_metrics(
                metrics,
                {
                    "observation/time": torch.tensor(
                        self._observation_time_sum / self._observation_time_steps,
                        dtype=torch.float32,
                        device=pl_module.device,
                    )
                },
            )
            self._observation_time_sum = 0.0
            self._observation_time_steps = 0
        if metrics:
            pl_module.log_dict(
                metrics,
                on_step=on_step,
                on_epoch=on_epoch,
                logger=True,
                sync_dist=False,
            )

    def _write_heatmaps(
        self,
        trainer: pl.Trainer,
        *,
        stage: Literal["train", "validation"],
        image_writer: experiment.Image | None,
        step: int,
    ) -> tuple[Literal["runtime"], str] | None:
        error: tuple[Literal["runtime"], str] | None = None
        for key in sorted(self._stage_keys(stage)):
            entry = self._states[key]
            state = entry.state
            if not isinstance(state, _HeatmapState) or not state.pending:
                continue
            if _capability.is_global_zero(trainer) and error is None:
                try:
                    value = state.value()
                    if value is None:
                        raise RuntimeError(
                            "global-zero heatmap observation did not retain an image."
                        )
                    if image_writer is None:
                        raise RuntimeError(
                            "heatmap image writer preflight was inconsistent."
                        )
                    _write_heatmap(
                        image_writer,
                        entry.tag,
                        value,
                        step=step,
                    )
                except Exception as caught:
                    error = (
                        "runtime",
                        f"heatmap image write failed for {entry.tag!r}: "
                        f"{type(caught).__name__}: {caught}",
                    )
            state.discard_pending()
        return error

    def _bind_restored_tags(self) -> None:
        self._output_tag_owners.clear()
        collectors = {
            (collector.module_path, collector.spec.name.strip("/")): collector
            for collector in self._collectors
        }
        for key, entry in self._states.items():
            owner = entry.owner
            collector = collectors.get((owner.module_path, owner.spec_name))
            if collector is None:
                raise ValueError(
                    f"restored observation {key!r} does not match the current module tree."
                )
            expected_key = join_tag(owner.context, collector.state_prefix, owner.item)
            if expected_key != key:
                raise ValueError(
                    f"restored observation {key!r} has incompatible source identity."
                )
            if isinstance(entry.state, (_HistogramState, _ProfileState)):
                history = collector.spec.history
                if history is None:
                    raise ValueError(
                        f"restored observation {key!r} no longer declares EMA history."
                    )
                half_life_steps = (
                    float(history.half_life_steps)
                    if self.half_life_steps is None
                    else self.half_life_steps
                )
                if entry.state.half_life_steps != half_life_steps:
                    raise ValueError(
                        f"restored observation {key!r} EMA half-life changed."
                    )
            tag = join_tag(owner.context, self.tag, collector.tag_prefix, owner.item)
            self._bind_state_identity(
                key,
                tag,
                entry.state,
                owner,
            )
            entry.tag = tag

    def _validate_distributed_schema(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.sync_distributed or not _capability.is_distributed(trainer):
            return
        signature, local_error = self._distributed_schema_signature(pl_module)
        gathered = _all_gather(
            trainer,
            signature,
            error=(
                "Distributed module observation requires trainer.strategy.all_gather; "
                "set sync_distributed=False for rank-local diagnostics."
            ),
        )
        if gathered.numel() % signature.numel() != 0:
            raise RuntimeError(
                "trainer.strategy.all_gather returned an invalid observation schema payload."
            )
        rows = gathered.detach().cpu().reshape(-1, signature.numel())
        if bool((rows[:, _SCHEMA_ERROR_INDEX] > 0).any().item()):
            _raise_synchronized_error(
                local_error,
                context="module observation",
            )
        expected = signature.detach().cpu()
        expected[_SCHEMA_ERROR_INDEX] = 0
        rows[:, _SCHEMA_ERROR_INDEX] = 0
        if any(not torch.equal(row, expected) for row in rows):
            raise RuntimeError(
                "sync_distributed=True requires every rank to emit the same observation "
                "keys and schemas for each optimizer step."
            )

    def _distributed_schema_signature(
        self,
        pl_module: pl.LightningModule,
    ) -> tuple[Tensor, _ObservationError | None]:
        local_error = self._pending_observation_error
        try:
            signature = _schema_signature(
                self._states,
                emitted_keys=self._emitted_keys,
                device=pl_module.device,
                observation_error=local_error is not None,
            )
        except Exception as error:
            if not self._defer_observation_errors:
                raise
            local_error = (
                "runtime",
                "observation schema materialization failed: "
                f"{type(error).__name__}: {error}",
            )
            signature = _schema_signature(
                {},
                emitted_keys=set(),
                device=pl_module.device,
                observation_error=True,
            )
        return signature, local_error

    def _validate_heatmap_logging(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        stage: Literal["train", "validation"],
        image_writer: experiment.Image | None,
        image_error: tuple[Literal["runtime"], str] | None,
    ) -> None:
        pending = self._pending_heatmaps(stage)
        if not pending:
            return
        errors = [state.pending_error for state in pending if state.pending_error]
        if image_error is not None:
            errors.append(image_error)
        elif self._warning_rank and image_writer is None:
            errors.append(
                (
                    "runtime",
                    "Heatmap observation requires the Trainer logger experiment "
                    "to provide add_image().",
                )
            )
        if self._warning_rank and any(
            state.pending_image is None and state.pending_error is None
            for state in pending
        ):
            errors.append(
                (
                    "runtime",
                    "global-zero heatmap observation did not retain an image.",
                )
            )

        if self.sync_distributed and _capability.is_distributed(trainer):
            local_error = torch.tensor(
                [int(bool(errors))],
                dtype=torch.int64,
                device=pl_module.device,
            )
            gathered = _all_gather(
                trainer,
                local_error,
                error=(
                    "Distributed Heatmap observation requires "
                    "trainer.strategy.all_gather; set sync_distributed=False for "
                    "rank-local diagnostics."
                ),
            )
            if bool((gathered.detach() > 0).any().item()):
                if errors:
                    detail = "; ".join(message for _kind, message in errors)
                    raise RuntimeError(
                        "distributed Heatmap observation failed: " + detail
                    )
                raise RuntimeError(
                    "distributed Heatmap observation failed on another rank."
                )
            return

        if not errors:
            return
        kind, message = errors[0]
        if kind == "value":
            raise ValueError(message)
        raise RuntimeError(message)

    def _validate_heatmap_write(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        pending: bool,
        error: tuple[Literal["runtime"], str] | None,
    ) -> None:
        if not pending:
            return
        self._synchronize_runtime_error(
            trainer,
            pl_module,
            pending=pending,
            error=error,
            context="Heatmap image logging",
            collective_error=(
                "Distributed Heatmap image logging requires "
                "trainer.strategy.all_gather; set sync_distributed=False for "
                "rank-local diagnostics."
            ),
        )

    def _synchronize_runtime_error(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        pending: bool,
        error: _ObservationError | None,
        context: str,
        collective_error: str,
    ) -> None:
        if not pending:
            return
        if self.sync_distributed and _capability.is_distributed(trainer):
            local_error = torch.tensor(
                [int(error is not None)],
                dtype=torch.int64,
                device=pl_module.device,
            )
            gathered = _all_gather(trainer, local_error, error=collective_error)
            if bool((gathered.detach() > 0).any().item()):
                _raise_synchronized_error(error, context=context)
            return
        if error is not None:
            raise RuntimeError(f"{context} failed: {error[1]}")

    def _discard_pending(self) -> None:
        self._active_stage = None
        self._forward_active = False
        self._render_active = False
        self._train_optimizer_steps = 0
        self._pending_observation_time = 0.0
        self._emitted_keys.clear()
        self._pending_observation_error = None
        for entry in self._states.values():
            entry.state.discard_pending()

    def _detach(self) -> None:
        self._enabled = False
        self._warning_rank = False
        self._defer_observation_errors = False
        self._pending_observation_error = None
        self._active_stage = None
        self._forward_active = False
        self._render_active = False
        self._train_optimizer_steps = 0
        self._pending_observation_time = 0.0
        self._observation_time_sum = 0.0
        self._observation_time_steps = 0
        while self._handles:
            self._handles.pop().remove()
        while self._claimed:
            release(self._claimed.pop())
        self._collectors.clear()
        self._collectors_by_id.clear()
        self._output_tag_owners.clear()
        self._emitted_keys.clear()

    def _start_observation_timer(self) -> float | None:
        if (
            self._active_stage != "train"
            or self.every_n_steps is None
            or not self._collectors
        ):
            return None
        return time.perf_counter()

    def _add_observation_time(self, started_at: float | None) -> None:
        if started_at is None:
            return
        self._pending_observation_time += time.perf_counter() - started_at

    def _finish_observation_step(self, started_at: float | None) -> None:
        self._add_observation_time(started_at)
        if self.every_n_steps is None or not self._collectors:
            return
        self._observation_time_sum += self._pending_observation_time
        self._observation_time_steps += 1
        self._pending_observation_time = 0.0


def _state_output_tags(
    tag: str,
    state: _CurveState | _HeatmapState | _HistogramState | _ProfileState,
) -> tuple[str, ...]:
    if isinstance(state, (_CurveState, _HeatmapState)):
        return (tag,)
    if isinstance(state, _ProfileState):
        profile_tags: list[str] = []
        for scope in ("step", "ema"):
            scope_tag = join_tag(tag, scope)
            profile_tags.extend(
                (
                    join_tag(scope_tag, "observations"),
                    join_tag(scope_tag, "active_ratio"),
                )
            )
            for index in range(state.schema.bins):
                bin_tag = join_tag(scope_tag, f"bin_{index}")
                profile_tags.extend(
                    (
                        join_tag(bin_tag, "count"),
                        join_tag(bin_tag, "mean"),
                        join_tag(bin_tag, "std"),
                    )
                )
        return tuple(profile_tags)
    result: list[str] = []
    for group_name in state.schema.group_names:
        for scope in ("step", "ema"):
            scope_tag = join_tag(tag, group_name, scope)
            result.append(scope_tag)
            metrics = [
                "observations",
                "active_ratio",
                "entropy",
                "perplexity",
            ]
            if state.schema.kind == "bins":
                metrics.extend(
                    (
                        "occupied_count",
                        "occupied_ratio",
                        "dead_count",
                        "dead_ratio",
                    )
                )
            result.extend(join_tag(scope_tag, metric) for metric in metrics)
    if len(set(result)) != len(result):
        raise RuntimeError("observation state produced duplicate logger tags.")
    return tuple(result)


def _observation_reduction(
    spec: ObservationSpec,
    value: Observation,
) -> Reduction:
    if isinstance(spec, (OutputObservation, ForwardEvent)):
        reduction = spec.reduction
    elif isinstance(value, (Curve, Heatmap)):
        reduction = Reduction.Last
    else:
        reduction = Reduction.Merge

    if reduction == Reduction.Merge and spec.history is None:
        raise ValueError("Merge observation reduction requires EMAHistory.")
    if reduction != Reduction.Merge and spec.history is not None:
        raise ValueError("only Merge observation reduction supports EMAHistory.")

    if reduction == Reduction.Last and isinstance(value, (Curve, Heatmap)):
        return reduction
    if reduction == Reduction.Mean and isinstance(value, Curve):
        return reduction
    if reduction == Reduction.Merge and isinstance(
        value,
        (Distribution, Profile, Bins),
    ):
        return reduction
    raise ValueError(
        f"observation reduction {reduction.value!r} does not support "
        f"{type(value).__name__}."
    )


def _validate_histograms(
    states: Mapping[str, _StateEntry],
    *,
    device: torch.device,
) -> None:
    entries: list[tuple[str, _HistogramState]] = []
    for entry in states.values():
        state = entry.state
        if isinstance(state, _HistogramState):
            entries.append((entry.tag, state))
    if not entries:
        return
    total = torch.stack(
        [state.last_invalid.to(device) for _tag, state in entries]
    ).sum()
    if not bool((total > 0).item()):
        return
    for tag, state in entries:
        invalid = float(state.last_invalid.detach().cpu().item())
        if invalid > 0:
            raise ValueError(
                f"observation {tag!r} contained {invalid:g} non-finite, "
                "out-of-range, or invalid active values."
            )
    raise RuntimeError("observation invalid-count aggregation was inconsistent.")


def _validate_profiles(
    states: Mapping[str, _StateEntry],
    *,
    device: torch.device,
) -> None:
    entries: list[tuple[str, _ProfileState]] = []
    for entry in states.values():
        state = entry.state
        if isinstance(state, _ProfileState):
            entries.append((entry.tag, state))
    if not entries:
        return
    total = torch.stack(
        [state.last_invalid.to(device) for _tag, state in entries]
    ).sum()
    if not bool((total > 0).item()):
        return
    for tag, state in entries:
        invalid = float(state.last_invalid.detach().cpu().item())
        if invalid > 0:
            raise ValueError(
                f"observation {tag!r} contained {invalid:g} non-finite or "
                "out-of-range profile pairs."
            )
    raise RuntimeError("observation profile invalid-count aggregation was inconsistent.")


def _all_gather(
    trainer: pl.Trainer,
    value: Tensor,
    *,
    error: str,
) -> Tensor:
    result = _capability.strategy_all_gather(trainer, value, error=error)
    if not isinstance(result, Tensor):
        raise TypeError("trainer.strategy.all_gather must return a Tensor.")
    return result


def _raise_synchronized_error(
    error: _ObservationError | None,
    *,
    context: str,
) -> NoReturn:
    if error is not None:
        raise RuntimeError(f"distributed {context} failed: {error[1]}")
    raise RuntimeError(f"distributed {context} failed on another rank.")


def _remove_observer_hooks(
    module: nn.Module,
    callback: ReferenceType[ObservationCallback],
    collector_id: int,
) -> None:
    for hook_id, hook in tuple(module._forward_hooks.items()):
        if not isinstance(hook, _ForwardCollectorHook):
            continue
        if hook.callback != callback or hook.collector_id != collector_id:
            continue
        module._forward_hooks.pop(hook_id, None)
        module._forward_hooks_with_kwargs.pop(hook_id, None)
        module._forward_hooks_always_called.pop(hook_id, None)


def _is_descendant(path: str, parent: str) -> bool:
    if path == parent:
        return False
    if not parent:
        return True
    return path.startswith(f"{parent}.")


def _collector_owner(
    collector: _Collector,
    item: str,
    *,
    stage: Literal["train", "validation"],
    context: str,
) -> _StateOwner:
    return _StateOwner(
        stage=stage,
        context=context,
        module_path=collector.module_path,
        spec_name=collector.spec.name.strip("/"),
        item=item,
    )


__all__ = ["ObservationCallback"]
