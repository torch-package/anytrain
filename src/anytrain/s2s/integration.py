"""Lightning integration for S2S probes and metric windows."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from anytrain.lightning import StepInterval, _capability, experiment
from anytrain.lightning.oom import annotate

from .condition import (
    ConditionGainProbe,
    _non_negative_int,
    _probes,
    _tag_segment,
    condition_gain,
    condition_gain_metrics,
)
from .data import Route
from .metrics import _Window, _window_metrics
from .model import TokenScorer
from .training import REPORT_KEY, StepReport


class ConditionGainCallback(Callback):
    """Periodically score fixed interventions through the strategy-wrapped model."""

    def __init__(
        self,
        probes: Sequence[ConditionGainProbe],
        *,
        probe_set_id: str,
        every_n_steps: int,
        bootstrap_samples: int = 1_000,
        bootstrap_seed: int = 0,
    ) -> None:
        super().__init__()
        self.probes = _probes(probes)
        _tag_segment(probe_set_id, "condition gain probe_set_id")
        _non_negative_int(bootstrap_samples, "condition gain bootstrap_samples")
        _non_negative_int(bootstrap_seed, "condition gain bootstrap_seed")
        self.probe_set_id = probe_set_id
        self.interval = StepInterval(every_n_steps=every_n_steps)
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            probe_set_id=self.probe_set_id,
            every_n_steps=self.interval.every_n_steps,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_if_due(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        self._run_if_due(trainer, pl_module)

    def state_dict(self) -> dict[str, dict[str, int | None]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if set(state_dict) != {"interval"}:
            raise ValueError("condition gain callback state must contain only interval.")
        interval = state_dict["interval"]
        if not isinstance(interval, Mapping):
            raise TypeError("condition gain interval state must be a mapping.")
        self.interval.load_state_dict(interval)

    def _run_if_due(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = _capability.global_step(trainer)
        if not self.interval.should_run(step):
            return
        device = getattr(pl_module, "device", None)
        if not isinstance(device, torch.device):
            raise TypeError("condition gain callback requires pl_module.device.")
        model = trainer.strategy.model
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("condition gain callback requires trainer.strategy.model.")
        provider = cast(TokenScorer, model)
        module_modes = tuple((module, module.training) for module in model.modules())
        rng_devices = [] if device.type == "cpu" else [device]
        rng_device_type = "cuda" if device.type == "cpu" else device.type
        try:
            with torch.random.fork_rng(
                devices=rng_devices,
                device_type=rng_device_type,
            ):
                model.eval()
                probes = tuple(probe.to(device) for probe in self.probes)
                with trainer.precision_plugin.forward_context():
                    results = condition_gain(
                        provider,
                        probes,
                        bootstrap_samples=self.bootstrap_samples,
                        bootstrap_seed=self.bootstrap_seed,
                    )
        except torch.OutOfMemoryError as error:
            annotate(
                error,
                phase="condition_gain",
                inputs={
                    "probe_set_id": self.probe_set_id,
                    "probe_count": len(self.probes),
                },
            )
            raise
        finally:
            for module, training in module_modes:
                module.training = training
        if not _capability.is_global_zero(trainer):
            return
        writer = experiment.scalar(trainer)
        if writer is None:
            raise RuntimeError(
                "ConditionGainCallback requires a logger with add_scalar capability."
            )
        for result in results:
            for tag, value in condition_gain_metrics(
                result,
                probe_set_id=self.probe_set_id,
            ).items():
                writer.add_scalar(tag, value, step)


class S2SMetricsCallback(Callback):
    """Accumulate exact S2S sums/counts and log one reduced metric window."""

    def __init__(
        self,
        routes: Sequence[Route],
        *,
        every_n_steps: int,
        sync_distributed: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(routes, (str, bytes)) or not isinstance(routes, Sequence):
            raise TypeError("S2S metric routes must be a sequence of Route values.")
        values = tuple(routes)
        if not values:
            raise ValueError("S2S metric routes must not be empty.")
        if any(not isinstance(route, Route) for route in values):
            raise TypeError("S2S metric routes must contain only Route values.")
        if len(values) != len(set(values)):
            raise ValueError("S2S metric routes must be unique.")
        if not isinstance(sync_distributed, bool):
            raise TypeError("sync_distributed must be a boolean.")
        self.routes = values
        self.interval = StepInterval(every_n_steps=every_n_steps)
        self.sync_distributed = sync_distributed
        self._window = _Window(values)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        if not isinstance(outputs, Mapping):
            return
        report = outputs.get(REPORT_KEY)
        if report is None:
            return
        if not isinstance(report, StepReport):
            raise TypeError(f"training output {REPORT_KEY!r} must be StepReport.")
        self._window.update(report)
        step = _capability.global_step(trainer)
        if step > 0 and self.interval.should_run(step):
            self._flush(trainer, step=step)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self._window:
            self._flush(trainer, step=_capability.global_step(trainer))

    def _flush(self, trainer: Trainer, *, step: int) -> None:
        packed = self._window.pack()
        if self.sync_distributed and _capability.world_size(trainer) > 1:
            packed = _capability.strategy_reduce(
                trainer,
                packed,
                reduce_op="sum",
                error="Distributed S2S metrics require trainer.strategy.reduce.",
            )
        if _capability.is_global_zero(trainer):
            writer = experiment.scalar(trainer)
            if writer is None:
                raise RuntimeError(
                    "S2SMetricsCallback requires a logger with add_scalar capability."
                )
            for tag, value in _window_metrics(packed, self.routes).items():
                writer.add_scalar(tag, value, step)
        self._window.reset()
