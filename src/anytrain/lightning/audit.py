"""Lightning lifecycle adapter for the shared audit runtime."""
from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from anytrain._audit import (
    AuditSink,
    BatchReporter,
    JsonlAuditSink,
    _release_audit_sink,
    _retain_audit_sink,
    _use_audit_context,
    _use_audit_sink,
    audit_close,
    audit_collective,
    audit_directory,
    audit_enabled,
    audit_event,
    audit_flush,
    audit_flush_interval,
    audit_keep_steps,
    audit_operation,
    set_audit_step,
)
from anytrain._audit import (
    _sink_lifecycle as _sink_lifecycle,
)

from . import _capability


class TrainingAuditCallback(Callback):
    """Rank/step-aware Lightning lifecycle audit callback."""

    def __init__(
        self,
        *,
        sink: AuditSink | None = None,
        directory: str | Path | None = None,
        batch_report: BatchReporter | None = None,
    ) -> None:
        super().__init__()
        if sink is not None and directory is not None:
            raise ValueError("pass either sink or directory, not both.")
        self._owns_sink = directory is not None
        if directory is not None:
            sink = JsonlAuditSink(directory)
        if batch_report is not None and not callable(batch_report):
            raise TypeError("batch_report must be callable when provided.")
        self.sink, self.batch_report = sink, batch_report
        self._sink_released = False
        _retain_audit_sink(self.sink, owner=self._owns_sink)
        self._phase_started: dict[str, list[tuple[int, str]]] = {}

    def __getstate__(self) -> dict[str, object]:
        """Keep only spawn-safe sink configuration in serialized callbacks."""

        state = self.__dict__.copy()
        # Arbitrary sinks commonly contain process-local locks, threads, or
        # file descriptors. They must be rebound by the child process just as
        # they are for ModelCheckpoint; JsonlAuditSink has an explicit pickle
        # contract and can safely retain its directory/configuration.
        if not isinstance(state.get("sink"), JsonlAuditSink):
            state["sink"] = None
        state["_sink_released"] = False
        return state

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Restore the sink lease after Lightning spawn/pickle reconstruction."""

        state = dict(state)
        state.setdefault("sink", None)
        state.setdefault("_owns_sink", False)
        state.setdefault("_phase_started", {})
        sink_released = bool(state.get("_sink_released", False))
        self.__dict__.update(state)
        self._sink_released = sink_released
        if not sink_released:
            _retain_audit_sink(
                self.sink,
                owner=bool(getattr(self, "_owns_sink", False)),
            )

    def __del__(self) -> None:
        """Release a lease if a callback is discarded before ``on_fit_end``."""

        try:
            if not getattr(self, "_sink_released", True):
                _release_audit_sink(
                    getattr(self, "sink", None),
                    owner=bool(getattr(self, "_owns_sink", False)),
                )
                self._sink_released = True
        except BaseException:
            # Destructors run during interpreter shutdown too; cleanup must
            # never turn object finalization into an application exception.
            pass

    def _enabled(self) -> bool:
        """Return whether this callback has an audit destination configured."""

        return self.sink is not None or audit_enabled()

    def _emit(self, event: str, trainer: Trainer, /, **fields: object) -> None:
        strategy = getattr(trainer, "strategy", None)
        root_device = getattr(strategy, "root_device", None)
        payload = dict(fields)
        # ``event`` and ``sink`` belong to the audit call itself. Reporters may
        # return broad metadata mappings, so remove these names before keyword
        # expansion rather than letting diagnostics break the training hook.
        payload.pop("event", None)
        payload.pop("sink", None)
        payload.update(
            rank=_capability.global_rank(trainer),
            world_size=_capability.world_size(trainer),
            global_step=_capability.global_step(trainer),
            device=None if root_device is None else str(root_device),
        )
        trusted_context = {
            name: payload[name]
            for name in ("rank", "world_size", "device")
            if payload.get(name) is not None
        }
        with _use_audit_sink(self.sink), _use_audit_context(trusted_context):
            audit_event(event, sink=self.sink, **payload)

    def _start_phase(
        self,
        key: str,
        event: str,
        trainer: Trainer,
        /,
        **fields: object,
    ) -> None:
        operation_id = uuid4().hex
        self._phase_started.setdefault(key, []).append((time.perf_counter_ns(), operation_id))
        payload = dict(fields)
        payload.update(
            operation_id=operation_id,
            execution_scope="foreground",
            blocks_train_loop=True,
        )
        self._emit(
            event,
            trainer,
            **payload,
        )

    def _finish_phase(
        self,
        key: str,
        event: str,
        trainer: Trainer,
        finish_all: bool = False,
        /,
        **fields: object,
    ) -> None:
        pending = self._phase_started.get(key)
        if not pending:
            return
        timings = list(pending) if finish_all else [pending.pop()]
        if finish_all or not pending:
            self._phase_started.pop(key, None)
        for started, operation_id in timings:
            terminal_fields = dict(fields)
            terminal_fields.update(
                operation_id=operation_id,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                execution_scope="foreground",
                blocks_train_loop=True,
            )
            self._emit(
                event,
                trainer,
                **terminal_fields,
            )

    def _trainer_context(self, trainer: Trainer) -> dict[str, object]:
        strategy = getattr(trainer, "strategy", None)
        root_device = getattr(strategy, "root_device", None)
        return {
            "rank": _capability.global_rank(trainer),
            "world_size": _capability.world_size(trainer),
            "device": None if root_device is None else str(root_device),
        }

    def _batch_fields(self, batch: object, trainer: Trainer) -> dict[str, object]:
        with _use_audit_sink(self.sink), _use_audit_context(self._trainer_context(trainer)):
            if self.batch_report is None:
                return {"batch_type": type(batch).__name__}
            return dict(self.batch_report(batch))

    def _batch_fields_or_raise(
        self,
        batch: object,
        trainer: Trainer,
        *,
        active_phase: str,
        terminal_event: str,
        **fields: object,
    ) -> dict[str, object]:
        """Collect batch metadata while closing an active span on reporter errors."""
        try:
            return self._batch_fields(batch, trainer)
        except BaseException as error:
            self._finish_phase(
                active_phase,
                terminal_event,
                trainer,
                True,
                status="batch_report_error",
                stage="batch_report",
                error_type=type(error).__name__,
                error=str(error)[:1000],
                **fields,
            )
            audit_flush(sink=self.sink)
            raise

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not self._enabled():
            return
        set_audit_step(_capability.global_step(trainer))
        self._emit("fit_start", trainer)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int
    ) -> None:
        del pl_module
        if not self._enabled():
            return
        set_audit_step(_capability.global_step(trainer))
        try:
            batch_fields = self._batch_fields(batch, trainer)
        except BaseException as error:
            self._emit(
                "batch_report_error",
                trainer,
                phase="train_batch_start",
                batch_idx=batch_idx,
                stage="batch_report",
                error_type=type(error).__name__,
                error=str(error)[:1000],
            )
            audit_flush(sink=self.sink)
            raise
        batch_fields["batch_idx"] = batch_idx
        self._start_phase("train_batch", "train_batch_start", trainer, **batch_fields)
        self._start_phase(
            "training_step",
            "training_step_start",
            trainer,
            batch_idx=batch_idx,
            latency_scope="train_batch_start_to_first_backward_or_batch_end",
        )

    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: object
    ) -> None:
        del pl_module
        if not self._enabled():
            return
        self._finish_phase(
            "training_step",
            "training_step_done",
            trainer,
            False,
            status="backward_started",
            latency_scope="train_batch_start_to_before_backward",
        )
        self._start_phase("backward", "backward_start", trainer, loss_type=type(loss).__name__)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not self._enabled():
            return
        self._finish_phase("backward", "backward_done", trainer)

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Any
    ) -> None:
        del pl_module, optimizer
        if not self._enabled():
            return
        self._start_phase("optimizer_step", "optimizer_step_start", trainer)

    def on_before_zero_grad(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Any
    ) -> None:
        del pl_module, optimizer
        if not self._enabled():
            return
        self._emit(
            "zero_grad",
            trainer,
            execution_scope="foreground",
            blocks_train_loop=True,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs
        if not self._enabled():
            return
        set_audit_step(_capability.global_step(trainer))
        self._finish_phase(
            "training_step",
            "training_step_done",
            trainer,
            True,
            status="batch_ended_without_backward_hook",
            latency_scope="train_batch_start_to_train_batch_end",
        )
        self._finish_phase(
            "optimizer_step",
            "optimizer_step_done",
            trainer,
            True,
            latency_scope="before_optimizer_step_to_train_batch_end",
        )
        batch_fields = self._batch_fields_or_raise(
            batch,
            trainer,
            active_phase="train_batch",
            terminal_event="train_batch_end",
            batch_idx=batch_idx,
        )
        # Merge callback-owned identity fields before expanding the reporter
        # payload.  A reporter is allowed to return arbitrary task metadata
        # (including a stale ``batch_idx``); wrapper fields must win without
        # triggering Python's duplicate-key error and leaving the span open.
        terminal_fields = dict(batch_fields)
        terminal_fields["batch_idx"] = batch_idx
        self._finish_phase(
            "train_batch",
            "train_batch_end",
            trainer,
            **terminal_fields,
        )

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module
        if not self._enabled():
            return
        try:
            batch_fields = self._batch_fields(batch, trainer)
        except BaseException as error:
            self._emit(
                "batch_report_error",
                trainer,
                phase="validation_batch_start",
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
                stage="batch_report",
                error_type=type(error).__name__,
                error=str(error)[:1000],
            )
            audit_flush(sink=self.sink)
            raise
        batch_fields.update(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self._start_phase("validation_batch", "validation_batch_start", trainer, **batch_fields)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, outputs
        if not self._enabled():
            return
        batch_fields = self._batch_fields_or_raise(
            batch,
            trainer,
            active_phase="validation_batch",
            terminal_event="validation_batch_end",
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        terminal_fields = dict(batch_fields)
        terminal_fields.update(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self._finish_phase(
            "validation_batch",
            "validation_batch_end",
            trainer,
            **terminal_fields,
        )

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        del pl_module
        if not self._enabled():
            return
        active_phases = tuple(sorted(self._phase_started))
        self._emit(
            "trainer_exception",
            trainer,
            error_type=type(exception).__name__,
            error=str(exception)[:1000],
            active_phases=active_phases,
        )
        # Lightning may stop dispatching normal phase-end hooks after an
        # exception. Close every span that was active at the failure site so
        # the audit report can associate the exception with its operation id
        # instead of leaving an indistinguishable open span behind.
        for phase in active_phases:
            self._finish_phase(
                phase,
                f"{phase}_error",
                trainer,
                True,
                phase=phase,
                status="trainer_exception",
                stage="trainer_exception",
                error_type=type(exception).__name__,
                error=str(exception)[:1000],
            )
        self._phase_started.clear()
        audit_flush(sink=self.sink)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not self._enabled():
            return
        # A fit can terminate without dispatching every batch-end hook (for
        # example after an external stop or callback-order interruption).
        # Close the spans that are still open so the bounded audit remains
        # pairable and records why their terminal timestamp was forced.
        active_phases = tuple(sorted(self._phase_started))
        for phase in active_phases:
            self._finish_phase(
                phase,
                f"{phase}_end",
                trainer,
                True,
                phase=phase,
                status="fit_end",
                stage="fit_end",
            )
        self._emit("fit_end", trainer, active_phases=active_phases)
        self._phase_started.clear()
        audit_flush(sink=self.sink)
        if not self._sink_released:
            _release_audit_sink(self.sink, owner=self._owns_sink)
            self._sink_released = True


__all__ = ['AuditSink', 'JsonlAuditSink', 'TrainingAuditCallback', 'audit_close', 'audit_collective', 'audit_directory', 'audit_enabled', 'audit_event', 'audit_flush', 'audit_flush_interval', 'audit_keep_steps', 'audit_operation', 'set_audit_step']
