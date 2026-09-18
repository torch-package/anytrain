from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from weakref import proxy

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint
from lightning_utilities.core.rank_zero import rank_zero_info

from .. import _capability
from ..audit import (
    AuditSink,
    JsonlAuditSink,
    _release_audit_sink,
    _retain_audit_sink,
)
from ..schedule import (
    UnitScheduleCallback,
    UnitScheduleUpdate,
    register_unit_schedule_consumer,
)
from ._checkpoint_storage import AsyncCheckpointStorage, CheckpointOperation

_checkpoint_trigger_context: ContextVar[str] = ContextVar(
    "anytrain_checkpoint_trigger_context", default="lightning",
)


class ModelCheckpoint(LightningModelCheckpoint):
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | Literal["link"] | None = None,
        save_top_k: int = 1,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
        *,
        async_save: bool = True,
        max_pending_async_saves: int = 1,
        unit_schedule: UnitScheduleCallback | None = None,
        every_n_unit_epochs: int | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        unit_epoch_trigger = every_n_unit_epochs is not None
        if unit_epoch_trigger:
            if not isinstance(unit_schedule, UnitScheduleCallback):
                raise TypeError(
                    "every_n_unit_epochs requires a UnitScheduleCallback."
                )
            if (
                isinstance(every_n_unit_epochs, bool)
                or not isinstance(every_n_unit_epochs, int)
            ):
                raise TypeError("every_n_unit_epochs must be an integer or None.")
            if every_n_unit_epochs <= 0:
                raise ValueError("every_n_unit_epochs must be positive.")
            if unit_schedule.clock.units_per_epoch is None:
                raise ValueError(
                    "unit epoch checkpointing requires clock.units_per_epoch."
                )
            if every_n_train_steps not in (None, 0):
                raise ValueError(
                    "every_n_unit_epochs and every_n_train_steps are mutually exclusive."
                )
            if train_time_interval is not None:
                raise ValueError(
                    "every_n_unit_epochs and train_time_interval are mutually exclusive."
                )
            if every_n_epochs not in (None, 0):
                raise ValueError(
                    "every_n_unit_epochs and every_n_epochs are mutually exclusive."
                )
            every_n_train_steps = 0
            every_n_epochs = 0
        elif unit_schedule is not None:
            raise ValueError(
                "unit_schedule requires every_n_unit_epochs."
            )
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_on_exception=save_on_exception,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        if not isinstance(async_save, bool):
            raise TypeError("async_save must be a boolean.")
        if isinstance(max_pending_async_saves, bool) or not isinstance(
            max_pending_async_saves, int
        ):
            raise TypeError("max_pending_async_saves must be an integer.")
        if max_pending_async_saves <= 0:
            raise ValueError("max_pending_async_saves must be a positive integer.")
        self.async_save = async_save
        self.max_pending_async_saves = max_pending_async_saves
        self.unit_schedule = unit_schedule
        self.every_n_unit_epochs = every_n_unit_epochs
        # The sink is owned by the caller (typically a shared
        # TrainingAuditCallback or an anydataset loader).  Keep it as a
        # runtime dependency so checkpoint worker threads can emit into the
        # same stream without relying on a ContextVar copied at submission.
        self.audit_sink = audit_sink
        self._audit_sink_released = False
        _retain_audit_sink(self.audit_sink)
        self._storage = self._new_storage()
        self._deferred_async_removes: list[str] = []
        self._prepared_last_paths: tuple[str, str] | None = None
        self._reused_last_save: tuple[str, str, Path, Path, Future[None] | None] | None = None
        self._restored_last_checkpoint: Path | None = None

    def _new_storage(self) -> AsyncCheckpointStorage:
        return AsyncCheckpointStorage(
            capacity=self.max_pending_async_saves, audit_sink=self.audit_sink,
            on_publication=self._record_storage_publication,
        )

    def _record_storage_publication(
        self, expected: str, saved: str, persistent: Path | None, success: bool,
    ) -> None:
        if success:
            self._last_checkpoint_saved = saved
        elif self._last_checkpoint_saved == expected:
            self._last_checkpoint_saved = "" if persistent is None else str(persistent)

    def wait_async_saves(self) -> None:
        if self.async_save:
            self._storage.wait_async_saves()

    def __del__(self) -> None:
        """Release an audit sink lease when a callback is discarded early."""

        try:
            if not getattr(self, "_audit_sink_released", True):
                _release_audit_sink(getattr(self, "audit_sink", None))
                self._audit_sink_released = True
        except BaseException:
            # Destructors also run during interpreter shutdown; cleanup must
            # never turn object finalization into an application exception.
            pass

    @property
    def state_key(self) -> str:
        if self.unit_schedule is None:
            return super().state_key
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            unit=self.unit_schedule.clock.unit,
            every_n_unit_epochs=self.every_n_unit_epochs,
        )

    def __getstate__(self) -> dict[str, Any]:
        if self._deferred_async_removes:
            raise RuntimeError("Cannot serialize ModelCheckpoint while async saves are pending.")

        state = self.__dict__.copy()
        state.pop("_storage")
        state["_last_persistent_checkpoint"] = self._storage.persistent_state()
        state["_deferred_async_removes"] = []
        state["_prepared_last_paths"] = None
        state["_reused_last_save"] = None
        # Generic sinks may contain locks/threads and are process-local runtime
        # resources. JsonlAuditSink has an explicit pickle contract, so retain
        # its directory/configuration for spawn-based DDP workers.
        if not isinstance(state.get("audit_sink"), JsonlAuditSink):
            state["audit_sink"] = None
        state["_audit_sink_released"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("max_pending_async_saves", 1)
        state.setdefault("unit_schedule", None)
        state.setdefault("every_n_unit_epochs", None)
        state.setdefault("_last_persistent_checkpoint", None)
        state.setdefault("_restored_last_checkpoint", None)
        state.setdefault("audit_sink", None)
        state.setdefault("_audit_sink_released", False)
        self.__dict__.update(state)
        self._audit_sink_released = False
        _retain_audit_sink(self.audit_sink)
        self._storage = self._new_storage()
        self._storage.restore_persistent_state(state.get("_last_persistent_checkpoint"))
        # Old callback pickles carried empty process-local storage fields.
        for key in tuple(self.__dict__):
            if key.startswith("_async_") or key in {
                "_last_persistent_checkpoint", "_latest_publication_future", "_latest_publication_path"
            }:
                self.__dict__.pop(key)
        self._deferred_async_removes = []
        self._prepared_last_paths = None
        self._reused_last_save = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._restored_last_checkpoint = None
        restored = state_dict.get("last_model_path")
        same_dirpath = self.dirpath == state_dict.get("dirpath", self.dirpath)
        if self.async_save and self.save_last == "link" and same_dirpath and restored:
            self._restored_last_checkpoint = self._target_path(restored)

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        super().on_train_start(trainer, pl_module)
        schedule = self.unit_schedule
        if schedule is None:
            return
        if not pl_module.automatic_optimization:
            raise RuntimeError(
                "unit epoch checkpointing requires automatic optimization."
            )
        if trainer.world_size > 1 and not schedule.clock.sync_distributed:
            raise RuntimeError(
                "distributed unit epoch checkpointing requires a globally "
                "synchronized unit clock."
            )
        register_unit_schedule_consumer(trainer, schedule, self)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._restored_last_checkpoint = None
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        schedule = self.unit_schedule
        every_n_epochs = self.every_n_unit_epochs
        if schedule is None or every_n_epochs is None:
            with self._checkpoint_trigger("train_batch_end"):
                super().on_train_batch_end(
                    trainer,
                    pl_module,
                    outputs,
                    batch,
                    batch_idx,
                )
            return
        # Unit-triggered saves are dispatched by UnitScheduleCallback after it
        # has produced the corresponding unit-clock update.
        del trainer, pl_module, outputs, batch, batch_idx

    def on_unit_schedule_update(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        update: UnitScheduleUpdate,
    ) -> None:
        every_n_epochs = self.every_n_unit_epochs
        if self.unit_schedule is None or every_n_epochs is None:
            return
        if update.advanced_unit_epochs <= 0:
            return
        if not update.crossed_unit_epoch_interval(every_n_epochs):
            return
        if self._should_skip_saving_checkpoint(trainer):
            return

        monitor_candidates = self._monitor_candidates(trainer)
        if self.monitor is not None and (
            self.monitor not in monitor_candidates or update.validation_due
        ):
            self._defer_save_until_validation = True
            return
        with self._checkpoint_trigger("unit_schedule"):
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        with self._checkpoint_trigger("validation_end"):
            super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        with self._checkpoint_trigger("train_epoch_end"):
            super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        restored = self._restored_last_checkpoint
        if restored is not None:
            checkpoint_path = getattr(trainer, "ckpt_path", None)
            reuse_last = (
                _capability.is_global_zero(trainer)
                and checkpoint_path is not None
                and self._same_checkpoint(restored, checkpoint_path)
            )
            self._restored_last_checkpoint = None
            if self._synchronize_rank_zero_flag(trainer, reuse_last):
                if _capability.is_global_zero(trainer):
                    rank_zero_info(
                        "Skipping last checkpoint save: fit resumed from unchanged "
                        f"checkpoint {restored!s} and ran no training batches."
                    )
                return
        with self._checkpoint_trigger("train_end"):
            super().on_train_end(trainer, pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restored_last_checkpoint = None
        hook_error: BaseException | None = None
        try:
            super().on_fit_end(trainer, pl_module)
        except BaseException as error:
            hook_error = error
        self._drain_and_close_async_storage(trainer, hook_error)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._restored_last_checkpoint = None
        prepared = self._prepare_exception_last_save(trainer)
        hook_error: BaseException | None = None
        try:
            with self._checkpoint_trigger("exception"):
                super().on_exception(trainer, pl_module, exception)
        except BaseException as error:
            hook_error = error
        finally:
            self._restore_unused_last_save(prepared)
        self._drain_and_close_async_storage(trainer, hook_error)

    @contextmanager
    def _checkpoint_trigger(self, trigger: str) -> Iterator[None]:
        token = _checkpoint_trigger_context.set(trigger)
        try:
            yield
        finally:
            _checkpoint_trigger_context.reset(token)

    def _begin_checkpoint_audit(
        self,
        trainer: pl.Trainer,
        filepath: str,
        *,
        checkpoint_kind: str | None = None,
        reason: str | None = None,
    ) -> CheckpointOperation:
        path = os.fspath(filepath)
        trigger = _checkpoint_trigger_context.get()
        kind = self._checkpoint_kind(path) if checkpoint_kind is None else checkpoint_kind
        fields: dict[str, object] = {
            "path": path,
            "target": path,
            "trigger": trigger,
            "reason": self._checkpoint_reason(trigger) if reason is None else reason,
            "checkpoint_kind": kind,
            "async_save": self.async_save,
            "rank": _capability.global_rank(trainer),
            "world_size": _capability.world_size(trainer),
            "global_step": _capability.global_step(trainer),
        }
        root_device = getattr(getattr(trainer, "strategy", None), "root_device", None)
        if root_device is not None:
            fields["device"] = str(root_device)
        return self._storage.begin_checkpoint(fields)

    def _checkpoint_reason(self, trigger: str) -> str:
        if trigger == "unit_schedule":
            return "unit_epoch_interval"
        if trigger == "exception":
            return "exception"
        if trigger == "train_batch_end":
            if self._every_n_train_steps:
                return "train_step_interval"
            if self._train_time_interval is not None:
                return "train_time_interval"
            return "train_batch_end"
        if trigger == "train_epoch_end":
            return "train_epoch_interval"
        if trigger == "validation_end":
            return "validation_metric" if self.monitor is not None else "validation_end"
        if trigger == "train_end":
            return "train_end"
        return "lightning_callback"

    def _checkpoint_kind(self, filepath: str) -> str:
        if filepath == self.last_model_path or Path(filepath).stem.startswith(
            self.CHECKPOINT_NAME_LAST
        ):
            return "last"
        return "top_k" if self.monitor is not None else "periodic"

    def _save_topk_checkpoint(
        self,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> None:
        if not self.async_save or not self.save_last or self.save_top_k == 0:
            super()._save_topk_checkpoint(trainer, monitor_candidates)
            return

        prepared = self._prepare_last_save(trainer, monitor_candidates)
        try:
            super()._save_topk_checkpoint(trainer, monitor_candidates)
        finally:
            self._restore_unused_last_save(prepared)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        audit = self._begin_checkpoint_audit(trainer, filepath)
        if not self.async_save:
            try:
                self._storage.serialize(
                    audit, filepath,
                    lambda: trainer.save_checkpoint(filepath, self.save_weights_only),
                )
                self._last_global_step_saved = _capability.global_step(trainer)
                self._last_checkpoint_saved = filepath
                if _capability.is_global_zero(trainer):
                    with self._storage.callback_phase(audit):
                        for callback in self._checkpoint_callbacks(trainer, filepath):
                            callback()
            except BaseException as error:
                self._storage.finish_checkpoint(audit, success=False, error=error)
                raise
            self._storage.finish_checkpoint(audit, success=True)
            return

        prepared = self._prepared_last_paths
        self._prepared_last_paths = None
        previous_checkpoint_saved = self._last_checkpoint_saved
        future: Future[None] | None = None
        try:
            self._validate_async_strategy(trainer)
            target = self._target_path(filepath)
            self._storage.retarget(audit, target)
            is_global_zero = _capability.is_global_zero(trainer)
            local = self._prepare_async_save(trainer, target, audit=audit)

            # DDP/strategy collectives may require every rank to enter save_checkpoint,
            # but only global zero typically materializes a regular file.
            self._storage.serialize(
                audit, local,
                lambda: trainer.save_checkpoint(str(local), self.save_weights_only),
            )
            self._last_global_step_saved = _capability.global_step(trainer)
            local_error: BaseException | None = None
            if is_global_zero:
                try:
                    if not local.is_file():
                        raise RuntimeError(
                            "async_save requires trainer.save_checkpoint() to produce a "
                            "regular file; use async_save=False for directory-based or "
                            "sharded checkpoints."
                        )
                    # Expose the local temp path until all publications finish so
                    # immediate readers see an existing file.
                    self._last_checkpoint_saved = str(local)

                    callbacks: tuple[Callable[[], None], ...] = ()
                    publication: tuple[str, str, Path] | None = None
                    if prepared is None:
                        callbacks = self._checkpoint_callbacks(trainer, filepath)
                        publication = (str(local), filepath, target)
                    future = self._storage.publish_copy(
                        local,
                        target,
                        callbacks=callbacks,
                        publication=publication,
                        defer_audit_completion=prepared is not None,
                        audit=audit,
                    )
                except BaseException as error:
                    local_error = error
            else:
                self._last_checkpoint_saved = filepath

            synchronized = self._synchronize_async_error(trainer, local_error)
            if synchronized is not None:
                if is_global_zero and future is None:
                    self._storage.discard_local(local)
                self._last_checkpoint_saved = previous_checkpoint_saved
                raise synchronized
            self._storage.finish_submit(
                audit,
                success=True,
                publication_pending=future is not None,
            )
            if future is None:
                self._storage.finish_checkpoint(audit, success=True)
            if prepared is not None:
                previous, last = prepared
                self._reused_last_save = (previous, last, local, target, future)
        except BaseException as error:
            self._storage.finish_submit(audit, success=False, error=error)
            self._storage.finish_checkpoint(audit, success=False, error=error)
            if prepared is not None:
                previous, _ = prepared
                self.last_model_path = previous
                self._reused_last_save = None
            raise

    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        if not self.async_save:
            super()._remove_checkpoint(trainer, filepath)
            return

        defer_remove = self.save_last == "link" and self._reused_last_save is not None
        local_error: BaseException | None = None
        audit: CheckpointOperation | None = None
        if _capability.is_global_zero(trainer):
            local_error = self._storage.collect_error()
            if local_error is None and not defer_remove:
                try:
                    target = self._target_path(filepath)
                    audit = self._begin_checkpoint_audit(
                        trainer,
                        str(target),
                        checkpoint_kind="remove",
                        reason="retention_cleanup",
                    )
                    self._storage.remove_after_pending(target, audit=audit)
                except BaseException as error:
                    local_error = error
        error = self._synchronize_async_error(trainer, local_error)
        if error is not None:
            if audit is not None:
                self._storage.finish_submit(audit, success=False, error=error)
                self._storage.finish_checkpoint(audit, success=False, error=error)
            raise error
        if audit is not None:
            self._storage.finish_submit(
                audit,
                success=True,
                publication_pending=False,
            )
        if defer_remove:
            self._deferred_async_removes.append(filepath)

    def _save_last_checkpoint(
        self,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> None:
        reused = self._reused_last_save
        if not self.async_save:
            super()._save_last_checkpoint(trainer, monitor_candidates)
            return

        if reused is None:
            if self.save_last != "link" or self.save_top_k == 0:
                super()._save_last_checkpoint(trainer, monitor_candidates)
                return

            local_error: BaseException | None = None
            if _capability.is_global_zero(trainer):
                local_error = self._storage.collect_error()
            error = self._synchronize_async_error(trainer, local_error)
            if error is not None:
                raise error

            persistent: Path | None = None
            prerequisite: Future[None] | None = None
            if _capability.is_global_zero(trainer):
                persistent, prerequisite = self._storage.latest_publication()
            has_persistent = self._synchronize_rank_zero_flag(
                trainer,
                persistent is not None,
            )
            if not has_persistent:
                self._last_checkpoint_saved = ""
                super()._save_last_checkpoint(trainer, monitor_candidates)
                return

            previous, last = self._last_save_paths(trainer, monitor_candidates)
            self.last_model_path = last
            local_error = None
            audit: CheckpointOperation | None = None
            future: Future[None] | None = None
            if _capability.is_global_zero(trainer):
                assert persistent is not None
                try:
                    audit = self._begin_checkpoint_audit(
                        trainer,
                        last,
                        checkpoint_kind="last",
                        reason="last_publication",
                    )
                    future = self._storage.publish_last(
                        prerequisite,
                        persistent,
                        Path(last),
                        link=True,
                        publication=(
                            self._last_checkpoint_saved,
                            str(persistent),
                            persistent,
                        ),
                        audit=audit,
                    )
                except BaseException as error:
                    local_error = error
            error = self._synchronize_async_error(trainer, local_error)
            if error is not None:
                self.last_model_path = previous
                if audit is not None:
                    self._storage.finish_submit(
                        audit,
                        success=False,
                        error=error,
                    )
                    self._storage.finish_checkpoint(
                        audit,
                        success=False,
                        error=error,
                    )
                raise error
            if audit is not None:
                self._storage.finish_submit(
                    audit,
                    success=True,
                    publication_pending=future is not None,
                )
            if previous and self._should_remove_checkpoint(trainer, previous, last):
                self._remove_checkpoint(trainer, previous)
            return

        self._reused_last_save = None
        deferred_removes = self._deferred_async_removes
        self._deferred_async_removes = []
        previous, last, _local, persistent, prerequisite = reused
        local_error = None
        audit = None
        created_audit = False
        if _capability.is_global_zero(trainer):
            try:
                audit = self._storage.operation_for(prerequisite)
                if audit is None:
                    audit = self._begin_checkpoint_audit(
                        trainer,
                        last,
                        checkpoint_kind="last",
                        reason="last_publication",
                    )
                    created_audit = True
                final_path = str(persistent) if self.save_last == "link" else last
                callbacks = self._checkpoint_callbacks(trainer, final_path)
                self._storage.publish_last(
                    prerequisite,
                    persistent,
                    Path(last),
                    link=self.save_last == "link",
                    callbacks=callbacks,
                    publication=(
                        str(_local),
                        final_path,
                        persistent if self.save_last == "link" else Path(last),
                    ),
                    audit=audit,
                )
            except BaseException as error:
                local_error = error
        elif self.save_last is True:
            self._last_checkpoint_saved = last
        error = self._synchronize_async_error(trainer, local_error)
        if error is not None:
            self.last_model_path = previous
            if audit is not None:
                if created_audit:
                    self._storage.finish_submit(
                        audit,
                        success=False,
                        error=error,
                    )
                self._storage.finish_checkpoint(audit, success=False, error=error)
            raise error
        if audit is not None and created_audit:
            self._storage.finish_submit(
                audit,
                success=True,
                publication_pending=True,
            )
        for deferred in deferred_removes:
            self._remove_checkpoint(trainer, deferred)
        if previous and self._should_remove_checkpoint(trainer, previous, last):
            self._remove_checkpoint(trainer, previous)

    def _prepare_exception_last_save(
        self,
        trainer: pl.Trainer,
    ) -> tuple[str, str] | None:
        if (
            not self.async_save
            or not self.save_last
            or not self._should_save_on_exception(trainer)
        ):
            return None
        return self._prepare_last_save(trainer, self._monitor_candidates(trainer))

    def _prepare_last_save(
        self,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> tuple[str, str]:
        if self._prepared_last_paths is not None:
            raise RuntimeError("A checkpoint save is already preparing its last path.")

        prepared = self._last_save_paths(trainer, monitor_candidates)
        _, filepath = prepared
        self.last_model_path = filepath
        self._prepared_last_paths = prepared
        return prepared

    def _last_save_paths(
        self,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> tuple[str, str]:
        filepath = self.format_checkpoint_name(
            monitor_candidates,
            self.CHECKPOINT_NAME_LAST,
        )
        if self._enable_version_counter:
            version = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(
                    monitor_candidates,
                    self.CHECKPOINT_NAME_LAST,
                    ver=version,
                )
                version += 1
        return self.last_model_path, filepath

    def _restore_unused_last_save(self, prepared: tuple[str, str] | None) -> None:
        if prepared is None or self._prepared_last_paths is not prepared:
            return
        previous, _ = prepared
        self.last_model_path = previous
        self._prepared_last_paths = None

    def _checkpoint_callbacks(
        self,
        trainer: pl.Trainer,
        filepath: str,
    ) -> tuple[Callable[[], None], ...]:
        del filepath
        return tuple(
            lambda logger=logger: logger.after_save_checkpoint(proxy(self))
            for logger in _capability.loggers(trainer)
        )

    def _prepare_async_save(
        self,
        trainer: pl.Trainer,
        target: Path,
        *,
        audit: CheckpointOperation | None = None,
    ) -> Path:
        local = target
        local_error: BaseException | None = None
        if _capability.is_global_zero(trainer):
            try:
                local = self._storage.prepare_local(target, audit=audit)
            except BaseException as error:
                local_error = error
        error = self._synchronize_async_error(trainer, local_error)
        if error is not None:
            raise error
        return local

    def _wait_async_saves_distributed(self, trainer: pl.Trainer) -> None:
        local_error: BaseException | None = None
        if _capability.is_global_zero(trainer):
            try:
                self.wait_async_saves()
            except BaseException as error:
                local_error = error
        error = self._synchronize_async_error(trainer, local_error)
        if error is not None:
            raise error

    def _drain_and_close_async_storage(
        self,
        trainer: pl.Trainer,
        hook_error: BaseException | None,
    ) -> None:
        first_error = hook_error
        try:
            self._wait_async_saves_distributed(trainer)
        except BaseException as error:
            if first_error is None:
                first_error = error
        finally:
            self._storage.close()
            self._deferred_async_removes = []
            if not self._audit_sink_released:
                _release_audit_sink(self.audit_sink)
                self._audit_sink_released = True
        if first_error is not None:
            raise first_error

    def _synchronize_async_error(
        self,
        trainer: pl.Trainer,
        local_error: BaseException | None,
    ) -> BaseException | None:
        if _capability.world_size(trainer) <= 1:
            return local_error

        strategy = getattr(trainer, "strategy", None)
        broadcast = getattr(strategy, "broadcast", None)
        if callable(broadcast):
            message = (
                str(local_error)
                if _capability.is_global_zero(trainer) and local_error is not None
                else None
            )
            synchronized = _capability.strategy_broadcast(
                trainer,
                message,
                src=0,
                error=(
                    "Distributed asynchronous checkpoint error propagation requires "
                    "trainer.strategy.broadcast() or reduce()."
                ),
            )
            if synchronized is None:
                return None
            if local_error is not None:
                return local_error
            return RuntimeError(str(synchronized))

        device = getattr(strategy, "root_device", None)
        flag = torch.tensor(
            int(local_error is not None),
            dtype=torch.int64,
            device=device,
        )
        reduced = _capability.strategy_reduce(
            trainer,
            flag,
            reduce_op="sum",
            error=(
                "Distributed asynchronous checkpoint error propagation requires "
                "trainer.strategy.broadcast() or reduce()."
            ),
        )
        if int(reduced.detach().cpu().item()) == 0:
            return None
        if local_error is not None:
            return local_error
        return RuntimeError("Asynchronous checkpoint storage failed on global rank zero.")

    def _synchronize_rank_zero_flag(self, trainer: pl.Trainer, value: bool) -> bool:
        if _capability.world_size(trainer) <= 1:
            return value

        strategy = getattr(trainer, "strategy", None)
        broadcast = getattr(strategy, "broadcast", None)
        if callable(broadcast):
            synchronized = _capability.strategy_broadcast(
                trainer,
                value if _capability.is_global_zero(trainer) else None,
                src=0,
                error=(
                    "Distributed asynchronous checkpoint state propagation requires "
                    "trainer.strategy.broadcast() or reduce()."
                ),
            )
            return bool(synchronized)

        device = getattr(strategy, "root_device", None)
        flag = torch.tensor(
            int(value) if _capability.is_global_zero(trainer) else 0,
            dtype=torch.int64,
            device=device,
        )
        reduced = _capability.strategy_reduce(
            trainer,
            flag,
            reduce_op="sum",
            error=(
                "Distributed asynchronous checkpoint state propagation requires "
                "trainer.strategy.broadcast() or reduce()."
            ),
        )
        return bool(int(reduced.detach().cpu().item()))

    @staticmethod
    def _validate_async_strategy(trainer: pl.Trainer) -> None:
        strategy_name = _capability.strategy_name(trainer).lower()
        if "fsdp" in strategy_name or "deepspeed" in strategy_name:
            raise RuntimeError(
                "async_save is not supported with FSDP or DeepSpeed strategies; "
                "use async_save=False."
            )

    @staticmethod
    def _target_path(filepath: str | Path) -> Path:
        filepath_str = os.fspath(filepath)
        parsed = urlparse(filepath_str)
        if parsed.scheme and parsed.scheme != "file":
            raise ValueError(
                "async_save only supports local filesystem paths, "
                f"but got {filepath_str!r}."
            )
        if parsed.scheme == "file":
            return Path(parsed.path)
        return Path(filepath_str)
    @classmethod
    def _same_checkpoint(cls, left: Path, right: str | Path) -> bool:
        try:
            return left.is_file() and left.samefile(cls._target_path(right))
        except (OSError, ValueError):
            return False


__all__ = [
    "ModelCheckpoint",
]
