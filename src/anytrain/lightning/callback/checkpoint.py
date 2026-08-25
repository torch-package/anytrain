from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4
from weakref import proxy

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint
from lightning_utilities.core.rank_zero import rank_zero_info

from .. import _capability
from ..schedule import (
    UnitScheduleCallback,
    UnitScheduleUpdate,
    register_unit_schedule_consumer,
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
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._async_futures: list[Future[None]] = []
        self._async_copy_futures: list[Future[None]] = []
        self._async_callbacks: dict[Future[None], tuple[Callable[[], None], ...]] = {}
        self._async_publications: dict[Future[None], tuple[str, str, Path]] = {}
        self._async_sources: dict[Path, Path] = {}
        self._async_source_futures: dict[Path, Future[None]] = {}
        self._last_persistent_checkpoint: Path | None = None
        self._latest_publication_future: Future[None] | None = None
        self._latest_publication_path: Path | None = None
        self._deferred_async_removes: list[str] = []
        self._prepared_last_paths: tuple[str, str] | None = None
        self._reused_last_save: tuple[str, str, Path, Path, Future[None] | None] | None = None
        self._restored_last_checkpoint: Path | None = None
        self._async_lock = threading.Lock()

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
        if self._async_futures or self._deferred_async_removes:
            raise RuntimeError("Cannot serialize ModelCheckpoint while async saves are pending.")

        state = self.__dict__.copy()
        state["_async_executor"] = None
        state["_async_tmp_dir"] = None
        state["_async_futures"] = []
        state["_async_copy_futures"] = []
        state["_async_callbacks"] = {}
        state["_async_publications"] = {}
        state["_async_sources"] = {}
        state["_async_source_futures"] = {}
        state["_latest_publication_future"] = None
        state["_latest_publication_path"] = state.get("_last_persistent_checkpoint")
        state["_deferred_async_removes"] = []
        state["_prepared_last_paths"] = None
        state["_reused_last_save"] = None
        state["_async_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("max_pending_async_saves", 1)
        state.setdefault("unit_schedule", None)
        state.setdefault("every_n_unit_epochs", None)
        state.setdefault("_last_persistent_checkpoint", None)
        state.setdefault("_restored_last_checkpoint", None)
        self.__dict__.update(state)
        self._async_executor = None
        self._async_tmp_dir = None
        self._async_futures = []
        self._async_copy_futures = []
        self._async_callbacks = {}
        self._async_publications = {}
        self._async_sources = {}
        self._async_source_futures = {}
        self._latest_publication_future = None
        self._latest_publication_path = self._last_persistent_checkpoint
        self._deferred_async_removes = []
        self._prepared_last_paths = None
        self._reused_last_save = None
        self._async_lock = threading.Lock()

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
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

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
        super().on_train_end(trainer, pl_module)

    def wait_async_saves(self) -> None:
        if not self.async_save:
            return

        futures = self._pop_async_futures()
        first_error: BaseException | None = None
        for future in futures:
            error = self._finish_async_future(future)
            if error is not None and first_error is None:
                first_error = error

        if first_error is not None:
            raise RuntimeError("Asynchronous checkpoint storage failed.") from first_error

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
            super().on_exception(trainer, pl_module, exception)
        except BaseException as error:
            hook_error = error
        finally:
            self._restore_unused_last_save(prepared)
        self._drain_and_close_async_storage(trainer, hook_error)

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
        if not self.async_save:
            super()._save_checkpoint(trainer, filepath)
            return

        prepared = self._prepared_last_paths
        self._prepared_last_paths = None
        previous_checkpoint_saved = self._last_checkpoint_saved
        try:
            self._validate_async_strategy(trainer)
            target = self._target_path(filepath)
            is_global_zero = _capability.is_global_zero(trainer)
            local = self._prepare_async_save(trainer, target)

            # DDP/strategy collectives may require every rank to enter save_checkpoint,
            # but only global zero typically materializes a regular file.
            trainer.save_checkpoint(str(local), self.save_weights_only)
            self._last_global_step_saved = _capability.global_step(trainer)
            future: Future[None] | None = None
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
                    with self._async_lock:
                        self._async_sources[local] = target

                    callbacks: tuple[Callable[[], None], ...] = ()
                    publication: tuple[str, str, Path] | None = None
                    if prepared is None:
                        callbacks = self._checkpoint_callbacks(trainer, filepath)
                        publication = (str(local), filepath, target)
                    future = self._submit_async_copy(
                        local,
                        target,
                        callbacks=callbacks,
                        publication=publication,
                    )
                except BaseException as error:
                    local_error = error
            else:
                self._last_checkpoint_saved = filepath

            synchronized = self._synchronize_async_error(trainer, local_error)
            if synchronized is not None:
                if is_global_zero and future is None:
                    self._remove_target(local)
                    self._forget_async_source(local)
                self._last_checkpoint_saved = previous_checkpoint_saved
                raise synchronized
            if prepared is not None:
                previous, last = prepared
                self._reused_last_save = (previous, last, local, target, future)
        except BaseException:
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
        if _capability.is_global_zero(trainer):
            local_error = self._collect_done_async_error()
            if local_error is None and not defer_remove:
                try:
                    target = self._target_path(filepath)
                    self._submit_async_remove(target)
                except BaseException as error:
                    local_error = error
        error = self._synchronize_async_error(trainer, local_error)
        if error is not None:
            raise error
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
                local_error = self._collect_done_async_error()
            error = self._synchronize_async_error(trainer, local_error)
            if error is not None:
                raise error

            persistent: Path | None = None
            prerequisite: Future[None] | None = None
            if _capability.is_global_zero(trainer):
                persistent, prerequisite = self._latest_persistent_publication()
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
            if _capability.is_global_zero(trainer):
                assert persistent is not None
                try:
                    self._submit_async_last_update(
                        prerequisite,
                        persistent,
                        Path(last),
                        link=True,
                        publication=(
                            self._last_checkpoint_saved,
                            str(persistent),
                            persistent,
                        ),
                    )
                except BaseException as error:
                    local_error = error
            error = self._synchronize_async_error(trainer, local_error)
            if error is not None:
                self.last_model_path = previous
                raise error
            if previous and self._should_remove_checkpoint(trainer, previous, last):
                self._remove_checkpoint(trainer, previous)
            return

        self._reused_last_save = None
        deferred_removes = self._deferred_async_removes
        self._deferred_async_removes = []
        previous, last, _local, persistent, prerequisite = reused
        local_error = None
        if _capability.is_global_zero(trainer):
            try:
                final_path = str(persistent) if self.save_last == "link" else last
                callbacks = self._checkpoint_callbacks(trainer, final_path)
                self._submit_async_last_update(
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
                )
            except BaseException as error:
                local_error = error
        elif self.save_last is True:
            self._last_checkpoint_saved = last
        error = self._synchronize_async_error(trainer, local_error)
        if error is not None:
            self.last_model_path = previous
            raise error
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

    def _submit_async_copy(
        self,
        source: Path,
        target: Path,
        *,
        callbacks: Iterable[Callable[[], None]] = (),
        publication: tuple[str, str, Path] | None = None,
    ) -> Future[None]:
        self._wait_for_async_capacity()
        future = self._async_executor_or_create().submit(
            self._copy_local_checkpoint,
            source,
            target,
        )

        def remove_source() -> None:
            source.unlink(missing_ok=True)
            self._forget_async_source(source)

        self._track_async_future(
            future,
            callbacks=(*callbacks, remove_source),
            is_copy=True,
        )
        with self._async_lock:
            self._async_source_futures[source] = future
        if publication is not None:
            self._track_async_publication(future, publication)
        return future

    def _submit_async_last_update(
        self,
        prerequisite: Future[None] | None,
        persistent: Path,
        last: Path,
        *,
        link: bool,
        callbacks: Iterable[Callable[[], None]] = (),
        publication: tuple[str, str, Path] | None = None,
    ) -> Future[None]:
        future = self._async_executor_or_create().submit(
            self._update_last_checkpoint,
            prerequisite,
            persistent,
            last,
            link,
        )
        self._track_async_future(future, callbacks=callbacks)
        if publication is not None:
            self._track_async_publication(future, publication)
        return future

    def _submit_async_remove(self, target: Path) -> Future[None]:
        with self._async_lock:
            prerequisites = tuple(self._async_futures)
        future = self._async_executor_or_create().submit(
            self._remove_target_after,
            prerequisites,
            target,
        )
        self._track_async_future(future)
        return future

    def _copy_local_checkpoint(self, source: Path, target: Path) -> None:
        try:
            self._copy_to_target(source, target)
        except BaseException:
            source.unlink(missing_ok=True)
            self._forget_async_source(source)
            raise

    def _copy_to_target(self, source: Path, target: Path) -> None:
        self._copy_to_target_static(source, target)

    @staticmethod
    def _copy_to_target_static(source: Path, target: Path) -> None:
        target_tmp = target.with_name(f".{target.name}.part.{uuid4().hex}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_tmp)
            os.replace(target_tmp, target)
        finally:
            target_tmp.unlink(missing_ok=True)

    def _update_last_checkpoint(
        self,
        prerequisite: Future[None] | None,
        persistent: Path,
        last: Path,
        link: bool,
    ) -> None:
        if prerequisite is not None:
            prerequisite.result()
        if persistent.absolute() == last.absolute():
            return
        directory_backup: Path | None = None
        last_tmp = last.with_name(f".{last.name}.part.{uuid4().hex}")
        try:
            last.parent.mkdir(parents=True, exist_ok=True)
            if last.is_dir() and not last.is_symlink():
                directory_backup = last.with_name(
                    f".{last.name}.backup.{uuid4().hex}"
                )
                os.replace(last, directory_backup)
            if not link:
                self._copy_to_target(persistent, last)
            else:
                try:
                    os.symlink(os.path.relpath(persistent, last.parent), last_tmp)
                    os.replace(last_tmp, last)
                except OSError:
                    last_tmp.unlink(missing_ok=True)
                    self._copy_to_target(persistent, last)
        except BaseException:
            if directory_backup is not None and directory_backup.exists():
                self._remove_target(last)
                os.replace(directory_backup, last)
            raise
        else:
            if directory_backup is not None:
                shutil.rmtree(directory_backup)
        finally:
            last_tmp.unlink(missing_ok=True)

    def _remove_target_after(
        self,
        prerequisites: tuple[Future[None], ...],
        target: Path,
    ) -> None:
        for prerequisite in prerequisites:
            prerequisite.result()
        self._remove_target(target)

    def _remove_target(self, target: Path) -> None:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    def _raise_async_error(self) -> None:
        error = self._collect_done_async_error()
        if error is not None:
            raise error

    def _collect_done_async_error(self) -> RuntimeError | None:
        futures = self._drain_done_async_futures()
        first_error: BaseException | None = None
        for future in futures:
            error = self._finish_async_future(future)
            if error is not None and first_error is None:
                first_error = error
        if first_error is not None:
            return _storage_error(first_error)
        return None

    def _prepare_async_save(self, trainer: pl.Trainer, target: Path) -> Path:
        local = target
        local_error: BaseException | None = None
        if _capability.is_global_zero(trainer):
            local_error = self._collect_done_async_error()
            if local_error is None:
                try:
                    # Apply backpressure before materializing another complete local file.
                    self._wait_for_async_capacity()
                except BaseException as error:
                    local_error = error
            if local_error is None:
                try:
                    local = self._local_checkpoint_path(target)
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
            self._close_async_storage()
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
            synchronized = broadcast(message, src=0)
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

    def _wait_for_async_capacity(self) -> None:
        while True:
            with self._async_lock:
                if len(self._async_copy_futures) < self.max_pending_async_saves:
                    return
                future = self._async_copy_futures[0]
                self._async_futures.remove(future)
            error = self._finish_async_future(future)
            if error is not None:
                raise RuntimeError("Asynchronous checkpoint storage failed.") from error

    def _finish_async_future(self, future: Future[None]) -> BaseException | None:
        try:
            future.result()
        except BaseException as error:
            self._finish_async_publication(future, success=False)
            self._discard_async_callbacks(future)
            self._forget_async_copy_future(future)
            return error
        self._finish_async_publication(future, success=True)
        try:
            self._run_async_callbacks(future)
        except BaseException as error:
            return error
        finally:
            self._forget_async_copy_future(future)
        return None

    def _async_executor_or_create(self) -> ThreadPoolExecutor:
        if self._async_executor is None:
            self._async_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="anytrain-checkpoint",
            )
        return self._async_executor

    def _local_checkpoint_path(self, target: Path) -> Path:
        if self._async_tmp_dir is None:
            self._async_tmp_dir = tempfile.TemporaryDirectory(
                prefix="anytrain-modelsavecheckpoint-"
            )
        suffix = target.suffix or ".ckpt"
        return Path(self._async_tmp_dir.name) / f"{uuid4().hex}{suffix}"

    def _close_async_storage(self) -> None:
        if self._async_executor is not None:
            self._async_executor.shutdown(wait=True)
            self._async_executor = None
        if self._async_tmp_dir is not None:
            self._async_tmp_dir.cleanup()
            self._async_tmp_dir = None
        with self._async_lock:
            self._async_copy_futures = []
            self._async_publications = {}
            self._async_sources = {}
            self._async_source_futures = {}
            self._latest_publication_future = None
            self._latest_publication_path = self._last_persistent_checkpoint
            self._deferred_async_removes = []

    def _track_async_future(
        self,
        future: Future[None],
        *,
        callbacks: Iterable[Callable[[], None]] = (),
        is_copy: bool = False,
    ) -> None:
        with self._async_lock:
            self._async_futures.append(future)
            if is_copy:
                self._async_copy_futures.append(future)
            self._async_callbacks[future] = tuple(callbacks)

    def _pop_async_futures(self) -> list[Future[None]]:
        with self._async_lock:
            futures = self._async_futures
            self._async_futures = []
        return futures

    def _drain_done_async_futures(self) -> list[Future[None]]:
        done: list[Future[None]] = []
        pending: list[Future[None]] = []
        with self._async_lock:
            for future in self._async_futures:
                if future.done():
                    done.append(future)
                else:
                    pending.append(future)
            self._async_futures = pending
        return done

    def _run_async_callbacks(self, future: Future[None]) -> None:
        with self._async_lock:
            callbacks = self._async_callbacks.pop(future, ())
        first_error: BaseException | None = None
        for callback in callbacks:
            try:
                callback()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _forget_async_copy_future(self, future: Future[None]) -> None:
        with self._async_lock:
            if future in self._async_copy_futures:
                self._async_copy_futures.remove(future)
            sources = [
                source
                for source, candidate in self._async_source_futures.items()
                if candidate is future
            ]
            for source in sources:
                self._async_source_futures.pop(source, None)

    def _forget_async_source(self, source: Path) -> None:
        with self._async_lock:
            self._async_sources.pop(source, None)
            self._async_source_futures.pop(source, None)

    def _discard_async_callbacks(self, future: Future[None]) -> None:
        with self._async_lock:
            self._async_callbacks.pop(future, None)

    def _track_async_publication(
        self,
        future: Future[None],
        publication: tuple[str, str, Path],
    ) -> None:
        with self._async_lock:
            self._async_publications[future] = publication
            self._latest_publication_future = future
            self._latest_publication_path = publication[2]

    def _finish_async_publication(
        self,
        future: Future[None],
        *,
        success: bool,
    ) -> None:
        with self._async_lock:
            publication = self._async_publications.pop(future, None)
            is_latest = self._latest_publication_future is future
        if publication is None:
            return

        expected_current, saved_path, persistent_path = publication
        if success:
            self._last_persistent_checkpoint = persistent_path
            self._last_checkpoint_saved = saved_path
        elif self._last_checkpoint_saved == expected_current:
            persistent = self._last_persistent_checkpoint
            self._last_checkpoint_saved = "" if persistent is None else str(persistent)

        if is_latest:
            with self._async_lock:
                self._latest_publication_future = None
                self._latest_publication_path = self._last_persistent_checkpoint

    def _latest_persistent_publication(
        self,
    ) -> tuple[Path | None, Future[None] | None]:
        with self._async_lock:
            if self._latest_publication_path is not None:
                return self._latest_publication_path, self._latest_publication_future
        return self._last_persistent_checkpoint, None

    def _synchronize_rank_zero_flag(self, trainer: pl.Trainer, value: bool) -> bool:
        if _capability.world_size(trainer) <= 1:
            return value

        strategy = getattr(trainer, "strategy", None)
        broadcast = getattr(strategy, "broadcast", None)
        if callable(broadcast):
            synchronized = broadcast(
                value if _capability.is_global_zero(trainer) else None,
                src=0,
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


def _storage_error(cause: BaseException) -> RuntimeError:
    error = RuntimeError("Asynchronous checkpoint storage failed.")
    error.__cause__ = cause
    error.__suppress_context__ = True
    return error


__all__ = [
    "ModelCheckpoint",
]
