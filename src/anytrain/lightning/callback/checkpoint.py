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

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint


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
    ) -> None:
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
        self.async_save = async_save
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._async_futures: list[Future[None]] = []
        self._async_callbacks: dict[Future[None], tuple[Callable[[], None], ...]] = {}
        self._async_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        if self._async_futures:
            raise RuntimeError("Cannot serialize ModelCheckpoint while async saves are pending.")

        state = self.__dict__.copy()
        state["_async_executor"] = None
        state["_async_tmp_dir"] = None
        state["_async_futures"] = []
        state["_async_callbacks"] = {}
        state["_async_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._async_executor = None
        self._async_tmp_dir = None
        self._async_futures = []
        self._async_callbacks = {}
        self._async_lock = threading.Lock()

    def wait_async_saves(self) -> None:
        if not self.async_save:
            return

        futures = self._pop_async_futures()
        first_error: BaseException | None = None
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self._run_async_callbacks(future)

        if first_error is not None:
            raise RuntimeError("Asynchronous checkpoint storage failed.") from first_error

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            super().on_fit_end(trainer, pl_module)
            self.wait_async_saves()
        finally:
            self._close_async_storage()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        try:
            super().on_exception(trainer, pl_module, exception)
            self.wait_async_saves()
        finally:
            self._close_async_storage()

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        if not self.async_save:
            super()._save_checkpoint(trainer, filepath)
            return

        self._raise_async_error()
        self._validate_async_strategy(trainer)
        target = self._target_path(filepath)
        local = self._local_checkpoint_path(target)

        trainer.save_checkpoint(str(local), self.save_weights_only)
        if not local.is_file():
            raise RuntimeError(
                "async_save requires trainer.save_checkpoint() to produce a regular file; "
                "use async_save=False for directory-based or sharded checkpoints."
            )
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        if trainer.is_global_zero:
            callbacks = tuple(
                lambda logger=logger: logger.after_save_checkpoint(proxy(self))
                for logger in trainer.loggers
            )
            self._submit_async_copy(local, target, callbacks=callbacks)

    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        if not self.async_save:
            super()._remove_checkpoint(trainer, filepath)
            return

        self._raise_async_error()
        if trainer.is_global_zero:
            target = self._target_path(filepath)
            self._submit_async_remove(target)

    def _submit_async_copy(
        self,
        source: Path,
        target: Path,
        *,
        callbacks: Iterable[Callable[[], None]] = (),
    ) -> None:
        future = self._async_executor_or_create().submit(self._copy_to_target, source, target)
        self._track_async_future(future, callbacks=callbacks)

    def _submit_async_remove(self, target: Path) -> None:
        future = self._async_executor_or_create().submit(self._remove_target, target)
        self._track_async_future(future)

    def _copy_to_target(self, source: Path, target: Path) -> None:
        target_tmp = target.with_name(f".{target.name}.part.{uuid4().hex}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_tmp)
            os.replace(target_tmp, target)
        finally:
            target_tmp.unlink(missing_ok=True)
            source.unlink(missing_ok=True)

    def _remove_target(self, target: Path) -> None:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    def _raise_async_error(self) -> None:
        futures = self._drain_done_async_futures()
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                raise RuntimeError("Asynchronous checkpoint storage failed.") from error
            self._run_async_callbacks(future)

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

    def _track_async_future(
        self,
        future: Future[None],
        *,
        callbacks: Iterable[Callable[[], None]] = (),
    ) -> None:
        with self._async_lock:
            self._async_futures.append(future)
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
        for callback in callbacks:
            callback()

    @staticmethod
    def _validate_async_strategy(trainer: pl.Trainer) -> None:
        strategy = getattr(trainer, "strategy", None)
        strategy_name = type(strategy).__name__.lower()
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


__all__ = [
    "ModelCheckpoint",
]
