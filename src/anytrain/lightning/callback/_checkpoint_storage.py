"""Asynchronous checkpoint storage, publication and job lifecycle.

This owner has no Trainer or Lightning callback dependency. The adapter supplies
one publication notification to update Lightning's visible checkpoint path.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from anytrain._audit import AuditSink, _use_audit_context, _use_audit_sink, audit_event, audit_flush


@dataclass
class CheckpointOperation:
    operation_id: str
    fields: dict[str, object]
    started_ns: int
    submit_started_ns: int | None = None
    phase: str = "start"
    serialize_latency_ms: float = 0.0
    copy_latency_ms: float = 0.0
    publication_latency_ms: float = 0.0
    backpressure_wait_ms: float = 0.0
    checkpoint_bytes: int | None = None
    copy_finished_ns: int | None = None
    completion_wait_ms: float = 0.0
    publication_recorded: bool = False
    submit_terminal_emitted: bool = False
    terminal_emitted: bool = False


@dataclass(frozen=True)
class _CheckpointFutureAudit:
    audit: CheckpointOperation
    finalizes_checkpoint: bool = True
    record_publication: bool = True
    record_callback: bool = True


@dataclass
class _StorageJob:
    future: Future[None]
    callbacks: tuple[Callable[[], None], ...] = ()
    publication: tuple[str, str, Path] | None = None
    binding: _CheckpointFutureAudit | None = None
    source: Path | None = None
    target: Path | None = None


_checkpoint_audit_context: ContextVar[CheckpointOperation | None] = ContextVar(
    "anytrain_checkpoint_audit_context",
    default=None,
)
@contextmanager
def _use_checkpoint_audit(audit: CheckpointOperation | None) -> Iterator[None]:
    token = _checkpoint_audit_context.set(audit)
    try:
        yield
    finally:
        _checkpoint_audit_context.reset(token)


class AsyncCheckpointStorage:
    def __init__(
        self, *, capacity: int, audit_sink: AuditSink | None,
        on_publication: Callable[[str, str, Path | None, bool], None],
    ) -> None:
        self.max_pending_async_saves = capacity
        self.audit_sink = audit_sink
        self._on_publication = on_publication
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._async_futures: list[Future[None]] = []
        self._async_copy_futures: list[Future[None]] = []
        self._jobs: dict[Future[None], _StorageJob] = {}
        self._last_persistent_checkpoint: Path | None = None
        self._latest_publication_future: Future[None] | None = None
        self._latest_publication_path: Path | None = None
        self._async_lock = threading.Lock()

    def persistent_state(self) -> Path | None:
        """Snapshot only durable state after all publications have been reaped."""
        with self._async_lock:
            if self._async_futures or self._jobs:
                raise RuntimeError("Cannot serialize ModelCheckpoint while async saves are pending.")
            return self._last_persistent_checkpoint

    def restore_persistent_state(self, persistent: Path | None) -> None:
        with self._async_lock:
            if self._async_futures or self._jobs:
                raise RuntimeError("Cannot restore checkpoint storage while async saves are pending.")
            self._last_persistent_checkpoint = persistent
            self._latest_publication_path = persistent

    def wait_async_saves(self) -> None:
        futures = self._pop_async_futures()
        first_error: BaseException | None = None
        for future in futures:
            error = self._finish_async_future(future)
            if error is not None and first_error is None:
                first_error = error

        if first_error is not None:
            raise RuntimeError("Asynchronous checkpoint storage failed.") from first_error

    def begin_checkpoint(self, fields: dict[str, object]) -> CheckpointOperation:
        """Start an operation from adapter metadata; storage owns phase state."""
        operation_id = uuid4().hex
        fields = {**fields, "operation_id": operation_id, "checkpoint_id": operation_id}
        fields["checkpoint_key"] = (
            f"{fields['global_step']}:{fields['trigger']}:"
            f"{fields['checkpoint_kind']}:{fields['target']}"
        )
        audit = CheckpointOperation(
            operation_id=operation_id, fields=fields, started_ns=time.perf_counter_ns(),
        )
        asynchronous = bool(fields["async_save"])
        self._emit_checkpoint_phase(
            "checkpoint_start", audit,
            execution_scope="mixed" if asynchronous else "foreground",
            blocks_train_loop=not asynchronous,
            **self._async_queue_fields(),
        )
        if asynchronous:
            audit.submit_started_ns = time.perf_counter_ns()
            self._emit_checkpoint_phase(
                "checkpoint_submit_start", audit, phase="submit", **self._async_queue_fields(),
            )
        return audit

    def retarget(self, audit: CheckpointOperation, target: Path) -> None:
        """Record the resolved persistent path before serialization starts."""
        audit.fields["path"] = str(target)
        audit.fields["target"] = str(target)
        audit.fields["checkpoint_key"] = (
            f"{audit.fields['global_step']}:{audit.fields['trigger']}:"
            f"{audit.fields['checkpoint_kind']}:{target!s}"
        )

    def serialize(
        self, audit: CheckpointOperation, path: str | Path, save: Callable[[], None],
    ) -> None:
        """Measure serialization while the adapter supplies the strategy call."""
        audit.phase = "serialize"
        self._emit_checkpoint_phase("checkpoint_serialize_start", audit)
        started = time.perf_counter_ns()
        try:
            save()
        except BaseException as error:
            audit.serialize_latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            self._emit_checkpoint_phase(
                "checkpoint_serialize_error", audit,
                latency_ms=audit.serialize_latency_ms,
                error_type=type(error).__name__, error=str(error)[:1000],
            )
            raise
        audit.serialize_latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if audit.fields["async_save"]:
            local = Path(path)
            audit.checkpoint_bytes = local.stat().st_size if local.is_file() else None
        else:
            audit.checkpoint_bytes = self._checkpoint_size(path)
        self._emit_checkpoint_phase(
            "checkpoint_serialize_done", audit,
            latency_ms=audit.serialize_latency_ms, checkpoint_bytes=audit.checkpoint_bytes,
        )

    @contextmanager
    def callback_phase(self, audit: CheckpointOperation) -> Iterator[None]:
        """Measure synchronous logger notification without knowing Lightning."""
        audit.phase = "callback"
        started = time.perf_counter_ns()
        self._emit_checkpoint_phase("checkpoint_callback_start", audit)
        try:
            yield
        except BaseException as error:
            self._emit_checkpoint_phase(
                "checkpoint_callback_error", audit,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                error_type=type(error).__name__, error=str(error)[:1000],
            )
            raise
        self._emit_checkpoint_phase(
            "checkpoint_callback_done", audit,
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
        )

    def prepare_local(self, target: Path, *, audit: CheckpointOperation | None = None) -> Path:
        """Reap completed jobs and apply backpressure before allocating a file."""
        error = self.collect_error()
        if error is not None:
            raise error
        with _use_checkpoint_audit(audit):
            self._wait_for_async_capacity()
        return self._local_checkpoint_path(target)

    def discard_local(self, path: Path) -> None:
        """Remove a serialization artifact whose publication was not submitted."""
        self._remove_target(path)

    def finish_submit(
        self,
        audit: CheckpointOperation,
        *,
        success: bool,
        publication_pending: bool = False,
        error: BaseException | None = None,
    ) -> None:
        if audit.submit_terminal_emitted:
            return
        fields: dict[str, object] = {
            "phase": "submit",
            "latency_ms": (
                time.perf_counter_ns()
                - (audit.submit_started_ns or audit.started_ns)
            )
            / 1_000_000.0,
            "serialize_latency_ms": audit.serialize_latency_ms,
            "backpressure_wait_ms": audit.backpressure_wait_ms,
            "publication_pending": publication_pending,
            **self._async_queue_fields(),
        }
        if error is not None:
            fields.update(
                error_type=type(error).__name__,
                error=str(error)[:1000],
            )
        self._emit_checkpoint_phase(
            "checkpoint_submit_done" if success else "checkpoint_submit_error",
            audit,
            **fields,
        )
        audit.submit_terminal_emitted = True

    def _emit_checkpoint_phase(
        self,
        event: str,
        audit: CheckpointOperation,
        **fields: object,
    ) -> None:
        payload = {
            **audit.fields,
            "phase": audit.phase,
            "execution_scope": "foreground",
            "blocks_train_loop": True,
            **fields,
        }
        trusted_context = {
            name: payload[name]
            for name in ("rank", "world_size", "device")
            if payload.get(name) is not None
        }
        with _use_audit_sink(self.audit_sink), _use_audit_context(trusted_context):
            audit_event(event, sink=self.audit_sink, **payload)

    def finish_checkpoint(
        self,
        audit: CheckpointOperation,
        *,
        success: bool,
        error: BaseException | None = None,
    ) -> None:
        if audit.terminal_emitted:
            return
        finished_ns = time.perf_counter_ns()
        total_latency_ms = (finished_ns - audit.started_ns) / 1_000_000.0
        if not audit.fields["async_save"] and audit.serialize_latency_ms == 0.0:
            audit.serialize_latency_ms = total_latency_ms
            audit.checkpoint_bytes = self._checkpoint_size(
                str(audit.fields["target"])
            )
            self._emit_checkpoint_phase(
                "checkpoint_serialize_done" if success else "checkpoint_serialize_error",
                audit,
                latency_ms=audit.serialize_latency_ms,
                checkpoint_bytes=audit.checkpoint_bytes,
                error_type=None if error is None else type(error).__name__,
                error=None if error is None else str(error)[:1000],
            )
        durable_latency_ms = (
            total_latency_ms
            if audit.copy_finished_ns is None
            else (audit.copy_finished_ns - audit.started_ns) / 1_000_000.0
        )
        common = {
            "latency_ms": total_latency_ms,
            "durable_latency_ms": durable_latency_ms,
            "serialize_latency_ms": audit.serialize_latency_ms,
            "copy_latency_ms": audit.copy_latency_ms,
            "publication_latency_ms": audit.publication_latency_ms,
            "backpressure_wait_ms": audit.backpressure_wait_ms,
            "completion_wait_ms": audit.completion_wait_ms,
            "checkpoint_bytes": audit.checkpoint_bytes,
            **self._async_queue_fields(),
            "execution_scope": (
                "mixed" if audit.fields["async_save"] else "foreground"
            ),
            "blocks_train_loop": not bool(audit.fields["async_save"]),
        }
        audit.terminal_emitted = True
        if success:
            audit.phase = "done"
            self._emit_checkpoint_phase("checkpoint_done", audit, **common)
            return
        self._emit_checkpoint_phase(
            "checkpoint_error",
            audit,
            **common,
            error_type="RuntimeError" if error is None else type(error).__name__,
            error="checkpoint failed" if error is None else str(error)[:1000],
        )
        audit_flush(sink=self.audit_sink)

    def _async_queue_fields(self) -> dict[str, object]:
        with self._async_lock:
            return self._async_queue_fields_unlocked()

    def _async_queue_fields_unlocked(self) -> dict[str, object]:
        return {
            "queue_depth": len(self._async_copy_futures),
            "pending_async_operations": len(self._async_futures),
            "queue_capacity": self.max_pending_async_saves,
        }

    @staticmethod
    def _checkpoint_size(path: str | Path) -> int | None:
        try:
            candidate = Path(path)
            return candidate.stat().st_size if candidate.is_file() else None
        except (OSError, ValueError):
            return None

    def operation_for(
        self,
        future: Future[None] | None,
    ) -> CheckpointOperation | None:
        if future is None:
            return None
        with self._async_lock:
            binding = self._binding_for(future)
        if binding is None or binding.audit.terminal_emitted:
            return None
        return binding.audit

    def _has_other_checkpoint_audit(
        self,
        future: Future[None],
        audit: CheckpointOperation,
    ) -> bool:
        with self._async_lock:
            return any(
                candidate is not future and binding.audit is audit
                for candidate, job in self._jobs.items()
                if (binding := job.binding) is not None
            )

    def publish_copy(
        self,
        source: Path,
        target: Path,
        *,
        callbacks: Iterable[Callable[[], None]] = (),
        publication: tuple[str, str, Path] | None = None,
        defer_audit_completion: bool = False,
        audit: CheckpointOperation | None = None,
    ) -> Future[None]:
        with _use_checkpoint_audit(audit):
            self._wait_for_async_capacity()
        future = self._async_executor_or_create().submit(
            self._copy_local_checkpoint,
            source,
            target,
            audit,
        )

        def remove_source() -> None:
            source.unlink(missing_ok=True)

        self._track_async_future(
            future,
            callbacks=(*callbacks, remove_source),
            is_copy=True,
            source=source,
            target=target,
            audit=audit,
            finalizes_checkpoint=not defer_audit_completion,
            record_publication=not defer_audit_completion,
            record_callback=not defer_audit_completion,
        )
        if publication is not None:
            self._track_async_publication(future, publication)
        return future

    def publish_last(
        self,
        prerequisite: Future[None] | None,
        persistent: Path,
        last: Path,
        *,
        link: bool,
        callbacks: Iterable[Callable[[], None]] = (),
        publication: tuple[str, str, Path] | None = None,
        audit: CheckpointOperation | None = None,
    ) -> Future[None]:
        if audit is None:
            audit = self.operation_for(prerequisite)
        future = self._async_executor_or_create().submit(
            self._update_last_checkpoint,
            prerequisite,
            persistent,
            last,
            link,
            audit,
        )
        self._track_async_future(
            future,
            callbacks=callbacks,
            audit=audit,
            audit_predecessor=prerequisite,
            source=persistent,
            target=last,
        )
        if publication is not None:
            self._track_async_publication(future, publication)
        return future

    def remove_after_pending(
        self,
        target: Path,
        *,
        audit: CheckpointOperation | None = None,
    ) -> Future[None]:
        with self._async_lock:
            prerequisites = tuple(self._async_futures)
        future = self._async_executor_or_create().submit(
            self._remove_target_after,
            prerequisites,
            target,
            audit,
        )
        self._track_async_future(
            future,
            audit=audit,
            record_publication=False,
            record_callback=False,
            target=target,
        )
        return future

    def _copy_local_checkpoint(
        self,
        source: Path,
        target: Path,
        audit: CheckpointOperation | None = None,
    ) -> None:
        started = time.perf_counter_ns()
        if audit is not None:
            audit.phase = "copy"
            self._emit_checkpoint_phase(
                "checkpoint_copy_start",
                audit,
                source_path=str(source),
                execution_scope="background",
                blocks_train_loop=False,
            )
        try:
            self._copy_to_target(source, target)
        except BaseException as error:
            if audit is not None:
                audit.copy_latency_ms = (
                    time.perf_counter_ns() - started
                ) / 1_000_000.0
                self._emit_checkpoint_phase(
                    "checkpoint_copy_error",
                    audit,
                    source_path=str(source),
                    latency_ms=audit.copy_latency_ms,
                    execution_scope="background",
                    blocks_train_loop=False,
                    error_type=type(error).__name__,
                    error=str(error)[:1000],
                )
            source.unlink(missing_ok=True)
            raise
        if audit is not None:
            audit.copy_finished_ns = time.perf_counter_ns()
            audit.copy_latency_ms = (
                audit.copy_finished_ns - started
            ) / 1_000_000.0
            self._emit_checkpoint_phase(
                "checkpoint_copy_done",
                audit,
                source_path=str(source),
                latency_ms=audit.copy_latency_ms,
                checkpoint_bytes=audit.checkpoint_bytes,
                execution_scope="background",
                blocks_train_loop=False,
            )

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
        audit: CheckpointOperation | None = None,
    ) -> None:
        started = time.perf_counter_ns()
        if audit is not None:
            audit.phase = "publication"
            self._emit_checkpoint_phase(
                "checkpoint_publication_start",
                audit,
                publication_path=str(last),
                publication_mode="link" if link else "copy",
                execution_scope="background",
                blocks_train_loop=False,
            )
        try:
            self._update_last_checkpoint_impl(
                prerequisite,
                persistent,
                last,
                link,
            )
        except BaseException as error:
            if audit is not None:
                audit.publication_latency_ms = (
                    time.perf_counter_ns() - started
                ) / 1_000_000.0
                audit.publication_recorded = True
                self._emit_checkpoint_phase(
                    "checkpoint_publication_error",
                    audit,
                    publication_path=str(last),
                    publication_mode="link" if link else "copy",
                    latency_ms=audit.publication_latency_ms,
                    execution_scope="background",
                    blocks_train_loop=False,
                    error_type=type(error).__name__,
                    error=str(error)[:1000],
                )
            raise
        if audit is not None:
            audit.publication_latency_ms = (
                time.perf_counter_ns() - started
            ) / 1_000_000.0
            audit.publication_recorded = True
            self._emit_checkpoint_phase(
                "checkpoint_publication_done",
                audit,
                publication_path=str(last),
                publication_mode="link" if link else "copy",
                latency_ms=audit.publication_latency_ms,
                execution_scope="background",
                blocks_train_loop=False,
            )

    def _update_last_checkpoint_impl(
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
        audit: CheckpointOperation | None = None,
    ) -> None:
        started = time.perf_counter_ns()
        if audit is not None:
            audit.phase = "remove"
            self._emit_checkpoint_phase(
                "checkpoint_remove_start",
                audit,
                execution_scope="background",
                blocks_train_loop=False,
            )
        try:
            for prerequisite in prerequisites:
                prerequisite.result()
            self._remove_target(target)
        except BaseException as error:
            if audit is not None:
                self._emit_checkpoint_phase(
                    "checkpoint_remove_error",
                    audit,
                    latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                    execution_scope="background",
                    blocks_train_loop=False,
                    error_type=type(error).__name__,
                    error=str(error)[:1000],
                )
            raise
        if audit is not None:
            self._emit_checkpoint_phase(
                "checkpoint_remove_done",
                audit,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                execution_scope="background",
                blocks_train_loop=False,
            )

    def _remove_target(self, target: Path) -> None:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    def _raise_async_error(self) -> None:
        error = self.collect_error()
        if error is not None:
            raise error

    def collect_error(self) -> RuntimeError | None:
        futures = self._drain_done_async_futures()
        first_error: BaseException | None = None
        for future in futures:
            error = self._finish_async_future(future)
            if error is not None and first_error is None:
                first_error = error
        if first_error is not None:
            return _storage_error(first_error)
        return None

    def _wait_for_async_capacity(self) -> None:
        audit = _checkpoint_audit_context.get()
        wait_started: int | None = None
        while True:
            with self._async_lock:
                if len(self._async_copy_futures) < self.max_pending_async_saves:
                    if wait_started is not None and audit is not None:
                        latency_ms = (
                            time.perf_counter_ns() - wait_started
                        ) / 1_000_000.0
                        audit.backpressure_wait_ms += latency_ms
                        self._emit_checkpoint_phase(
                            "checkpoint_backpressure_done",
                            audit,
                            phase="backpressure",
                            latency_ms=latency_ms,
                            **self._async_queue_fields_unlocked(),
                        )
                    return
                if wait_started is None:
                    wait_started = time.perf_counter_ns()
                    if audit is not None:
                        audit.phase = "backpressure"
                        self._emit_checkpoint_phase(
                            "checkpoint_backpressure_start",
                            audit,
                            **self._async_queue_fields_unlocked(),
                        )
                future = self._async_copy_futures[0]
                self._async_futures.remove(future)
            error = self._finish_async_future(future)
            if error is not None:
                if wait_started is not None and audit is not None:
                    latency_ms = (
                        time.perf_counter_ns() - wait_started
                    ) / 1_000_000.0
                    audit.backpressure_wait_ms += latency_ms
                    self._emit_checkpoint_phase(
                        "checkpoint_backpressure_error",
                        audit,
                        phase="backpressure",
                        latency_ms=latency_ms,
                        error_type=type(error).__name__,
                        error=str(error)[:1000],
                    )
                raise RuntimeError("Asynchronous checkpoint storage failed.") from error

    def _finish_async_future(self, future: Future[None]) -> BaseException | None:
        with self._async_lock:
            binding = self._binding_for(future)
        audit = None if binding is None else binding.audit
        waited = audit is not None and not future.done()
        wait_started = time.perf_counter_ns()
        if waited and audit is not None:
            self._emit_checkpoint_phase(
                "checkpoint_wait_start",
                audit,
                phase="completion_wait",
            )
        try:
            future.result()
        except BaseException as error:
            if waited and audit is not None:
                wait_latency_ms = (
                    time.perf_counter_ns() - wait_started
                ) / 1_000_000.0
                audit.completion_wait_ms += wait_latency_ms
                self._emit_checkpoint_phase(
                    "checkpoint_wait_done",
                    audit,
                    phase="completion_wait",
                    latency_ms=wait_latency_ms,
                )
            self._finish_async_publication(future, success=False)
            self._discard_async_callbacks(future)
            self._forget_async_copy_future(future)
            self._forget_async_audit(future)
            if audit is not None:
                self.finish_checkpoint(audit, success=False, error=error)
            return error
        if waited and audit is not None:
            wait_latency_ms = (
                time.perf_counter_ns() - wait_started
            ) / 1_000_000.0
            audit.completion_wait_ms += wait_latency_ms
            self._emit_checkpoint_phase(
                "checkpoint_wait_done",
                audit,
                phase="completion_wait",
                latency_ms=wait_latency_ms,
            )
        with self._async_lock:
            job = self._jobs.get(future)
            has_publication = job is not None and job.publication is not None
        publication_started = time.perf_counter_ns()
        publication_pending = (
            audit is not None
            and binding is not None
            and binding.record_publication
            and has_publication
            and not audit.publication_recorded
        )
        if publication_pending and audit is not None:
            audit.phase = "publication"
            self._emit_checkpoint_phase(
                "checkpoint_publication_start",
                audit,
                # The target copy has already completed in the executor.  This
                # event only records publication bookkeeping while the future
                # is reaped; it does not block the training loop.
                execution_scope="background",
                blocks_train_loop=False,
            )
        self._finish_async_publication(future, success=True)
        if publication_pending and audit is not None:
            audit.publication_latency_ms = (
                time.perf_counter_ns() - publication_started
            ) / 1_000_000.0
            audit.publication_recorded = True
            self._emit_checkpoint_phase(
                "checkpoint_publication_done",
                audit,
                latency_ms=audit.publication_latency_ms,
                execution_scope="background",
                blocks_train_loop=False,
            )
        callback_started = time.perf_counter_ns()
        if audit is not None and binding is not None and binding.record_callback:
            self._emit_checkpoint_phase(
                "checkpoint_callback_start",
                audit,
                phase="callback",
            )
        try:
            self._run_async_callbacks(future)
        except BaseException as error:
            if audit is not None and binding is not None and binding.record_callback:
                self._emit_checkpoint_phase(
                    "checkpoint_callback_error",
                    audit,
                    phase="callback",
                    latency_ms=(time.perf_counter_ns() - callback_started)
                    / 1_000_000.0,
                    error_type=type(error).__name__,
                    error=str(error)[:1000],
                )
            if audit is not None:
                self.finish_checkpoint(audit, success=False, error=error)
            self._forget_async_audit(future)
            return error
        finally:
            self._forget_async_copy_future(future)
        if audit is not None and binding is not None and binding.record_callback:
            self._emit_checkpoint_phase(
                "checkpoint_callback_done",
                audit,
                phase="callback",
                latency_ms=(time.perf_counter_ns() - callback_started) / 1_000_000.0,
            )
        if audit is not None and binding is not None and binding.finalizes_checkpoint:
            self.finish_checkpoint(audit, success=True)
        retain_for_successor = (
            audit is not None
            and binding is not None
            and not binding.finalizes_checkpoint
            and not audit.terminal_emitted
            and not self._has_other_checkpoint_audit(future, audit)
        )
        if not retain_for_successor:
            self._forget_async_audit(future)
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

    def close(self) -> None:
        if self._async_executor is not None:
            self._async_executor.shutdown(wait=True)
            self._async_executor = None
        if self._async_tmp_dir is not None:
            self._async_tmp_dir.cleanup()
            self._async_tmp_dir = None
        with self._async_lock:
            self._async_copy_futures = []
            self._jobs = {}
            self._latest_publication_future = None
            self._latest_publication_path = self._last_persistent_checkpoint

    def _track_async_future(
        self,
        future: Future[None],
        *,
        callbacks: Iterable[Callable[[], None]] = (),
        is_copy: bool = False,
        source: Path | None = None,
        target: Path | None = None,
        audit: CheckpointOperation | None = None,
        audit_predecessor: Future[None] | None = None,
        finalizes_checkpoint: bool = True,
        record_publication: bool = True,
        record_callback: bool = True,
    ) -> None:
        with self._async_lock:
            self._async_futures.append(future)
            if is_copy:
                self._async_copy_futures.append(future)
            job = _StorageJob(future, callbacks=tuple(callbacks), source=source, target=target)
            self._jobs[future] = job
            if audit is not None:
                job.binding = _CheckpointFutureAudit(
                    audit=audit,
                    finalizes_checkpoint=finalizes_checkpoint,
                    record_publication=record_publication,
                    record_callback=record_callback,
                )
            if (
                audit_predecessor is not None
                and (
                    predecessor := self._binding_for(audit_predecessor)
                )
                is not None
                and predecessor.audit is audit
                and audit is not None
                and audit_predecessor not in self._async_futures
            ):
                self._jobs.pop(audit_predecessor, None)

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
            job = self._jobs.get(future)
            callbacks = () if job is None else job.callbacks
            if job is not None:
                job.callbacks = ()
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
            self._discard_finished_job(future)

    def _binding_for(self, future: Future[None]) -> _CheckpointFutureAudit | None:
        job = self._jobs.get(future)
        return None if job is None else job.binding

    def _discard_finished_job(self, future: Future[None]) -> None:
        job = self._jobs.get(future)
        if (job is not None and job.binding is None and job.publication is None
                and not job.callbacks and future not in self._async_copy_futures):
            self._jobs.pop(future, None)

    def _forget_async_audit(self, future: Future[None]) -> None:
        with self._async_lock:
            job = self._jobs.get(future)
            if job is not None:
                job.binding = None
            self._discard_finished_job(future)

    def _discard_async_callbacks(self, future: Future[None]) -> None:
        with self._async_lock:
            job = self._jobs.get(future)
            if job is not None:
                job.callbacks = ()

    def _track_async_publication(
        self,
        future: Future[None],
        publication: tuple[str, str, Path],
    ) -> None:
        with self._async_lock:
            self._jobs[future].publication = publication
            self._latest_publication_future = future
            self._latest_publication_path = publication[2]

    def _finish_async_publication(
        self,
        future: Future[None],
        *,
        success: bool,
    ) -> None:
        with self._async_lock:
            job = self._jobs.get(future)
            publication = None if job is None else job.publication
            if job is not None:
                job.publication = None
            is_latest = self._latest_publication_future is future
        if publication is None:
            return

        expected_current, saved_path, persistent_path = publication
        if success:
            self._last_persistent_checkpoint = persistent_path
        self._on_publication(
            expected_current, saved_path, self._last_persistent_checkpoint, success,
        )

        if is_latest:
            with self._async_lock:
                self._latest_publication_future = None
                self._latest_publication_path = self._last_persistent_checkpoint

    def latest_publication(
        self,
    ) -> tuple[Path | None, Future[None] | None]:
        with self._async_lock:
            if self._latest_publication_path is not None:
                return self._latest_publication_path, self._latest_publication_future
        return self._last_persistent_checkpoint, None
def _storage_error(cause: BaseException) -> RuntimeError:
    error = RuntimeError("Asynchronous checkpoint storage failed.")
    error.__cause__ = cause
    error.__suppress_context__ = True
    return error
