"""Task-agnostic Lightning training and collective audit hooks."""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Reversible, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from . import _capability

_AUDIT_DIR_ENV = "ANYTRAIN_AUDIT_DIR"
_AUDIT_STEP_ENV = "ANY_AUDIT_STEP"
_AUDIT_KEEP_STEPS_ENV = "ANYTRAIN_AUDIT_KEEP_STEPS"
_AUDIT_FLUSH_INTERVAL_ENV = "ANYTRAIN_AUDIT_FLUSH_INTERVAL"
_DEFAULT_KEEP_STEPS = 128
_DEFAULT_FLUSH_INTERVAL = 1.0
_MAX_EVENTS_PER_SINK = 8192
_SINK_OWNED_FIELDS = frozenset(
    {
        "schema_version",
        "component",
        "event",
        "time_ns",
        "monotonic_ns",
    }
)
_PROCESS_CONTEXT_FIELDS = frozenset(
    {
        "hostname",
        "run_id",
        "rank",
        "world_size",
        "node_rank",
        "local_rank",
        "local_world_size",
        "device",
        "cuda_visible_devices",
        "worker",
        "worker_count",
        "pid",
    }
)
_CONTEXT_FIELDS = _SINK_OWNED_FIELDS | _PROCESS_CONTEXT_FIELDS | {"step_hint"}
_audit_step: int | None = None
_collective_occurrences: dict[tuple[str | None, str, str], tuple[object, int]] = {}
_collective_occurrence_lock = threading.Lock()
_hostname = socket.gethostname()
_T = TypeVar("_T")
BatchReporter = Callable[[object], Mapping[str, object]]
_audit_context_override: ContextVar[Mapping[str, object] | None] = ContextVar(
    "anytrain_audit_context_override",
    default=None,
)
_audit_sink_context: ContextVar[AuditSink | None] = ContextVar(
    "anytrain_audit_sink_context",
    default=None,
)


class AuditSink:
    def emit(self, event: str, **fields: object) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()


_sink_lifecycle_lock = threading.Lock()
_sink_lifecycle: dict[int, tuple[AuditSink, int, int, bool]] = {}


def _retain_audit_sink(sink: AuditSink | None, *, owner: bool = False) -> None:
    """Keep a shared sink alive while callbacks still may emit into it."""

    if sink is None:
        return
    key = id(sink)
    with _sink_lifecycle_lock:
        current = _sink_lifecycle.get(key)
        if current is None or current[0] is not sink:
            users = owners = 0
            managed = False
        else:
            _, users, owners, managed = current
        _sink_lifecycle[key] = (
            sink,
            users + 1,
            owners + int(owner),
            managed or owner,
        )


def _release_audit_sink(sink: AuditSink | None, *, owner: bool = False) -> None:
    """Release one callback's sink lease and close the final owned sink."""

    if sink is None:
        return
    key = id(sink)
    should_close = False
    with _sink_lifecycle_lock:
        current = _sink_lifecycle.get(key)
        if current is None or current[0] is not sink:
            return
        _, users, owners, managed = current
        users = max(0, users - 1)
        owners = max(0, owners - int(owner))
        if users == 0 and owners == 0:
            _sink_lifecycle.pop(key, None)
            should_close = managed
        else:
            _sink_lifecycle[key] = (sink, users, owners, managed)
    if should_close:
        sink.close()


class JsonlAuditSink(AuditSink):
    """Buffer a bounded recent-step snapshot and flush it asynchronously."""

    def __init__(
        self,
        directory: str | Path,
        *,
        keep_steps: int | None = None,
        flush_interval: float | None = None,
        max_events: int | None = None,
    ) -> None:
        self.directory = Path(directory).expanduser()
        if not str(self.directory):
            raise ValueError("audit directory must not be empty.")
        self.keep_steps = _positive_int(keep_steps, _AUDIT_KEEP_STEPS_ENV, _DEFAULT_KEEP_STEPS)
        self.flush_interval = _non_negative_float(
            flush_interval, _AUDIT_FLUSH_INTERVAL_ENV, _DEFAULT_FLUSH_INTERVAL
        )
        self.max_events = _positive_int(max_events, None, _MAX_EVENTS_PER_SINK)
        self._pid = os.getpid()
        self._records: deque[dict[str, object]] = deque(maxlen=self.max_events)
        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._dirty = False
        self._generation = 0
        self._closed = False
        self._thread: threading.Thread | None = None
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)
        if self.flush_interval > 0:
            self._thread = threading.Thread(
                target=self._flush_loop,
                name="anytrain-audit-flush",
                daemon=True,
            )
            self._thread.start()
        atexit.register(self.close)

    def emit(self, event: str, **fields: object) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("audit event must be a non-empty string.")
        context = _process_context()
        trusted_context = _audit_context_override.get()
        if trusted_context:
            context.update(
                {
                    name: value
                    for name, value in trusted_context.items()
                    if name in _PROCESS_CONTEXT_FIELDS and value is not None
                }
            )
        record: dict[str, object] = {
            "schema_version": 1,
            "component": "anytrain",
            "event": event,
            "time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **context,
            **_step_context(),
        }
        record.update(
            {
                name: _json_value(value)
                for name, value in fields.items()
                if value is not None and name not in _CONTEXT_FIELDS
            }
        )
        with self._lock:
            if self._closed:
                return
            self._records.append(record)
            self._generation += 1
            self._dirty = True

    def flush(self) -> None:
        if os.getpid() != self._pid:
            return
        with self._lock:
            if not self._dirty or not self._records:
                return
            snapshot = list(self._records)
            generation = self._generation
        try:
            with self._flush_lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self._path(snapshot)
                lock_descriptor = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o644)
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                    records = _bounded_records(
                        _read_records(path) + snapshot, self.keep_steps, self.max_events
                    )
                    temporary: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="wb",
                            dir=self.directory,
                            prefix=f".{path.name}.",
                            suffix=".tmp",
                            delete=False,
                        ) as stream:
                            temporary = stream.name
                            for record in records:
                                payload = (
                                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                                ).encode("utf-8")
                                stream.write(payload)
                            stream.flush()
                        os.replace(temporary, path)
                        temporary = None
                    finally:
                        if temporary is not None:
                            with contextlib.suppress(OSError):
                                os.unlink(temporary)
                finally:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                    os.close(lock_descriptor)
        except (OSError, TypeError, ValueError):
            return
        with self._lock:
            if generation == self._generation:
                self._dirty = False

    def close(self) -> None:
        if os.getpid() != self._pid:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            self._stop.set()
            thread.join(timeout=max(1.0, self.flush_interval * 2.0))
        self.flush()

    def discard(self) -> None:
        if os.getpid() != self._pid:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.flush_interval * 2.0))
        with self._lock:
            self._records.clear()
            self._dirty = False

    def __getstate__(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "keep_steps": self.keep_steps,
            "flush_interval": self.flush_interval,
            "max_events": self.max_events,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__init__(
            cast(str | Path, state["directory"]),
            keep_steps=cast(int, state["keep_steps"]),
            flush_interval=cast(float, state["flush_interval"]),
            max_events=cast(int, state["max_events"]),
        )

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.flush_interval):
            with contextlib.suppress(BaseException):
                self.flush()

    def _after_fork(self) -> None:
        self._pid = os.getpid()
        self._records = deque(maxlen=self.max_events)
        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._dirty = False
        self._generation = 0
        self._closed = False
        self._thread = None
        if self.flush_interval > 0:
            self._thread = threading.Thread(
                target=self._flush_loop,
                name="anytrain-audit-flush",
                daemon=True,
            )
            self._thread.start()

    def _path(self, records: Sequence[Mapping[str, object]]) -> Path:
        record = records[-1]
        return (
            self.directory
            / f"rank-{record.get('rank', 0)}-worker-{record.get('worker', 'main')}.jsonl"
        )


def audit_enabled() -> bool:
    if _audit_sink_context.get() is not None:
        return True
    value = os.environ.get(_AUDIT_DIR_ENV)
    return value is not None and bool(value.strip())


def audit_directory() -> Path | None:
    value = os.environ.get(_AUDIT_DIR_ENV)
    return None if value is None or not value.strip() else Path(value).expanduser()


def set_audit_step(step: int) -> None:
    global _audit_step
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("audit step must be a non-negative integer.")
    _audit_step = step
    os.environ[_AUDIT_STEP_ENV] = str(step)


def audit_event(event: str, *, sink: AuditSink | None = None, **fields: object) -> None:
    actual = sink if sink is not None else _audit_sink_context.get()
    if actual is None:
        actual = _env_sink()
    if actual is not None:
        actual.emit(event, **fields)


def audit_flush(*, sink: AuditSink | None = None) -> None:
    actual = sink if sink is not None else _audit_sink_context.get()
    if actual is None:
        actual = _env_sink()
    if actual is not None:
        actual.flush()


@contextlib.contextmanager
def _use_audit_context(fields: Mapping[str, object] | None) -> Iterator[None]:
    """Install trusted integration context for the duration of an event."""

    if not fields:
        yield
        return
    token = _audit_context_override.set(dict(fields))
    try:
        yield
    finally:
        _audit_context_override.reset(token)


@contextlib.contextmanager
def _use_audit_sink(sink: AuditSink | None) -> Iterator[None]:
    """Make an explicitly supplied sink available to nested audit calls."""

    if sink is None:
        yield
        return
    token = _audit_sink_context.set(sink)
    try:
        yield
    finally:
        _audit_sink_context.reset(token)


def audit_close(*, sink: AuditSink | None = None) -> None:
    global _cached_sink, _cached_signature, _cached_pid
    if sink is not None:
        sink.close()
        return
    with _cached_sink_lock:
        actual = _cached_sink if _cached_pid == os.getpid() else None
        if actual is not None:
            actual.close()
        _cached_sink = None
        _cached_signature = None
        _cached_pid = None


def audit_collective(
    name: str, operation: Callable[[], _T], *, sink: AuditSink | None = None, **fields: object
) -> _T:
    if not isinstance(name, str) or not name:
        raise ValueError("collective name must be a non-empty string.")
    if not callable(operation):
        raise TypeError("collective operation must be callable.")
    effective_sink = sink if sink is not None else _audit_sink_context.get()
    if not audit_enabled() and effective_sink is None:
        return operation()

    # Generic integrations commonly forward a broad metadata mapping. Keep
    # wrapper-owned fields authoritative and, importantly, remove keys such as
    # ``event`` and ``collective`` before expanding the mapping into
    # ``audit_event``; otherwise Python raises a duplicate-key TypeError before
    # the collective can run or emit a terminal record.
    for field in (
        "event",
        "collective",
        "execution_scope",
        "blocks_train_loop",
        "status",
        "latency_ms",
        "error_type",
        "error",
    ):
        fields.pop(field, None)

    source_value = fields.pop("source", None)
    source = str(source_value) if source_value is not None else _collective_caller()
    run_id_value = fields.get("run_id")
    run_id = (
        str(run_id_value)
        if run_id_value is not None
        else _first_env(("ANY_AUDIT_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "JOB_ID"))
    )
    step = fields.get("global_step", fields.get("step", _step_context()["step_hint"]))
    collective_seq_value = fields.pop("collective_seq", None)
    if isinstance(collective_seq_value, int) and not isinstance(collective_seq_value, bool):
        collective_seq = collective_seq_value
    else:
        collective_seq = _next_collective_occurrence(name, source, step, run_id)
    operation_id_value = fields.pop("operation_id", None)
    operation_id = (
        str(operation_id_value)
        if operation_id_value is not None
        else (
            f"collective:{name}:{source or 'unknown'}:run:{run_id or 'none'}:"
            f"step:{step}:occurrence:{collective_seq}"
        )
    )
    common = {
        **fields,
        "source": source or "anytrain",
        "run_id": run_id,
        "operation_id": operation_id,
        "collective_seq": collective_seq,
        "execution_scope": "foreground",
        "blocks_train_loop": True,
    }
    trusted_context = {
        name: common[name]
        for name in _PROCESS_CONTEXT_FIELDS
        if name in common and common[name] is not None
    }
    started = time.perf_counter_ns()
    with _use_audit_context(trusted_context):
        audit_event("collective_start", sink=effective_sink, collective=name, **common)
    try:
        # Nested instrumentation inside the collective must retain the same
        # trainer/process identity as the wrapper event, even when the
        # process environment is stale or absent (as in unit tests and some
        # launcher setups).
        with _use_audit_sink(effective_sink), _use_audit_context(trusted_context):
            result = operation()
    except BaseException as error:
        error_fields = dict(common)
        error_fields.update(
            collective=name,
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
            error_type=type(error).__name__,
            error=str(error)[:500],
        )
        with _use_audit_context(trusted_context):
            audit_event("collective_error", sink=effective_sink, **error_fields)
        audit_flush(sink=effective_sink)
        raise
    done_fields = dict(common)
    done_fields.update(
        collective=name,
        latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
    )
    with _use_audit_context(trusted_context):
        audit_event("collective_done", sink=effective_sink, **done_fields)
    return result


def _next_collective_occurrence(
    name: str,
    source: object,
    step: object,
    run_id: str | None,
) -> int:
    """Return a per-callsite occurrence that does not drift with other collectives."""

    key = (run_id, name, str(source or "unknown"))
    with _collective_occurrence_lock:
        previous_step, occurrence = _collective_occurrences.get(key, (step, 0))
        if previous_step != step:
            occurrence = 0
        _collective_occurrences[key] = (step, occurrence + 1)
    return occurrence


def _collective_caller() -> str | None:
    try:
        frame = sys._getframe(2)
    except ValueError:
        return None
    skipped = {__name__, _capability.__name__}
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if isinstance(module, str) and module not in skipped:
            return f"{module}.{frame.f_code.co_name}"
        frame = frame.f_back
    return None


def audit_operation(
    name: str,
    operation: Callable[[], _T],
    *,
    sink: AuditSink | None = None,
    result_report: Callable[[object], Mapping[str, object]] | None = None,
    **fields: object,
) -> _T:
    """Trace a train-side phase, foreground and blocking unless overridden."""
    if not isinstance(name, str) or not name:
        raise ValueError("operation name must be a non-empty string.")
    if not callable(operation):
        raise TypeError("audited operation must be callable.")
    effective_sink = sink if sink is not None else _audit_sink_context.get()
    if not audit_enabled() and effective_sink is None:
        return operation()
    fields.pop("event", None)
    fields.pop("sink", None)
    operation_id = fields.pop("operation_id", uuid4().hex)
    common = {
        "execution_scope": "foreground",
        "blocks_train_loop": True,
        **fields,
        "operation_id": operation_id,
    }
    trusted_context = {
        name: common[name]
        for name in _PROCESS_CONTEXT_FIELDS
        if name in common and common[name] is not None
    }
    started = time.perf_counter_ns()
    with _use_audit_context(trusted_context):
        audit_event(f"{name}_start", sink=effective_sink, **common)
    try:
        # Keep the caller-supplied process identity active while the operation
        # runs so nested audit calls inherit the same trusted rank/device.
        with _use_audit_sink(effective_sink), _use_audit_context(trusted_context):
            result = operation()
    except BaseException as error:
        error_fields = dict(common)
        error_fields.update(
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        with _use_audit_context(trusted_context):
            audit_event(f"{name}_error", sink=effective_sink, **error_fields)
        audit_flush(sink=effective_sink)
        raise
    try:
        with _use_audit_sink(effective_sink), _use_audit_context(trusted_context):
            result_fields = {} if result_report is None else dict(result_report(result))
    except BaseException as error:
        # Reporting is part of the audited operation boundary.  If a caller's
        # reporter fails, close the span before propagating the reporter error;
        # otherwise the summary presents a successful operation as an open
        # stall and hides the actual failure stage.
        error_fields = dict(common)
        error_fields.update(
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
            error_type=type(error).__name__,
            error=str(error)[:1000],
            stage="result_report",
        )
        with _use_audit_context(trusted_context):
            audit_event(f"{name}_error", sink=effective_sink, **error_fields)
        audit_flush(sink=effective_sink)
        raise
    # Reporter fields are optional metadata.  Merge them first and then apply
    # the wrapper-owned identity/timing fields so a reporter cannot overwrite
    # them (or trigger duplicate-key failures that would leave the span open).
    result_fields.pop("event", None)
    result_fields.pop("sink", None)
    done_fields = dict(result_fields)
    done_fields.update(common)
    done_fields["latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
    with _use_audit_context(trusted_context):
        audit_event(f"{name}_done", sink=effective_sink, **done_fields)
    return result


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


_cached_sink: JsonlAuditSink | None = None
_cached_signature: tuple[object, ...] | None = None
_cached_pid: int | None = None
_cached_sink_lock = threading.Lock()


def _after_fork_env_sink() -> None:
    global _cached_pid, _cached_sink_lock, _collective_occurrence_lock
    _cached_sink_lock = threading.Lock()
    _collective_occurrence_lock = threading.Lock()
    _collective_occurrences.clear()
    _cached_pid = None if _cached_sink is None else os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_env_sink)


def _env_sink() -> AuditSink | None:
    global _cached_sink, _cached_signature, _cached_pid
    with _cached_sink_lock:
        directory = audit_directory()
        pid = os.getpid()
        if directory is None:
            if _cached_sink is not None and _cached_pid == pid:
                _cached_sink.discard()
            _cached_sink = None
            _cached_signature = None
            _cached_pid = None
            return None
        signature = (str(directory), audit_keep_steps(), audit_flush_interval())
        if _cached_sink is not None and _cached_pid == pid and _cached_signature == signature:
            return _cached_sink
        if _cached_sink is not None and _cached_pid == pid:
            _cached_sink.close()
        _cached_sink = JsonlAuditSink(
            directory,
            keep_steps=signature[1],
            flush_interval=signature[2],
        )
        _cached_signature, _cached_pid = signature, pid
        return _cached_sink


def audit_keep_steps() -> int:
    return _positive_int(None, _AUDIT_KEEP_STEPS_ENV, _DEFAULT_KEEP_STEPS)


def audit_flush_interval() -> float:
    return _non_negative_float(None, _AUDIT_FLUSH_INTERVAL_ENV, _DEFAULT_FLUSH_INTERVAL)


def _process_context() -> dict[str, object]:
    rank, world_size = _env_int("RANK", 0), _env_int("WORLD_SIZE", 1)
    if rank == 0 and world_size == 1:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
        except (RuntimeError, ValueError):
            pass
    local_rank = _env_int("LOCAL_RANK", 0)
    local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
    node_rank = _first_env_int(("NODE_RANK", "GROUP_RANK", "SLURM_NODEID"), 0)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "hostname": _hostname,
        "run_id": _first_env(("ANY_AUDIT_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "JOB_ID")),
        "rank": rank,
        "world_size": world_size,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "device": None,
        "cuda_visible_devices": cuda_visible_devices,
        "worker": "main",
        "worker_count": 1,
        "pid": os.getpid(),
    }


def _step_context() -> dict[str, object]:
    value = _audit_step
    if value is None:
        raw = os.environ.get(_AUDIT_STEP_ENV)
        try:
            value = None if raw is None else int(raw)
        except ValueError:
            value = None
    return {"step_hint": value}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _first_env_int(names: Sequence[str], default: int) -> int:
    value = _first_env(names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _json_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(child) for child in value]
    return str(value)


def _record_step(record: Mapping[str, object]) -> int | None:
    for name in ("global_step", "step_hint", "step"):
        value = record.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _read_records(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, object]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            records.append(dict(value))
    return records


def _bounded_records(
    records: Sequence[Mapping[str, object]], keep_steps: int, max_events: int
) -> list[dict[str, object]]:
    unique: dict[str, dict[str, object]] = {}
    for record in records:
        normalized = dict(record)
        key = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        unique[key] = normalized
    ordered = sorted(
        unique.values(),
        key=lambda record: (
            record.get("time_ns") if isinstance(record.get("time_ns"), int) else 0,
            record.get("monotonic_ns") if isinstance(record.get("monotonic_ns"), int) else 0,
        ),
    )
    recent_steps = _recent_steps(ordered, keep_steps)
    if recent_steps:
        ordered = [
            record
            for record in ordered
            if (step := _record_step(record)) is None or step in recent_steps
        ]
    return ordered[-max_events:]


def _recent_steps(
    records: Reversible[Mapping[str, object]],
    keep_steps: int,
) -> set[int]:
    recent: set[int] = set()
    for record in reversed(records):
        step = _record_step(record)
        if step is None or step in recent:
            continue
        recent.add(step)
        if len(recent) >= keep_steps:
            break
    return recent


def _positive_int(value: int | None, env_name: str | None, default: int) -> int:
    if value is None:
        raw = None if env_name is None else os.environ.get(env_name)
        try:
            value = default if raw is None else int(raw)
        except (TypeError, ValueError):
            value = default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{env_name} must be a positive integer.")
    return value


def _non_negative_float(value: float | None, env_name: str, default: float) -> float:
    if value is None:
        raw = os.environ.get(env_name)
        try:
            value = default if raw is None else float(raw)
        except (TypeError, ValueError):
            value = default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{env_name} must be a non-negative number.")
    return float(value)


__all__ = [
    "AuditSink",
    "JsonlAuditSink",
    "TrainingAuditCallback",
    "audit_close",
    "audit_collective",
    "audit_directory",
    "audit_enabled",
    "audit_event",
    "audit_flush",
    "audit_flush_interval",
    "audit_keep_steps",
    "audit_operation",
    "set_audit_step",
]
