"""Task-agnostic Lightning training and collective audit hooks."""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from . import _capability

_AUDIT_DIR_ENV = "ANYTRAIN_AUDIT_DIR"
_AUDIT_STEP_ENV = "ANY_AUDIT_STEP"
_AUDIT_KEEP_STEPS_ENV = "ANYTRAIN_AUDIT_KEEP_STEPS"
_AUDIT_FLUSH_INTERVAL_ENV = "ANYTRAIN_AUDIT_FLUSH_INTERVAL"
_DEFAULT_KEEP_STEPS = 8
_DEFAULT_FLUSH_INTERVAL = 1.0
_MAX_EVENTS_PER_SINK = 8192
_audit_step: int | None = None
_T = TypeVar("_T")
BatchReporter = Callable[[object], Mapping[str, object]]


class AuditSink:
    def emit(self, event: str, **fields: object) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()

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
        self._records: deque[dict[str, object]] = deque()
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
        rank = fields.get("rank", context["rank"])
        world_size = fields.get("world_size", context["world_size"])
        record: dict[str, object] = {
            "event": event,
            "time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **{**context, "rank": rank, "world_size": world_size},
            **_step_context(),
        }
        record.update({name: _json_value(value) for name, value in fields.items() if value is not None})
        with self._lock:
            if self._closed:
                return
            self._records.append(record)
            self._prune()
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
                lock_descriptor = os.open(
                    path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o644
                )
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                    records = _bounded_records(_read_records(path) + snapshot, self.keep_steps, self.max_events)
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
        self._records = deque()
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

    def _prune(self) -> None:
        steps = [step for step in (_record_step(record) for record in self._records) if step is not None]
        if steps:
            recent_steps = set(steps[-self.keep_steps :])
            records = (
                record
                for record in self._records
                if (step := _record_step(record)) is None or step in recent_steps
            )
            self._records = deque(records, maxlen=self.max_events)
        elif len(self._records) > self.max_events:
            self._records = deque(list(self._records)[-self.max_events :], maxlen=self.max_events)

    def _path(self, records: Sequence[Mapping[str, object]]) -> Path:
        record = records[-1]
        return self.directory / f"rank-{record.get('rank', 0)}-worker-{record.get('worker', 'main')}.jsonl"


def audit_enabled() -> bool:
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
    actual = sink if sink is not None else _env_sink()
    if actual is not None:
        actual.emit(event, **fields)


def audit_flush(*, sink: AuditSink | None = None) -> None:
    actual = sink if sink is not None else _env_sink()
    if actual is not None:
        actual.flush()


def audit_close(*, sink: AuditSink | None = None) -> None:
    global _cached_sink, _cached_signature, _cached_pid
    if sink is not None:
        sink.close()
        return
    actual = _env_sink()
    if actual is not None:
        actual.close()
    _cached_sink = None
    _cached_signature = None
    _cached_pid = None


def audit_collective(name: str, operation: Callable[[], _T], *, sink: AuditSink | None = None, **fields: object) -> _T:
    if not callable(operation):
        raise TypeError("collective operation must be callable.")
    if not audit_enabled() and sink is None:
        return operation()
    started = time.perf_counter_ns()
    audit_event("collective_start", sink=sink, collective=name, **fields)
    try:
        result = operation()
    except BaseException as error:
        audit_event("collective_error", sink=sink, collective=name, latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0, error_type=type(error).__name__, error=str(error)[:500], **fields)
        audit_flush(sink=sink)
        raise
    audit_event("collective_done", sink=sink, collective=name, latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0, **fields)
    return result


def audit_operation(name: str, operation: Callable[[], _T], *, sink: AuditSink | None = None, result_report: Callable[[object], Mapping[str, object]] | None = None, **fields: object) -> _T:
    """Trace an arbitrary train-side phase such as host-to-device transfer."""
    if not callable(operation):
        raise TypeError("audited operation must be callable.")
    if not audit_enabled() and sink is None:
        return operation()
    started = time.perf_counter_ns()
    audit_event(f"{name}_start", sink=sink, **fields)
    try:
        result = operation()
    except BaseException as error:
        audit_event(f"{name}_error", sink=sink, latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0, error_type=type(error).__name__, error=str(error)[:1000], **fields)
        audit_flush(sink=sink)
        raise
    result_fields = {} if result_report is None else dict(result_report(result))
    audit_event(f"{name}_done", sink=sink, latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0, **fields, **result_fields)
    return result


class TrainingAuditCallback(Callback):
    """Rank/step-aware Lightning lifecycle audit callback."""

    def __init__(self, *, sink: AuditSink | None = None, directory: str | Path | None = None, batch_report: BatchReporter | None = None) -> None:
        super().__init__()
        if sink is not None and directory is not None:
            raise ValueError("pass either sink or directory, not both.")
        self._owns_sink = directory is not None
        if directory is not None:
            sink = JsonlAuditSink(directory)
        if batch_report is not None and not callable(batch_report):
            raise TypeError("batch_report must be callable when provided.")
        self.sink, self.batch_report = sink, batch_report

    def _emit(self, event: str, trainer: Trainer, **fields: object) -> None:
        audit_event(
            event,
            sink=self.sink,
            rank=_capability.global_rank(trainer),
            world_size=_capability.world_size(trainer),
            global_step=_capability.global_step(trainer),
            **fields,
        )

    def _batch_fields(self, batch: object) -> dict[str, object]:
        if self.batch_report is None:
            return {"batch_type": type(batch).__name__}
        return dict(self.batch_report(batch))

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        set_audit_step(_capability.global_step(trainer))
        self._emit("fit_start", trainer)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int) -> None:
        del pl_module
        set_audit_step(_capability.global_step(trainer))
        self._emit("train_batch_start", trainer, batch_idx=batch_idx, **self._batch_fields(batch))

    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss: object) -> None:
        del pl_module
        self._emit("backward_start", trainer, loss_type=type(loss).__name__)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._emit("backward_done", trainer)

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        del pl_module, optimizer
        self._emit("optimizer_step_start", trainer)

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        del pl_module, optimizer
        self._emit("optimizer_step_done", trainer)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: object, batch: object, batch_idx: int) -> None:
        del pl_module, outputs
        set_audit_step(_capability.global_step(trainer))
        self._emit("train_batch_end", trainer, batch_idx=batch_idx, **self._batch_fields(batch))

    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int, dataloader_idx: int = 0) -> None:
        del pl_module
        self._emit("validation_batch_start", trainer, batch_idx=batch_idx, dataloader_idx=dataloader_idx, **self._batch_fields(batch))

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: object, batch: object, batch_idx: int, dataloader_idx: int = 0) -> None:
        del pl_module, outputs
        self._emit("validation_batch_end", trainer, batch_idx=batch_idx, dataloader_idx=dataloader_idx, **self._batch_fields(batch))

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        del pl_module
        self._emit("trainer_exception", trainer, error_type=type(exception).__name__, error=str(exception)[:1000])
        audit_flush(sink=self.sink)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._emit("fit_end", trainer)
        audit_flush(sink=self.sink)
        if self._owns_sink:
            audit_close(sink=self.sink)


_cached_sink: JsonlAuditSink | None = None
_cached_signature: tuple[object, ...] | None = None
_cached_pid: int | None = None


def _env_sink() -> AuditSink | None:
    global _cached_sink, _cached_signature, _cached_pid
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
    return {"rank": rank, "world_size": world_size, "worker": "main", "worker_count": 1, "pid": os.getpid()}


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
            record.get("monotonic_ns") if isinstance(record.get("monotonic_ns"), int) else 0,
            record.get("time_ns") if isinstance(record.get("time_ns"), int) else 0,
        ),
    )
    steps = [step for step in (_record_step(record) for record in ordered) if step is not None]
    if steps:
        recent_steps = set(steps[-keep_steps:])
        ordered = [
            record
            for record in ordered
            if (step := _record_step(record)) is None or step in recent_steps
        ]
    return ordered[-max_events:]


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


__all__ = ["AuditSink", "JsonlAuditSink", "TrainingAuditCallback", "audit_close", "audit_collective", "audit_directory", "audit_enabled", "audit_event", "audit_flush", "audit_flush_interval", "audit_keep_steps", "audit_operation", "set_audit_step"]
