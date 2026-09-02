from __future__ import annotations

import gc
import json
import pickle
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

import anytrain.lightning.audit as audit_module
from anytrain.lightning import _capability
from anytrain.lightning.audit import (
    AuditSink,
    JsonlAuditSink,
    TrainingAuditCallback,
    audit_close,
    audit_collective,
    audit_operation,
)


class _AuditCheckpointTrainer:
    def __init__(self, payload: bytes = b"checkpoint") -> None:
        self.global_step = 7
        self.global_rank = 0
        self.world_size = 1
        self.is_global_zero = True
        self.loggers: list[object] = []
        self.strategy = SimpleNamespace(root_device="cpu")
        self.payload = payload

    def save_checkpoint(
        self,
        filepath: str | Path,
        weights_only: bool | None = None,
        storage_options: object | None = None,
    ) -> None:
        del weights_only, storage_options
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)


def _assert_checkpoint_spans_pair(records: list[dict[str, object]]) -> None:
    starts: Counter[tuple[object, str]] = Counter()
    terminals: Counter[tuple[object, str]] = Counter()
    for record in records:
        event = str(record["event"])
        if event.endswith("_start"):
            starts[(record.get("operation_id"), event.removesuffix("_start"))] += 1
        elif event.endswith("_done"):
            terminals[(record.get("operation_id"), event.removesuffix("_done"))] += 1
        elif event.endswith("_error"):
            terminals[(record.get("operation_id"), event.removesuffix("_error"))] += 1
    assert starts == terminals


def test_sink_emit_uses_a_fixed_capacity_buffer_without_step_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_step_scan(*args: object, **kwargs: object) -> set[int]:
        del args, kwargs
        raise AssertionError("emit must not scan the retained step window")

    monkeypatch.setattr(audit_module, "_recent_steps", unexpected_step_scan)
    sink = JsonlAuditSink(tmp_path, flush_interval=0, max_events=3)
    for step in range(10):
        sink.emit("point", global_step=step)

    assert sink._records.maxlen == 3
    assert [record["global_step"] for record in sink._records] == [7, 8, 9]
    sink.discard()


def test_env_sink_initialization_is_thread_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANYTRAIN_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("ANYTRAIN_AUDIT_FLUSH_INTERVAL", "0")
    audit_close()
    ready = Event()
    release = Event()
    original = audit_module.JsonlAuditSink
    constructors = 0

    class PausingSink(original):
        def __init__(
            self,
            directory: str | Path,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal constructors
            constructors += 1
            if constructors == 1:
                ready.set()
                assert release.wait(timeout=5)
            super().__init__(directory, *args, **kwargs)

    monkeypatch.setattr(audit_module, "JsonlAuditSink", PausingSink)
    threads = [
        Thread(target=audit_module.audit_event, args=(f"thread_{index}",)) for index in range(2)
    ]
    threads[0].start()
    assert ready.wait(timeout=5)
    threads[1].start()
    release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    audit_close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert constructors == 1
    assert {record["event"] for record in records} == {"thread_0", "thread_1"}


def test_audit_operation_defaults_to_foreground_blocking(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    result = audit_operation("batch_transfer", lambda: "batch", sink=sink)
    sink.close()

    assert result == "batch"
    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "batch_transfer_start",
        "batch_transfer_done",
    ]
    assert all(record["execution_scope"] == "foreground" for record in records)
    assert all(record["blocks_train_loop"] is True for record in records)


def test_audit_operation_context_covers_nested_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    def operation() -> str:
        audit_module.audit_event(
            "nested_event",
            rank=99,
            world_size=99,
        )
        return "done"

    audit_operation(
        "outer_phase",
        operation,
        sink=sink,
        rank=3,
        world_size=4,
    )
    sink.close()

    path = tmp_path / "rank-3-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    nested = next(record for record in records if record["event"] == "nested_event")
    assert nested["rank"] == 3
    assert nested["world_size"] == 4


def test_audit_operation_result_report_inherits_explicit_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporter callbacks must not fall back to the environment sink."""

    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    def report(_result: object) -> dict[str, object]:
        audit_module.audit_event("result_report_nested", source="test")
        return {"reported": True}

    result = audit_operation(
        "reported_phase",
        lambda: "value",
        sink=sink,
        result_report=report,
        rank=2,
        world_size=3,
    )
    sink.close()

    assert result == "value"
    path = tmp_path / "rank-2-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "reported_phase_start",
        "result_report_nested",
        "reported_phase_done",
    ]
    assert all(record["rank"] == 2 for record in records)
    assert all(record["world_size"] == 3 for record in records)
    assert records[-1]["reported"] is True


def test_nested_audit_operation_error_keeps_inherited_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    def outer() -> None:
        with pytest.raises(RuntimeError, match="nested failure"):
            audit_operation(
                "inner_phase",
                lambda: (_ for _ in ()).throw(RuntimeError("nested failure")),
            )

    audit_operation("outer_phase", outer, sink=sink, rank=1, world_size=2)
    sink.close()

    path = tmp_path / "rank-1-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "outer_phase_start",
        "inner_phase_start",
        "inner_phase_error",
        "outer_phase_done",
    ]


def test_nested_audit_collective_error_keeps_inherited_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    def outer() -> None:
        with pytest.raises(RuntimeError, match="nested collective failure"):
            audit_collective(
                "inner_collective",
                lambda: (_ for _ in ()).throw(RuntimeError("nested collective failure")),
            )

    audit_operation("outer_phase", outer, sink=sink, rank=1, world_size=2)
    sink.close()

    path = tmp_path / "rank-1-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "outer_phase_start",
        "collective_start",
        "collective_error",
        "outer_phase_done",
    ]


def test_audit_operation_result_report_failure_closes_span(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    with pytest.raises(RuntimeError, match="report failed"):
        audit_operation(
            "reported_phase",
            lambda: object(),
            sink=sink,
            result_report=lambda _result: (_ for _ in ()).throw(RuntimeError("report failed")),
        )
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "reported_phase_start",
        "reported_phase_error",
    ]
    assert records[0]["operation_id"] == records[1]["operation_id"]
    assert records[1]["stage"] == "result_report"


def test_audit_operation_ignores_reserved_event_sink_metadata(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    result = audit_operation(
        "reserved_phase",
        lambda: "ok",
        sink=sink,
        event="caller-event",
        sink_metadata="kept-as-task-field",
        result_report=lambda _result: {
            "event": "report-event",
            "sink": "report-sink",
            "reported": True,
        },
    )
    sink.close()

    assert result == "ok"
    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "reserved_phase_start",
        "reserved_phase_done",
    ]
    assert records[-1]["reported"] is True


def test_training_callback_is_a_noop_when_audit_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_module.audit_close()

    def report(_batch: object) -> Mapping[str, object]:
        raise AssertionError("disabled audit must not invoke batch_report")

    callback = TrainingAuditCallback(batch_report=report)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=0,
        ),
    )
    module = cast(Any, object())
    callback.on_fit_start(trainer, module)
    callback.on_train_batch_start(trainer, module, object(), 0)
    callback.on_before_backward(trainer, module, object())
    callback.on_after_backward(trainer, module)
    callback.on_before_optimizer_step(trainer, module, object())
    callback.on_before_zero_grad(trainer, module, object())
    callback.on_train_batch_end(trainer, module, None, object(), 0)
    callback.on_fit_end(trainer, module)


def test_training_batch_report_failure_is_a_point_error(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = TrainingAuditCallback(
        sink=sink,
        batch_report=lambda _batch: (_ for _ in ()).throw(RuntimeError("batch report failed")),
    )
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=0,
        ),
    )

    with pytest.raises(RuntimeError, match="batch report failed"):
        callback.on_train_batch_start(trainer, cast(Any, object()), [1], 0)
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["batch_report_error"]
    assert records[0]["phase"] == "train_batch_start"
    assert records[0]["stage"] == "batch_report"


def test_training_callback_context_captures_nested_batch_report_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    def report(_batch: object) -> dict[str, object]:
        audit_module.audit_event("batch_report_nested", source="test")
        return {"sample_id": "sample-1"}

    callback = TrainingAuditCallback(sink=sink, batch_report=report)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cuda:5"),
            global_rank=5,
            world_size=8,
            global_step=0,
        ),
    )
    callback.on_train_batch_start(trainer, cast(Any, object()), [1], 0)
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-5-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "batch_report_nested",
        "train_batch_start",
        "training_step_start",
    ]
    assert all(record["rank"] == 5 for record in records)
    assert all(record["world_size"] == 8 for record in records)
    assert all(record["device"] == "cuda:5" for record in records)


def test_training_batch_report_cannot_override_structural_fields(
    tmp_path: Path,
) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    def report(_batch: object) -> dict[str, object]:
        return {
            "event": "caller-event",
            "sink": "caller-sink",
            "key": "task-key",
            "trainer": "task-trainer",
            "operation_id": "caller-operation",
            "execution_scope": "background",
            "blocks_train_loop": False,
            "rank": 99,
            "world_size": 99,
            "custom_field": "kept",
        }

    callback = TrainingAuditCallback(sink=sink, batch_report=report)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=2,
            world_size=4,
            global_step=3,
        ),
    )

    callback.on_train_batch_start(trainer, cast(Any, object()), [1], 0)
    callback.on_train_batch_end(trainer, cast(Any, object()), None, [1], 0)
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-2-worker-main.jsonl").read_text().splitlines()
    ]
    train_records = [record for record in records if str(record["event"]).startswith("train_batch")]
    assert [record["event"] for record in train_records] == [
        "train_batch_start",
        "train_batch_end",
    ]
    assert train_records[0]["operation_id"] == train_records[1]["operation_id"]
    assert train_records[0]["operation_id"] != "caller-operation"
    assert all(record["execution_scope"] == "foreground" for record in train_records)
    assert all(record["blocks_train_loop"] is True for record in train_records)
    assert all(record["rank"] == 2 for record in train_records)
    assert all(record["world_size"] == 4 for record in train_records)
    assert all(record["key"] == "task-key" for record in train_records)
    assert all(record["trainer"] == "task-trainer" for record in train_records)
    assert all(record["custom_field"] == "kept" for record in train_records)
    assert all("sink" not in record for record in train_records)


def test_training_exception_closes_active_phase_spans(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = TrainingAuditCallback(sink=sink)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=4,
        ),
    )

    callback.on_train_batch_start(trainer, cast(Any, object()), [1], 3)
    callback.on_exception(
        trainer,
        cast(Any, object()),
        ValueError("worker stalled"),
    )
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    events = [record["event"] for record in records]
    assert events == [
        "train_batch_start",
        "training_step_start",
        "trainer_exception",
        "train_batch_error",
        "training_step_error",
    ]
    starts = {
        record["operation_id"]: record["event"]
        for record in records
        if str(record["event"]).endswith("_start")
    }
    terminals = {
        record["operation_id"]: record["event"]
        for record in records
        if str(record["event"]).endswith("_error")
    }
    assert set(starts) == set(terminals)
    assert all(
        record["status"] == "trainer_exception"
        and record["stage"] == "trainer_exception"
        and record["error_type"] == "ValueError"
        for record in records
        if str(record["event"]).endswith("_error")
    )
    assert records[2]["active_phases"] == ["train_batch", "training_step"]


def test_fit_end_closes_active_phase_spans_with_terminal_status(
    tmp_path: Path,
) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = TrainingAuditCallback(sink=sink)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=4,
        ),
    )

    callback.on_train_batch_start(trainer, cast(Any, object()), [1], 3)
    callback.on_fit_end(trainer, cast(Any, object()))
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "train_batch_start",
        "training_step_start",
        "train_batch_end",
        "training_step_end",
        "fit_end",
    ]
    starts = {
        record["operation_id"]: record["event"]
        for record in records
        if str(record["event"]).endswith("_start")
    }
    terminals = {
        record["operation_id"]: record
        for record in records
        if str(record["event"]).endswith("_end")
        and record["event"] != "fit_end"
    }
    assert set(starts) == set(terminals)
    assert all(
        record["status"] == "fit_end" and record["stage"] == "fit_end"
        for record in terminals.values()
    )
    assert records[-1]["active_phases"] == ["train_batch", "training_step"]


def test_capability_collectives_emit_audit_spans(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANYTRAIN_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("ANYTRAIN_AUDIT_FLUSH_INTERVAL", "0")
    monkeypatch.setenv("ANY_AUDIT_RUN_ID", "collective-test")

    class Strategy:
        def reduce(self, tensor, *, reduce_op):
            return tensor

        def barrier(self, name=None):
            return None

        def all_gather(self, tensor):
            return tensor

        def broadcast(self, value, *, src):
            return value

    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=Strategy(),
            global_rank=0,
            world_size=1,
            global_step=7,
        ),
    )
    value = torch.tensor([1.0])
    _capability.strategy_reduce(trainer, value, reduce_op="sum")
    _capability.strategy_barrier(trainer, name="audit-barrier")
    _capability.strategy_all_gather(trainer, value)
    _capability.strategy_broadcast(trainer, "payload", src=0)
    audit_close()

    path = tmp_path / "rank-0-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert events.count("collective_start") == 4
    assert events.count("collective_done") == 4
    assert {
        record.get("collective") for record in records if record["event"] == "collective_start"
    } == {
        "strategy.reduce",
        "strategy.barrier",
        "strategy.all_gather",
        "strategy.broadcast",
    }
    barrier = next(
        record
        for record in records
        if record["event"] == "collective_start" and record["collective"] == "strategy.barrier"
    )
    assert barrier["barrier_name"] == "audit-barrier"
    assert barrier["run_id"] == "collective-test"
    assert barrier["source"].endswith("test_capability_collectives_emit_audit_spans")


def test_optimizer_phases_use_one_span_name_and_keep_all_occurrences(
    tmp_path: Path,
) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = TrainingAuditCallback(sink=sink)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=3,
        ),
    )
    module = cast(Any, object())
    optimizer = cast(Any, object())

    callback.on_train_batch_start(trainer, module, [1], 0)
    callback.on_before_optimizer_step(trainer, module, optimizer)
    callback.on_before_optimizer_step(trainer, module, optimizer)
    callback.on_train_batch_end(trainer, module, None, [1], 0)
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    optimizer_records = [
        record for record in records if str(record["event"]).startswith("optimizer_step")
    ]
    assert [record["event"] for record in optimizer_records] == [
        "optimizer_step_start",
        "optimizer_step_start",
        "optimizer_step_done",
        "optimizer_step_done",
    ]
    starts = {
        record["operation_id"]
        for record in optimizer_records
        if record["event"] == "optimizer_step_start"
    }
    terminals = {
        record["operation_id"]
        for record in optimizer_records
        if record["event"] == "optimizer_step_done"
    }
    assert starts == terminals
    assert len(starts) == 2
    assert all(
        record["latency_scope"] == "before_optimizer_step_to_train_batch_end"
        for record in optimizer_records
        if record["event"] == "optimizer_step_done"
    )


def test_accumulation_batch_does_not_invent_optimizer_terminal(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = TrainingAuditCallback(sink=sink)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=3,
        ),
    )
    module = cast(Any, object())

    callback.on_train_batch_start(trainer, module, [1], 0)
    callback.on_train_batch_end(trainer, module, None, [1], 0)
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert not any(str(record["event"]).startswith("optimizer_step") for record in records)
    training_step = [
        record for record in records if str(record["event"]).startswith("training_step")
    ]
    assert [record["event"] for record in training_step] == [
        "training_step_start",
        "training_step_done",
    ]
    assert training_step[0]["operation_id"] == training_step[1]["operation_id"]
    assert training_step[1]["status"] == "batch_ended_without_backward_hook"
    assert training_step[1]["latency_scope"] == "train_batch_start_to_train_batch_end"


def test_training_step_closes_at_first_backward_hook(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = TrainingAuditCallback(sink=sink)
    trainer = cast(
        Any,
        SimpleNamespace(
            strategy=SimpleNamespace(root_device="cpu"),
            global_rank=0,
            world_size=1,
            global_step=3,
        ),
    )
    module = cast(Any, object())

    callback.on_train_batch_start(trainer, module, [1], 0)
    callback.on_before_backward(trainer, module, torch.tensor(1.0))
    callback.on_after_backward(trainer, module)
    callback.on_train_batch_end(trainer, module, None, [1], 0)
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    training_step = [
        record for record in records if str(record["event"]).startswith("training_step")
    ]
    assert [record["event"] for record in training_step] == [
        "training_step_start",
        "training_step_done",
    ]
    assert training_step[0]["operation_id"] == training_step[1]["operation_id"]
    assert training_step[1]["status"] == "backward_started"
    assert training_step[1]["latency_scope"] == "train_batch_start_to_before_backward"


def test_collective_occurrences_are_isolated_by_source(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    audit_collective("all_reduce", lambda: None, sink=sink, source="source-a")
    audit_collective("all_reduce", lambda: None, sink=sink, source="source-b")
    audit_collective("all_reduce", lambda: None, sink=sink, source="source-a")
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    starts = [record for record in records if record["event"] == "collective_start"]
    source_a = [record for record in starts if record["source"] == "source-a"]
    source_b = [record for record in starts if record["source"] == "source-b"]
    assert [record["collective_seq"] for record in source_a] == [0, 1]
    assert [record["collective_seq"] for record in source_b] == [0]
    assert len({record["operation_id"] for record in starts}) == 3


def test_collective_wrapper_owns_structural_and_terminal_fields(
    tmp_path: Path,
) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    result = audit_collective(
        "all_reduce",
        lambda: "result",
        sink=sink,
        event="caller-event",
        collective="caller-collective",
        execution_scope="background",
        blocks_train_loop=False,
        status="caller-status",
        latency_ms=-1.0,
        error_type="CallerError",
        error="caller-error",
        collective_seq=7,
        source="reserved-fields-source",
        custom_field="kept",
    )
    sink.close()

    assert result == "result"
    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "collective_start",
        "collective_done",
    ]
    assert all(record["collective"] == "all_reduce" for record in records)
    assert all(record["collective_seq"] == 7 for record in records)
    assert all(record["execution_scope"] == "foreground" for record in records)
    assert all(record["blocks_train_loop"] is True for record in records)
    assert all(record["custom_field"] == "kept" for record in records)
    assert all("status" not in record for record in records)
    assert all("error_type" not in record for record in records)
    assert all("error" not in record for record in records)
    assert "latency_ms" not in records[0]
    assert records[1]["latency_ms"] >= 0


def test_collective_default_operation_id_uses_none_for_missing_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ANY_AUDIT_RUN_ID",
        "TORCHELASTIC_RUN_ID",
        "SLURM_JOB_ID",
        "JOB_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    audit_collective(
        "all_reduce",
        lambda: None,
        sink=sink,
        source="missing-run-source",
    )
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    assert all("run:none:" in record["operation_id"] for record in records)
    assert all("run:None:" not in record["operation_id"] for record in records)


def test_collective_occurrences_restart_for_each_run(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path, flush_interval=0)

    audit_collective(
        "all_reduce",
        lambda: None,
        sink=sink,
        source="run-isolation-source",
        run_id="retry-a",
    )
    audit_collective(
        "all_reduce",
        lambda: None,
        sink=sink,
        source="run-isolation-source",
        run_id="retry-a",
    )
    audit_collective(
        "all_reduce",
        lambda: None,
        sink=sink,
        source="run-isolation-source",
        run_id="retry-b",
    )
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    starts = [record for record in records if record["event"] == "collective_start"]
    assert [(record["run_id"], record["collective_seq"]) for record in starts] == [
        ("retry-a", 0),
        ("retry-a", 1),
        ("retry-b", 0),
    ]
    assert all(f"run:{record['run_id']}" in record["operation_id"] for record in starts)


def test_sync_checkpoint_audit_spans_pair_and_share_checkpoint_key(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    callback = ModelCheckpoint(async_save=False)
    target = tmp_path / "sync.ckpt"
    with patch(
        "anytrain.lightning.callback.checkpoint.audit_event",
        side_effect=capture,
    ):
        callback._save_checkpoint(cast(Any, _AuditCheckpointTrainer()), str(target))

    _assert_checkpoint_spans_pair(records)
    assert {record["event"] for record in records} == {
        "checkpoint_start",
        "checkpoint_serialize_start",
        "checkpoint_serialize_done",
        "checkpoint_callback_start",
        "checkpoint_callback_done",
        "checkpoint_done",
    }
    checkpoint_keys = {record.get("checkpoint_key") for record in records}
    assert checkpoint_keys == {f"7:lightning:periodic:{target}"}


def test_checkpoint_audit_uses_explicit_sink_without_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = ModelCheckpoint(async_save=False, audit_sink=sink)
    target = tmp_path / "explicit-sink.ckpt"

    callback._save_checkpoint(cast(Any, _AuditCheckpointTrainer()), str(target))
    sink.close()

    path = tmp_path / "rank-0-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert "checkpoint_start" in events
    assert "checkpoint_done" in events
    assert all(record["component"] == "anytrain" for record in records)
    assert {record["checkpoint_key"] for record in records} == {
        f"7:lightning:periodic:{target}"
    }


def test_async_checkpoint_audit_uses_explicit_sink_in_worker_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    monkeypatch.delenv("ANYTRAIN_AUDIT_DIR", raising=False)
    audit_close()
    sink = JsonlAuditSink(tmp_path, flush_interval=0)
    callback = ModelCheckpoint(async_save=True, audit_sink=sink)
    target = tmp_path / "explicit-async-sink.ckpt"
    try:
        callback._save_checkpoint(
            cast(Any, _AuditCheckpointTrainer(payload=b"async-sink")),
            str(target),
        )
        callback.wait_async_saves()
    finally:
        callback._close_async_storage()
        sink.close()

    path = tmp_path / "rank-0-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    events = {record["event"] for record in records}
    assert {"checkpoint_copy_start", "checkpoint_copy_done", "checkpoint_done"} <= events
    assert target.read_bytes() == b"async-sink"


def test_discarded_callbacks_release_explicit_sink_leases(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    training_sink = JsonlAuditSink(tmp_path / "training", flush_interval=0)
    training = TrainingAuditCallback(sink=training_sink)
    training_id = id(training_sink)
    training_sink.close()
    del training

    checkpoint_sink = JsonlAuditSink(tmp_path / "checkpoint", flush_interval=0)
    checkpoint = ModelCheckpoint(async_save=False, audit_sink=checkpoint_sink)
    checkpoint_id = id(checkpoint_sink)
    checkpoint_sink.close()
    del checkpoint
    gc.collect()

    assert training_id not in audit_module._sink_lifecycle
    assert checkpoint_id not in audit_module._sink_lifecycle


def test_training_callback_re_registers_owned_sink_after_pickle(
    tmp_path: Path,
) -> None:
    callback = TrainingAuditCallback(directory=tmp_path, batch_report=None)
    payload = pickle.dumps(callback)

    # Finish and release the source callback before checking the reconstructed
    # lease, so the two callbacks do not mask each other's ownership.
    callback.on_fit_end(
        cast(
            Any,
            SimpleNamespace(
                strategy=SimpleNamespace(root_device="cpu"),
                global_rank=0,
                world_size=1,
                global_step=0,
            ),
        ),
        cast(Any, object()),
    )
    restored = pickle.loads(payload)
    restored_id = id(restored.sink)
    assert restored_id in audit_module._sink_lifecycle
    assert not restored.sink._closed

    restored.on_fit_end(
        cast(
            Any,
            SimpleNamespace(
                strategy=SimpleNamespace(root_device="cpu"),
                global_rank=0,
                world_size=1,
                global_step=0,
            ),
        ),
        cast(Any, object()),
    )
    assert restored.sink._closed
    assert restored_id not in audit_module._sink_lifecycle


def test_training_callback_drops_process_local_sink_when_pickled() -> None:
    class ProcessLocalSink(AuditSink):
        def __init__(self) -> None:
            self.lock = Lock()

        def emit(self, event: str, **fields: object) -> None:
            del event, fields

    sink = ProcessLocalSink()
    callback = TrainingAuditCallback(sink=sink)
    sink_id = id(sink)

    restored = pickle.loads(pickle.dumps(callback))

    assert restored.sink is None
    del callback
    gc.collect()
    assert sink_id not in audit_module._sink_lifecycle


def test_shared_owned_audit_sink_stays_open_until_checkpoint_drain(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    training = TrainingAuditCallback(directory=tmp_path)
    checkpoint = ModelCheckpoint(async_save=True, audit_sink=training.sink)
    trainer = _AuditCheckpointTrainer(payload=b"shared-sink")
    copy_started = Event()
    release_copy = Event()
    original_copy = checkpoint._copy_to_target

    def blocked_copy(source: Path, target: Path) -> None:
        copy_started.set()
        assert release_copy.wait(timeout=5)
        original_copy(source, target)

    try:
        with patch.object(checkpoint, "_copy_to_target", side_effect=blocked_copy):
            checkpoint._save_checkpoint(trainer, str(tmp_path / "shared.ckpt"))
            assert copy_started.wait(timeout=2)

            # Callback ordering is not guaranteed. The directory-owning
            # callback may finish before the checkpoint callback drains its
            # background copy, but terminal checkpoint events must survive.
            training.on_fit_end(trainer, cast(Any, object()))
            release_copy.set()
            checkpoint.wait_async_saves()
            assert not training.sink._closed
    finally:
        release_copy.set()
        checkpoint._drain_and_close_async_storage(trainer, None)

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-0-worker-main.jsonl").read_text().splitlines()
    ]
    events = {record["event"] for record in records}
    assert {"checkpoint_copy_done", "checkpoint_done"} <= events


def test_sync_checkpoint_failure_closes_serialize_and_checkpoint_spans(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    class FailingTrainer(_AuditCheckpointTrainer):
        def save_checkpoint(
            self,
            filepath: str | Path,
            weights_only: bool | None = None,
            storage_options: object | None = None,
        ) -> None:
            del filepath, weights_only, storage_options
            raise OSError("local storage unavailable")

    records: list[dict[str, object]] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    callback = ModelCheckpoint(async_save=False)
    with (
        patch(
            "anytrain.lightning.callback.checkpoint.audit_event",
            side_effect=capture,
        ),
        pytest.raises(OSError, match="local storage unavailable"),
    ):
        callback._save_checkpoint(
            cast(Any, FailingTrainer()),
            str(tmp_path / "sync-failed.ckpt"),
        )

    _assert_checkpoint_spans_pair(records)
    events = {record["event"] for record in records}
    assert "checkpoint_serialize_error" in events
    assert "checkpoint_error" in events
    assert "checkpoint_done" not in events


def test_checkpoint_audit_uses_trainer_process_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    monkeypatch.setenv("ANYTRAIN_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("ANYTRAIN_AUDIT_FLUSH_INTERVAL", "0")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    audit_close()

    trainer = _AuditCheckpointTrainer(payload=b"ranked")
    trainer.global_rank = 5
    trainer.world_size = 8
    trainer.strategy = SimpleNamespace(root_device="cuda:1")
    callback = ModelCheckpoint(async_save=False)
    callback._save_checkpoint(cast(Any, trainer), str(tmp_path / "ranked.ckpt"))
    audit_close()

    path = tmp_path / "rank-5-worker-main.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records
    assert {record["rank"] for record in records} == {5}
    assert {record["world_size"] for record in records} == {8}
    assert {record["device"] for record in records} == {"cuda:1"}


def test_overlapping_checkpoint_triggers_are_thread_local() -> None:
    from anytrain.lightning import ModelCheckpoint

    callback = ModelCheckpoint(async_save=False)
    trainer = cast(Any, _AuditCheckpointTrainer())
    entered = Barrier(2)
    triggers: list[str] = []

    def capture(trigger: str) -> None:
        with callback._checkpoint_trigger(trigger):
            entered.wait(timeout=5)
            audit = callback._begin_checkpoint_audit(trainer, f"{trigger}.ckpt")
            triggers.append(str(audit.fields["trigger"]))

    threads = [
        Thread(target=capture, args=(trigger,)) for trigger in ("validation_end", "train_end")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert sorted(triggers) == ["train_end", "validation_end"]


def test_async_checkpoint_audit_pairs_copy_wait_publication_and_callback(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []
    copy_started = Event()
    release_copy = Event()
    wait_started = Event()
    wait_errors: list[BaseException] = []
    callback = ModelCheckpoint(async_save=True)
    original_copy = callback._copy_to_target

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})
        if event == "checkpoint_wait_start":
            wait_started.set()

    def blocked_copy(source: Path, target: Path) -> None:
        copy_started.set()
        if not release_copy.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint copy")
        original_copy(source, target)

    def wait_for_checkpoint() -> None:
        try:
            callback.wait_async_saves()
        except BaseException as error:
            wait_errors.append(error)

    target = tmp_path / "async.ckpt"
    wait_thread = Thread(target=wait_for_checkpoint)
    try:
        with (
            patch(
                "anytrain.lightning.callback.checkpoint.audit_event",
                side_effect=capture,
            ),
            patch.object(callback, "_copy_to_target", side_effect=blocked_copy),
        ):
            callback._save_checkpoint(
                cast(Any, _AuditCheckpointTrainer(payload=b"async")),
                str(target),
            )
            assert copy_started.wait(timeout=2)
            wait_thread.start()
            assert wait_started.wait(timeout=2)
            release_copy.set()
            wait_thread.join(timeout=5)

        assert not wait_thread.is_alive()
        assert wait_errors == []
        assert target.read_bytes() == b"async"
        _assert_checkpoint_spans_pair(records)
        events = {record["event"] for record in records}
        assert {
            "checkpoint_submit_start",
            "checkpoint_submit_done",
            "checkpoint_serialize_start",
            "checkpoint_serialize_done",
            "checkpoint_copy_start",
            "checkpoint_copy_done",
            "checkpoint_wait_start",
            "checkpoint_wait_done",
            "checkpoint_publication_start",
            "checkpoint_publication_done",
            "checkpoint_callback_start",
            "checkpoint_callback_done",
            "checkpoint_start",
            "checkpoint_done",
        } <= events
        publication_records = [
            record
            for record in records
            if record["event"]
            in {
                "checkpoint_publication_start",
                "checkpoint_publication_done",
            }
        ]
        assert publication_records
        assert all(record["execution_scope"] == "background" for record in publication_records)
        assert all(record["blocks_train_loop"] is False for record in publication_records)
    finally:
        release_copy.set()
        if wait_thread.is_alive():
            wait_thread.join(timeout=5)
        callback._close_async_storage()


def test_overlapping_async_checkpoint_saves_keep_audit_bindings(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []
    save_barrier = Barrier(2)
    first_serialized = Event()
    first_save_returned = Event()
    save_errors: list[BaseException] = []

    class OverlappingTrainer(_AuditCheckpointTrainer):
        def __init__(self, name: str, payload: bytes) -> None:
            super().__init__(payload=payload)
            self.name = name

        def save_checkpoint(
            self,
            filepath: str | Path,
            weights_only: bool | None = None,
            storage_options: object | None = None,
        ) -> None:
            del weights_only, storage_options
            save_barrier.wait(timeout=5)
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.name == "first":
                path.write_bytes(self.payload)
                first_serialized.set()
                return
            if not first_serialized.wait(timeout=5):
                raise TimeoutError("first checkpoint did not serialize")
            if not first_save_returned.wait(timeout=5):
                raise TimeoutError("first checkpoint did not return")
            path.write_bytes(self.payload)

    def save(trainer: OverlappingTrainer, target: Path) -> None:
        try:
            callback._save_checkpoint(cast(Any, trainer), str(target))
        except BaseException as error:
            save_errors.append(error)
        finally:
            if trainer.name == "first":
                first_save_returned.set()

    callback = ModelCheckpoint(async_save=True, max_pending_async_saves=2)
    first_target = tmp_path / "first.ckpt"
    second_target = tmp_path / "second.ckpt"
    first_trainer = OverlappingTrainer("first", b"first")
    second_trainer = OverlappingTrainer("second", b"second")
    first_thread = Thread(target=save, args=(first_trainer, first_target))
    second_thread = Thread(target=save, args=(second_trainer, second_target))

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    try:
        with patch(
            "anytrain.lightning.callback.checkpoint.audit_event",
            side_effect=capture,
        ):
            first_thread.start()
            second_thread.start()
            first_thread.join(timeout=5)
            second_thread.join(timeout=5)
            callback.wait_async_saves()

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert save_errors == []
        assert first_target.read_bytes() == b"first"
        assert second_target.read_bytes() == b"second"

        starts = [record for record in records if record["event"] == "checkpoint_start"]
        assert len(starts) == 2
        operation_ids = {record["operation_id"] for record in starts}
        assert len(operation_ids) == 2
        for event in ("checkpoint_copy_start", "checkpoint_copy_done"):
            copies = [record for record in records if record["event"] == event]
            assert len(copies) == 2
            assert {record["operation_id"] for record in copies} == operation_ids
        assert {
            record["operation_id"] for record in records if record["event"] == "checkpoint_done"
        } == operation_ids
        _assert_checkpoint_spans_pair(records)
    finally:
        first_save_returned.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)
        callback._close_async_storage()


def test_async_checkpoint_copy_failure_closes_all_started_audit_spans(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    callback = ModelCheckpoint(async_save=True)
    target = tmp_path / "failed.ckpt"
    try:
        with (
            patch(
                "anytrain.lightning.callback.checkpoint.audit_event",
                side_effect=capture,
            ),
            patch.object(
                callback,
                "_copy_to_target",
                side_effect=OSError("storage unavailable"),
            ),
        ):
            callback._save_checkpoint(cast(Any, _AuditCheckpointTrainer()), str(target))
            with pytest.raises(RuntimeError, match="Asynchronous checkpoint storage failed"):
                callback.wait_async_saves()

        _assert_checkpoint_spans_pair(records)
        events = {record["event"] for record in records}
        assert "checkpoint_copy_error" in events
        assert "checkpoint_error" in events
        assert "checkpoint_done" not in events
        assert all(record.get("checkpoint_key") for record in records)
    finally:
        callback._close_async_storage()


def test_sync_checkpoint_callback_failure_is_not_reported_as_serialization(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    class FailingLogger:
        def after_save_checkpoint(self, checkpoint: object) -> None:
            del checkpoint
            raise RuntimeError("logger callback failed")

    records: list[dict[str, object]] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    trainer = _AuditCheckpointTrainer(payload=b"durable")
    trainer.loggers = [FailingLogger()]
    callback = ModelCheckpoint(async_save=False)
    target = tmp_path / "sync-callback-error.ckpt"
    with (
        patch(
            "anytrain.lightning.callback.checkpoint.audit_event",
            side_effect=capture,
        ),
        pytest.raises(RuntimeError, match="logger callback failed"),
    ):
        callback._save_checkpoint(cast(Any, trainer), str(target))

    assert target.read_bytes() == b"durable"
    _assert_checkpoint_spans_pair(records)
    events = {record["event"] for record in records}
    assert "checkpoint_serialize_done" in events
    assert "checkpoint_serialize_error" not in events
    assert "checkpoint_callback_error" in events
    assert "checkpoint_error" in events
    assert "checkpoint_done" not in events


def test_async_checkpoint_callback_failure_preserves_publication_success(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    class FailingLogger:
        def after_save_checkpoint(self, checkpoint: object) -> None:
            del checkpoint
            raise RuntimeError("async logger callback failed")

    records: list[dict[str, object]] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    trainer = _AuditCheckpointTrainer(payload=b"published")
    trainer.loggers = [FailingLogger()]
    callback = ModelCheckpoint(async_save=True)
    target = tmp_path / "async-callback-error.ckpt"
    try:
        with patch(
            "anytrain.lightning.callback.checkpoint.audit_event",
            side_effect=capture,
        ):
            callback._save_checkpoint(cast(Any, trainer), str(target))
            with pytest.raises(
                RuntimeError,
                match="Asynchronous checkpoint storage failed",
            ):
                callback.wait_async_saves()

        assert target.read_bytes() == b"published"
        _assert_checkpoint_spans_pair(records)
        events = {record["event"] for record in records}
        assert "checkpoint_copy_done" in events
        assert "checkpoint_publication_done" in events
        assert "checkpoint_publication_error" not in events
        assert "checkpoint_callback_error" in events
        assert "checkpoint_error" in events
        assert "checkpoint_done" not in events
    finally:
        callback._close_async_storage()


def test_reaped_prepared_copy_waits_for_last_publication_before_success(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})

    trainer = _AuditCheckpointTrainer(payload=b"prepared")
    callback = ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        save_top_k=-1,
        enable_version_counter=False,
        async_save=True,
    )
    persistent = tmp_path / "top.ckpt"
    last = tmp_path / "last.ckpt"
    original_copy = callback._copy_to_target

    def fail_last_publication(source: Path, target: Path) -> None:
        if target == last:
            raise OSError("last publication failed")
        original_copy(source, target)

    try:
        with (
            patch(
                "anytrain.lightning.callback.checkpoint.audit_event",
                side_effect=capture,
            ),
            patch.object(
                callback,
                "_copy_to_target",
                side_effect=fail_last_publication,
            ),
        ):
            callback.last_model_path = str(last)
            callback._prepared_last_paths = ("", str(last))
            callback._save_checkpoint(cast(Any, trainer), str(persistent))
            reused = callback._reused_last_save
            assert reused is not None
            prerequisite = reused[4]
            assert prerequisite is not None
            prerequisite.result(timeout=5)

            assert callback._collect_done_async_error() is None
            assert "checkpoint_done" not in {record["event"] for record in records}

            callback._save_last_checkpoint(cast(Any, trainer), {})
            with pytest.raises(
                RuntimeError,
                match="Asynchronous checkpoint storage failed",
            ):
                callback.wait_async_saves()

        assert persistent.read_bytes() == b"prepared"
        assert not last.exists()
        _assert_checkpoint_spans_pair(records)
        events = [record["event"] for record in records]
        assert "checkpoint_copy_done" in events
        assert "checkpoint_publication_error" in events
        assert events.count("checkpoint_error") == 1
        assert "checkpoint_done" not in events
    finally:
        callback._close_async_storage()


def test_reused_checkpoint_copy_reports_wait_before_it_is_released(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []
    copy_started = Event()
    release_copy = Event()
    wait_started = Event()
    wait_errors: list[BaseException] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})
        if event == "checkpoint_wait_start":
            wait_started.set()

    trainer = _AuditCheckpointTrainer(payload=b"prepared")
    callback = ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        save_top_k=-1,
        enable_version_counter=False,
        async_save=True,
    )
    persistent = tmp_path / "top.ckpt"
    last = tmp_path / "last.ckpt"
    original_copy = callback._copy_to_target

    def blocked_copy(source: Path, target: Path) -> None:
        if target == persistent:
            copy_started.set()
            if not release_copy.wait(timeout=5):
                raise TimeoutError("test did not release prepared checkpoint copy")
        original_copy(source, target)

    def wait_for_checkpoint() -> None:
        try:
            callback.wait_async_saves()
        except BaseException as error:
            wait_errors.append(error)

    wait_thread = Thread(target=wait_for_checkpoint)
    try:
        with (
            patch(
                "anytrain.lightning.callback.checkpoint.audit_event",
                side_effect=capture,
            ),
            patch.object(callback, "_copy_to_target", side_effect=blocked_copy),
        ):
            callback.last_model_path = str(last)
            callback._prepared_last_paths = ("", str(last))
            callback._save_checkpoint(cast(Any, trainer), str(persistent))
            assert copy_started.wait(timeout=2)
            callback._save_last_checkpoint(cast(Any, trainer), {})

            wait_thread.start()
            assert wait_started.wait(timeout=2)
            assert "checkpoint_done" not in {record["event"] for record in records}
            release_copy.set()
            wait_thread.join(timeout=5)

        assert not wait_thread.is_alive()
        assert wait_errors == []
        assert persistent.read_bytes() == b"prepared"
        assert last.read_bytes() == b"prepared"
        _assert_checkpoint_spans_pair(records)
        assert [record["event"] for record in records].count("checkpoint_done") == 1
    finally:
        release_copy.set()
        if wait_thread.is_alive():
            wait_thread.join(timeout=5)
        callback._close_async_storage()


def test_independent_last_publication_failure_pairs_wait_and_error_spans(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []
    publication_started = Event()
    release_publication = Event()
    wait_started = Event()
    wait_errors: list[BaseException] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})
        if event == "checkpoint_wait_start":
            wait_started.set()

    def fail_publication(
        prerequisite: object,
        persistent: Path,
        last: Path,
        link: bool,
    ) -> None:
        del prerequisite, persistent, last, link
        publication_started.set()
        if not release_publication.wait(timeout=5):
            raise TimeoutError("test did not release last publication")
        raise OSError("last publication failed")

    callback = ModelCheckpoint(async_save=True)
    trainer = cast(Any, _AuditCheckpointTrainer())
    persistent = tmp_path / "persistent.ckpt"
    persistent.write_bytes(b"persistent")
    last = tmp_path / "last.ckpt"

    def wait_for_checkpoint() -> None:
        try:
            callback.wait_async_saves()
        except BaseException as error:
            wait_errors.append(error)

    wait_thread = Thread(target=wait_for_checkpoint)
    try:
        with (
            patch(
                "anytrain.lightning.callback.checkpoint.audit_event",
                side_effect=capture,
            ),
            patch.object(
                callback,
                "_update_last_checkpoint_impl",
                side_effect=fail_publication,
            ),
        ):
            checkpoint_audit = callback._begin_checkpoint_audit(
                trainer,
                str(last),
                checkpoint_kind="last",
                reason="last_publication",
            )
            callback._submit_async_last_update(
                None,
                persistent,
                last,
                link=False,
                publication=(str(persistent), str(last), last),
                audit=checkpoint_audit,
            )
            callback._finish_checkpoint_submit(
                checkpoint_audit,
                success=True,
                publication_pending=True,
            )
            assert publication_started.wait(timeout=2)
            wait_thread.start()
            assert wait_started.wait(timeout=2)
            release_publication.set()
            wait_thread.join(timeout=5)

        assert not wait_thread.is_alive()
        assert len(wait_errors) == 1
        assert isinstance(wait_errors[0], RuntimeError)
        _assert_checkpoint_spans_pair(records)
        events = {record["event"] for record in records}
        assert "checkpoint_wait_done" in events
        assert "checkpoint_publication_error" in events
        assert "checkpoint_error" in events
        assert "checkpoint_done" not in events
    finally:
        release_publication.set()
        if wait_thread.is_alive():
            wait_thread.join(timeout=5)
        callback._close_async_storage()


def test_async_remove_failure_pairs_wait_remove_and_checkpoint_spans(
    tmp_path: Path,
) -> None:
    from anytrain.lightning import ModelCheckpoint

    records: list[dict[str, object]] = []
    remove_started = Event()
    release_remove = Event()
    wait_started = Event()
    wait_errors: list[BaseException] = []

    def capture(event: str, **fields: object) -> None:
        records.append({"event": event, **fields})
        if event == "checkpoint_wait_start":
            wait_started.set()

    def fail_remove(target: Path) -> None:
        del target
        remove_started.set()
        if not release_remove.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint removal")
        raise OSError("checkpoint removal failed")

    callback = ModelCheckpoint(async_save=True)
    trainer = cast(Any, _AuditCheckpointTrainer())
    target = tmp_path / "expired.ckpt"
    target.write_bytes(b"expired")

    def wait_for_checkpoint() -> None:
        try:
            callback.wait_async_saves()
        except BaseException as error:
            wait_errors.append(error)

    wait_thread = Thread(target=wait_for_checkpoint)
    try:
        with (
            patch(
                "anytrain.lightning.callback.checkpoint.audit_event",
                side_effect=capture,
            ),
            patch.object(callback, "_remove_target", side_effect=fail_remove),
        ):
            callback._remove_checkpoint(trainer, str(target))
            assert remove_started.wait(timeout=2)
            wait_thread.start()
            assert wait_started.wait(timeout=2)
            release_remove.set()
            wait_thread.join(timeout=5)

        assert not wait_thread.is_alive()
        assert len(wait_errors) == 1
        assert isinstance(wait_errors[0], RuntimeError)
        assert target.read_bytes() == b"expired"
        _assert_checkpoint_spans_pair(records)
        events = {record["event"] for record in records}
        assert "checkpoint_wait_done" in events
        assert "checkpoint_remove_error" in events
        assert "checkpoint_error" in events
        assert "checkpoint_done" not in events
        assert {
            record.get("checkpoint_kind")
            for record in records
            if str(record["event"]).startswith("checkpoint_")
        } == {"remove"}
    finally:
        release_remove.set()
        if wait_thread.is_alive():
            wait_thread.join(timeout=5)
        callback._close_async_storage()
