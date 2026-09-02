from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "summarize_audit.py"
    spec = importlib.util.spec_from_file_location("summarize_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _span_records(
    name: str,
    *,
    start_ms: float,
    duration_ms: float,
    operation_id: str,
    component: str = "anytrain",
    run_id: str = "one-run",
    hostname: str | None = "node-a",
    pid: int = 10,
    rank: int = 0,
    worker: str = "main",
    execution_scope: str = "foreground",
    blocks_train_loop: bool = True,
    terminal_suffix: str = "done",
    **fields: object,
) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "component": component,
        "run_id": run_id,
        "hostname": hostname,
        "pid": pid,
        "rank": rank,
        "worker": worker,
        "operation_id": operation_id,
        **fields,
    }
    started_ns = int(start_ms * 1_000_000)
    return [
        {
            **common,
            "event": f"{name}_start",
            "time_ns": started_ns,
            "monotonic_ns": started_ns,
            "execution_scope": execution_scope,
            "blocks_train_loop": blocks_train_loop,
        },
        {
            **common,
            "event": f"{name}_{terminal_suffix}",
            "time_ns": started_ns + int(duration_ms * 1_000_000),
            "monotonic_ns": started_ns + int(duration_ms * 1_000_000),
            "latency_ms": duration_ms,
            "execution_scope": execution_scope,
            "blocks_train_loop": blocks_train_loop,
        },
    ]


def test_summary_correlates_blocking_spans_and_collective_arrival(tmp_path: Path) -> None:
    module = _module()
    records = [
        {
            "component": "anydataset",
            "event": "batch_fetch_start",
            "time_ns": 1_000_000_000,
            "monotonic_ns": 10,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "worker": "main",
            "global_step": 8,
            "operation_id": "fetch-8",
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        },
        {
            "component": "anydataset",
            "event": "batch_fetch_done",
            "time_ns": 1_500_000_000,
            "monotonic_ns": 500_000_010,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "worker": "main",
            "global_step": 8,
            "operation_id": "fetch-8",
            "latency_ms": 500.0,
        },
        {
            "component": "anytrain",
            "event": "collective_start",
            "time_ns": 2_000_000_000,
            "monotonic_ns": 20,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "worker": "main",
            "global_step": 8,
            "operation_id": "reduce-8",
            "collective_seq": 2,
            "collective": "strategy.reduce",
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        },
        {
            "component": "anytrain",
            "event": "collective_start",
            "time_ns": 2_300_000_000,
            "monotonic_ns": 22,
            "hostname": "node-b",
            "pid": 11,
            "rank": 1,
            "worker": "main",
            "global_step": 8,
            "operation_id": "reduce-8",
            "collective_seq": 2,
            "collective": "strategy.reduce",
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        },
        {
            "component": "anytrain",
            "event": "collective_done",
            "time_ns": 2_500_000_000,
            "monotonic_ns": 500_000_020,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "worker": "main",
            "global_step": 8,
            "operation_id": "reduce-8",
            "collective_seq": 2,
            "collective": "strategy.reduce",
            "latency_ms": 500.0,
        },
        {
            "component": "anytrain",
            "event": "collective_done",
            "time_ns": 2_500_000_000,
            "monotonic_ns": 200_000_022,
            "hostname": "node-b",
            "pid": 11,
            "rank": 1,
            "worker": "main",
            "global_step": 8,
            "operation_id": "reduce-8",
            "collective_seq": 2,
            "collective": "strategy.reduce",
            "latency_ms": 200.0,
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = module.summarize(module.load_records([path]))

    assert summary["top_blocking_spans"][0]["duration_ms"] == 500.0
    assert summary["collectives"][0]["arrival_skew_ms"] == 300.0
    assert summary["collectives"][0]["latest_arrival_rank"] is None
    assert summary["collectives"][0]["raw_latest_arrival_rank"] == 1
    assert summary["collectives"][0]["arrival_skew_reliable"] is False
    assert summary["collectives"][0]["latency_spread_ms"] == 300.0
    assert set(summary["likely_categories"][:2]) == {"dataloader", "collective"}


def test_summary_keeps_open_span_visible() -> None:
    module = _module()
    summary = module.summarize(
        [
            {
                "component": "anytrain",
                "event": "checkpoint_serialize_start",
                "time_ns": 1,
                "monotonic_ns": 2,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "operation_id": "checkpoint-1",
                "blocks_train_loop": True,
                "execution_scope": "foreground",
            }
        ]
    )

    assert summary["open_span_count"] == 1
    assert summary["open_spans"][0]["name"] == "checkpoint_serialize"
    rendered = module.render_text(summary)
    assert "open spans:" in rendered
    assert "checkpoint_serialize" in rendered


def test_summary_treats_loader_exhaustion_as_terminal() -> None:
    module = _module()
    summary = module.summarize(
        [
            {
                "component": "anydataset",
                "event": "batch_fetch_start",
                "time_ns": 1,
                "monotonic_ns": 1,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "worker": "main",
                "operation_id": "fetch-end",
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            },
            {
                "component": "anydataset",
                "event": "batch_fetch_exhausted",
                "time_ns": 2,
                "monotonic_ns": 2,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "worker": "main",
                "operation_id": "fetch-end",
                "latency_ms": 0.0,
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            },
        ]
    )

    assert summary["open_span_count"] == 0


def test_summary_pairs_lightning_end_events() -> None:
    module = _module()
    summary = module.summarize(
        [
            {
                "component": "anytrain",
                "event": "train_batch_start",
                "time_ns": 1,
                "monotonic_ns": 1,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "worker": "main",
                "operation_id": "train-1",
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            },
            {
                "component": "anytrain",
                "event": "train_batch_end",
                "time_ns": 3,
                "monotonic_ns": 3,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "worker": "main",
                "operation_id": "train-1",
                "latency_ms": 2.0,
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            },
        ]
    )

    assert summary["open_span_count"] == 0
    assert summary["top_blocking_spans"][0]["name"] == "train_batch"


def test_validation_batch_is_a_direct_validation_span_not_an_envelope() -> None:
    module = _module()
    summary = module.summarize(
        _span_records(
            "validation_batch",
            start_ms=0.0,
            duration_ms=250.0,
            operation_id="validation-1",
            terminal_suffix="end",
        )
    )

    assert summary["blocking_time_by_category_ms"] == {"validation": 250.0}
    assert summary["top_foreground_spans"][0]["name"] == "validation_batch"
    assert summary["top_foreground_envelopes"] == []
    assert summary["likely_categories"] == ["validation"]


def test_dataloader_iterator_summary_keeps_worker_configuration() -> None:
    module = _module()
    records = _span_records(
        "dataloader_iterator",
        start_ms=0.0,
        duration_ms=500.0,
        operation_id="loader-iterator-3",
        component="anydataset",
        dataloader="train",
        iterator_sequence=3,
        num_workers=8,
        persistent_workers=False,
        prefetch_factor=4,
        pin_memory=True,
        worker_start_method="fork",
        worker_restart_expected=True,
        timeout_seconds=0,
    )

    summary = module.summarize(records)
    span = summary["top_foreground_spans"][0]

    assert span["num_workers"] == 8
    assert span["persistent_workers"] is False
    assert span["prefetch_factor"] == 4
    assert span["worker_start_method"] == "fork"
    assert span["worker_restart_expected"] is True
    rendered = module.render_text(summary)
    assert "workers=8" in rendered
    assert "persistent=False" in rendered
    assert "prefetch=4" in rendered
    assert "start_method=fork" in rendered
    assert "restart_expected=True" in rendered


def test_summary_includes_collective_origin_and_plan_sync_breakdown() -> None:
    module = _module()
    records = [
        {
            "component": "anydataset",
            "event": "plan_sync_start",
            "time_ns": 1_000_000_000,
            "monotonic_ns": 1,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "world_size": 2,
            "worker": "main",
            "operation_id": "loader:epoch:3:chunk:0",
            "chunk": 0,
            "mode": "window",
            "window_id": 0,
            "dataloader": "s2tt",
            "local_count": 4,
            "planning_ms": 400.0,
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        },
        {
            "component": "anydataset",
            "event": "plan_sync_start",
            "time_ns": 1_250_000_000,
            "monotonic_ns": 2,
            "hostname": "node-b",
            "pid": 11,
            "rank": 1,
            "world_size": 2,
            "worker": "main",
            "operation_id": "loader:epoch:3:chunk:0",
            "chunk": 0,
            "mode": "window",
            "window_id": 0,
            "dataloader": "s2tt",
            "local_count": 2,
            "planning_ms": 50.0,
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        },
        {
            "component": "anydataset",
            "event": "plan_sync_done",
            "time_ns": 1_500_000_000,
            "monotonic_ns": 3,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "world_size": 2,
            "worker": "main",
            "operation_id": "loader:epoch:3:chunk:0",
            "chunk": 0,
            "counts": [4, 2],
            "local_count": 4,
            "kept_count": 2,
            "planning_ms": 400.0,
            "plan_sync_ms": 100.0,
            "count_skew": 2,
            "latency_ms": 100.0,
        },
        {
            "component": "anydataset",
            "event": "plan_sync_done",
            "time_ns": 1_500_000_000,
            "monotonic_ns": 4,
            "hostname": "node-b",
            "pid": 11,
            "rank": 1,
            "world_size": 2,
            "worker": "main",
            "operation_id": "loader:epoch:3:chunk:0",
            "chunk": 0,
            "counts": [4, 2],
            "local_count": 2,
            "kept_count": 2,
            "planning_ms": 50.0,
            "plan_sync_ms": 200.0,
            "count_skew": 2,
            "latency_ms": 200.0,
        },
        {
            "component": "anytrain",
            "event": "collective_start",
            "time_ns": 2_000_000_000,
            "monotonic_ns": 5,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "world_size": 1,
            "worker": "main",
            "operation_id": "vq-collective",
            "collective_seq": 9,
            "collective": "torch.distributed.all_reduce",
            "source": "EmbeddingVectorQuantizer._sync_ema_stats.counts",
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        },
        {
            "component": "anytrain",
            "event": "collective_done",
            "time_ns": 2_010_000_000,
            "monotonic_ns": 6,
            "hostname": "node-a",
            "pid": 10,
            "rank": 0,
            "world_size": 1,
            "worker": "main",
            "operation_id": "vq-collective",
            "collective_seq": 9,
            "collective": "torch.distributed.all_reduce",
            "source": "EmbeddingVectorQuantizer._sync_ema_stats.counts",
            "latency_ms": 10.0,
        },
    ]

    summary = module.summarize(records)
    plan_sync = next(
        item
        for item in summary["collectives"]
        if item["collective"] == "anydataset.plan_sync"
    )
    native = next(
        item
        for item in summary["collectives"]
        if item["collective"] == "torch.distributed.all_reduce"
    )

    assert plan_sync["rank_count"] == 2
    assert plan_sync["hostname_count"] == 2
    assert plan_sync["arrival_skew_ms"] == 250.0
    assert plan_sync["max_planning_ms"] == 400.0
    assert plan_sync["max_plan_sync_ms"] == 200.0
    assert plan_sync["max_count_skew"] == 2.0
    assert plan_sync["rank_details"] == [
        {
            "rank": 0,
            "terminal_event": "plan_sync_done",
            "stage": None,
            "local_count": 4,
            "counts": [4, 2],
            "kept_count": 2,
            "planning_ms": 400.0,
            "plan_sync_ms": 100.0,
            "latency_ms": 100.0,
        },
        {
            "rank": 1,
            "terminal_event": "plan_sync_done",
            "stage": None,
            "local_count": 2,
            "counts": [4, 2],
            "kept_count": 2,
            "planning_ms": 50.0,
            "plan_sync_ms": 200.0,
            "latency_ms": 200.0,
        },
    ]
    assert native["origin"] == "EmbeddingVectorQuantizer._sync_ema_stats.counts"
    rendered = module.render_text(summary)
    assert "planning_max=400.0ms" in rendered
    assert "origin=EmbeddingVectorQuantizer._sync_ema_stats.counts" in rendered
    assert "latency_spread=100.0ms" in rendered
    assert "local_counts={0: 4, 1: 2}" in rendered
    assert "counts=[4, 2]" in rendered
    assert "cross-host-wall-time-untrusted" in rendered


def test_plan_sync_failure_preserves_stage_and_rank_counts() -> None:
    module = _module()
    records = _span_records(
        "plan_sync",
        start_ms=0.0,
        duration_ms=25.0,
        operation_id="planner:chunk:0",
        component="anydataset",
        terminal_suffix="error",
        world_size=2,
        chunk=0,
        window=8,
        mode="window",
        stage="remote_planning",
        local_count=4,
        counts=[4, -1],
        planning_ms=10.0,
        plan_sync_ms=25.0,
        error_type="RuntimeError",
        error="remote planning failure",
    )

    summary = module.summarize(records)
    span = summary["top_foreground_spans"][0]
    collective = summary["collectives"][0]

    assert span["stage"] == "remote_planning"
    assert span["counts"] == [4, -1]
    assert span["local_count"] == 4
    assert span["chunk"] == 0
    assert span["window"] == 8
    assert span["mode"] == "window"
    assert collective["error_stages"] == ["remote_planning"]
    assert collective["rank_details"][0]["counts"] == [4, -1]
    rendered = module.render_text(summary)
    assert "stage=remote_planning" in rendered
    assert "counts=[4, -1]" in rendered


def test_summary_does_not_count_orphan_terminal_as_blocking() -> None:
    module = _module()
    summary = module.summarize(
        [
            {
                "component": "anytrain",
                "run_id": "retry-a",
                "event": "checkpoint_submit_done",
                "time_ns": 1_000,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "worker": "main",
                "operation_id": "checkpoint-1",
                "latency_ms": 420.0,
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            }
        ]
    )

    assert summary["orphan_terminal_count"] == 1
    assert summary["top_blocking_spans"] == []
    assert summary["blocking_time_by_category_ms"] == {}
    rendered = module.render_text(summary)
    assert "checkpoint_submit_done" in rendered
    assert "not counted as blockers" in rendered


def test_summary_ignores_dataset_identity_failure_point_event() -> None:
    module = _module()
    summary = module.summarize(
        [
            {
                "component": "anydataset",
                "run_id": "retry-a",
                "event": "dataset_identity_failure",
                "time_ns": 1_000,
                "hostname": "node-a",
                "pid": 10,
                "rank": 0,
                "worker": "main",
                "dataset": "train",
                "error_type": "ValueError",
                "error": "sample id unavailable",
            }
        ]
    )

    assert summary["span_count"] == 0
    assert summary["orphan_terminal_count"] == 0
    assert summary["top_blocking_spans"] == []


def test_summary_never_pairs_or_aggregates_different_runs() -> None:
    module = _module()
    common = {
        "component": "anytrain",
        "hostname": "node-a",
        "pid": 10,
        "worker": "main",
        "operation_id": "shared-operation",
        "global_step": 7,
        "execution_scope": "foreground",
        "blocks_train_loop": True,
    }
    records = [
        {
            **common,
            "run_id": "retry-a",
            "event": "collective_start",
            "time_ns": 1_000,
            "rank": 0,
            "world_size": 2,
            "collective_seq": 0,
            "collective": "strategy.reduce",
        },
        {
            **common,
            "run_id": "retry-b",
            "event": "collective_start",
            "time_ns": 2_000,
            "rank": 1,
            "world_size": 2,
            "collective_seq": 0,
            "collective": "strategy.reduce",
        },
        {
            **common,
            "run_id": "retry-b",
            "event": "collective_done",
            "time_ns": 3_000,
            "rank": 1,
            "world_size": 2,
            "collective_seq": 0,
            "collective": "strategy.reduce",
            "latency_ms": 1.0,
        },
    ]

    summary = module.summarize(records)

    assert summary["run_ids"] == ["retry-a", "retry-b"]
    assert summary["selected_run_id"] == "retry-b"
    assert summary["ignored_record_count"] == 1
    assert summary["open_span_count"] == 0
    assert len(summary["collectives"]) == 1
    assert summary["collectives"][0]["run_id"] == "retry-b"

    first_attempt = module.summarize(records, run_id="retry-a")
    assert first_attempt["selected_run_id"] == "retry-a"
    assert first_attempt["open_span_count"] == 1
    assert len(first_attempt["collectives"]) == 1
    assert first_attempt["collectives"][0]["run_id"] == "retry-a"


def test_likely_category_uses_worst_span_instead_of_rank_sum() -> None:
    module = _module()
    records: list[dict[str, object]] = []
    for rank in range(8):
        common = {
            "component": "anytrain",
            "run_id": "one-run",
            "hostname": "node-a",
            "pid": 100 + rank,
            "rank": rank,
            "worker": "main",
            "operation_id": "reduce-1",
            "collective_seq": 0,
            "collective": "strategy.reduce",
            "execution_scope": "foreground",
            "blocks_train_loop": True,
        }
        records.extend(
            (
                {**common, "event": "collective_start", "time_ns": 1_000 + rank},
                {
                    **common,
                    "event": "collective_done",
                    "time_ns": 2_000 + rank,
                    "latency_ms": 100.0,
                },
            )
        )
    records.extend(
        (
            {
                "component": "anydataset",
                "run_id": "one-run",
                "event": "batch_fetch_start",
                "time_ns": 3_000,
                "hostname": "node-a",
                "pid": 100,
                "rank": 0,
                "worker": "main",
                "operation_id": "fetch-1",
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            },
            {
                "component": "anydataset",
                "run_id": "one-run",
                "event": "batch_fetch_done",
                "time_ns": 4_000,
                "hostname": "node-a",
                "pid": 100,
                "rank": 0,
                "worker": "main",
                "operation_id": "fetch-1",
                "latency_ms": 500.0,
            },
        )
    )
    records.extend(
        (
            {
                "component": "anytrain",
                "run_id": "one-run",
                "event": "train_batch_start",
                "time_ns": 5_000,
                "hostname": "node-a",
                "pid": 100,
                "rank": 0,
                "worker": "main",
                "operation_id": "train-1",
                "execution_scope": "foreground",
                "blocks_train_loop": True,
            },
            {
                "component": "anytrain",
                "run_id": "one-run",
                "event": "train_batch_end",
                "time_ns": 6_000,
                "hostname": "node-a",
                "pid": 100,
                "rank": 0,
                "worker": "main",
                "operation_id": "train-1",
                "latency_ms": 400.0,
            },
        )
    )

    summary = module.summarize(records)

    assert summary["blocking_time_by_category_ms"]["collective"] == 100.0
    assert summary["inclusive_span_time_by_category_ms"]["collective"] == 800.0
    assert summary["max_blocking_span_by_category_ms"] == {
        "dataloader": 500.0,
        "train_envelope": 400.0,
        "collective": 100.0,
    }
    assert summary["likely_categories"][0] == "dataloader"
    assert summary["top_blocking_spans"][0]["name"] == "batch_fetch"


def test_likely_category_uses_single_worst_cause_not_retention_window_union() -> None:
    module = _module()
    records: list[dict[str, object]] = []
    for occurrence in range(128):
        records.extend(
            _span_records(
                "backward",
                start_ms=occurrence * 100.0,
                duration_ms=100.0,
                operation_id=f"backward-{occurrence}",
            )
        )
    records.extend(
        _span_records(
            "batch_fetch",
            start_ms=20_000.0,
            duration_ms=10_000.0,
            operation_id="fetch-stall",
            component="anydataset",
        )
    )

    summary = module.summarize(records)

    assert summary["blocking_time_by_category_ms"]["train"] == 12_800.0
    assert summary["blocking_time_by_category_ms"]["dataloader"] == 10_000.0
    assert summary["max_foreground_cause_span_by_category_ms"] == {
        "dataloader": 10_000.0,
        "train": 100.0,
    }
    assert summary["likely_categories"][0] == "dataloader"


def test_summary_unions_nested_spans_without_double_counting() -> None:
    module = _module()
    records = [
        *_span_records(
            "backward",
            start_ms=0.0,
            duration_ms=1_000.0,
            operation_id="backward-outer",
        ),
        *_span_records(
            "backward",
            start_ms=250.0,
            duration_ms=500.0,
            operation_id="backward-inner",
        ),
    ]

    summary = module.summarize(records)

    assert summary["blocking_time_by_category_ms"]["train"] == 1_000.0
    assert summary["inclusive_span_time_by_category_ms"]["train"] == 1_500.0
    assert summary["max_blocking_span_by_category_ms"]["train"] == 1_000.0


def test_train_envelope_reports_unattributed_residual() -> None:
    module = _module()
    records = [
        *_span_records(
            "train_batch",
            start_ms=0.0,
            duration_ms=10_000.0,
            operation_id="train-1",
            terminal_suffix="end",
        ),
        *_span_records(
            "training_step",
            start_ms=0.0,
            duration_ms=2_000.0,
            operation_id="forward-1",
        ),
        *_span_records(
            "backward",
            start_ms=2_000.0,
            duration_ms=3_000.0,
            operation_id="backward-1",
        ),
        *_span_records(
            "optimizer_step",
            start_ms=8_000.0,
            duration_ms=1_000.0,
            operation_id="optimizer-1",
        ),
    ]

    summary = module.summarize(records)

    envelope = summary["top_foreground_envelopes"][0]
    assert envelope["name"] == "train_batch"
    assert envelope["unattributed_ms"] == 4_000.0
    assert summary["max_unattributed_train_ms"] == 4_000.0
    assert summary["likely_categories"][0] == "train_unattributed"
    assert summary["max_foreground_cause_span_by_category_ms"]["train"] == 3_000.0
    assert "unattributed=4000.0ms" in module.render_text(summary)


def test_summary_separates_worker_and_background_spans() -> None:
    module = _module()
    records = [
        *_span_records(
            "dataset_read",
            start_ms=0.0,
            duration_ms=700.0,
            operation_id="read-1",
            component="anydataset",
            worker="3",
            execution_scope="worker",
            blocks_train_loop=False,
            sample_ids=["a", "b"],
        ),
        *_span_records(
            "checkpoint_copy",
            start_ms=1_000.0,
            duration_ms=900.0,
            operation_id="copy-1",
            execution_scope="background",
            blocks_train_loop=False,
            checkpoint_key="12:step:periodic:model.ckpt",
        ),
    ]

    summary = module.summarize(records)

    assert summary["top_worker_spans"][0]["name"] == "dataset_read"
    assert summary["top_worker_spans"][0]["sample_ids"] == ["a", "b"]
    assert summary["top_background_spans"][0]["name"] == "checkpoint_copy"
    assert (
        summary["top_background_spans"][0]["checkpoint_key"]
        == "12:step:periodic:model.ckpt"
    )


def test_summary_categorizes_loader_subphases_and_transfer() -> None:
    module = _module()
    records: list[dict[str, object]] = []
    for name, start_ms, duration_ms in (
        ("collate", 0.0, 100.0),
        ("raw_read", 100.0, 200.0),
        ("parse", 300.0, 300.0),
        ("batch_transfer", 600.0, 400.0),
    ):
        records.extend(
            _span_records(
                name,
                start_ms=start_ms,
                duration_ms=duration_ms,
                operation_id=name,
            )
        )

    summary = module.summarize(records)

    assert summary["blocking_time_by_category_ms"] == {
        "dataloader": 600.0,
        "transfer": 400.0,
    }
    assert {span["name"] for span in summary["top_foreground_spans"]} == {
        "collate",
        "raw_read",
        "parse",
        "batch_transfer",
    }


def test_partial_hostname_coverage_never_marks_arrival_skew_reliable() -> None:
    module = _module()
    records = [
        *_span_records(
            "collective",
            start_ms=0.0,
            duration_ms=100.0,
            operation_id="reduce-partial-host",
            rank=0,
            hostname="node-a",
            collective="strategy.reduce",
            collective_seq=0,
            world_size=2,
        ),
        *_span_records(
            "collective",
            start_ms=50.0,
            duration_ms=50.0,
            operation_id="reduce-partial-host",
            rank=1,
            pid=11,
            hostname=None,
            collective="strategy.reduce",
            collective_seq=0,
            world_size=2,
        ),
    ]

    collective = module.summarize(records)["collectives"][0]

    assert collective["arrival_skew_clock_scope"] == "partial-host"
    assert collective["arrival_skew_reliable"] is False
    assert collective["latest_arrival_rank"] is None
    assert collective["raw_latest_arrival_rank"] == 1


def test_collective_hostname_count_is_unique_hosts_not_rank_records() -> None:
    module = _module()
    records = [
        *_span_records(
            "collective",
            start_ms=0.0,
            duration_ms=100.0,
            operation_id="same-host-reduce",
            rank=0,
            hostname="node-a",
            collective="strategy.reduce",
            collective_seq=0,
            world_size=2,
        ),
        *_span_records(
            "collective",
            start_ms=25.0,
            duration_ms=75.0,
            operation_id="same-host-reduce",
            rank=1,
            pid=11,
            hostname="node-a",
            collective="strategy.reduce",
            collective_seq=0,
            world_size=2,
        ),
    ]

    collective = module.summarize(records)["collectives"][0]

    assert collective["rank_count"] == 2
    assert collective["hostname_count"] == 1
    assert collective["rank_coverage_complete"] is True
    assert collective["arrival_skew_reliable"] is True


def test_missing_collective_rank_makes_arrival_skew_unreliable() -> None:
    module = _module()
    records = _span_records(
        "collective",
        start_ms=0.0,
        duration_ms=100.0,
        operation_id="partial-rank-reduce",
        rank=0,
        hostname="node-a",
        collective="strategy.reduce",
        collective_seq=0,
        world_size=2,
    )

    collective = module.summarize(records)["collectives"][0]

    assert collective["rank_count"] == 1
    assert collective["world_size"] == 2
    assert collective["rank_coverage_complete"] is False
    assert collective["arrival_skew_reliable"] is False
    assert collective["latest_arrival_rank"] is None


def test_duplicate_collective_rank_start_makes_arrival_skew_unreliable() -> None:
    module = _module()
    records = [
        *_span_records(
            "collective",
            start_ms=0.0,
            duration_ms=100.0,
            operation_id="duplicate-rank-reduce",
            rank=0,
            pid=10,
            hostname="node-a",
            collective="strategy.reduce",
            collective_seq=0,
            world_size=1,
        ),
        *_span_records(
            "collective",
            start_ms=200.0,
            duration_ms=50.0,
            operation_id="duplicate-rank-reduce",
            rank=0,
            pid=11,
            hostname="node-a",
            collective="strategy.reduce",
            collective_seq=0,
            world_size=1,
        ),
    ]

    collective = module.summarize(records)["collectives"][0]

    assert collective["duplicate_rank_start_count"] == 1
    assert collective["rank_coverage_complete"] is True
    assert collective["arrival_skew_reliable"] is False
    assert collective["latest_arrival_rank"] is None


def test_cross_host_collectives_sort_by_rank_local_latency_not_wall_skew() -> None:
    module = _module()
    records: list[dict[str, object]] = []
    for rank, hostname, start_ms, duration_ms in (
        (0, "node-a", 0.0, 10.0),
        (1, "node-b", 100_000.0, 5.0),
    ):
        records.extend(
            _span_records(
                "collective",
                start_ms=start_ms,
                duration_ms=duration_ms,
                operation_id="huge-wall-skew",
                rank=rank,
                pid=10 + rank,
                hostname=hostname,
                collective="strategy.reduce",
                collective_seq=0,
                world_size=2,
            )
        )
    for rank, hostname, start_ms, duration_ms in (
        (0, "node-a", 200_000.0, 1_000.0),
        (1, "node-b", 200_001.0, 900.0),
    ):
        records.extend(
            _span_records(
                "collective",
                start_ms=start_ms,
                duration_ms=duration_ms,
                operation_id="slow-rank-local",
                rank=rank,
                pid=10 + rank,
                hostname=hostname,
                collective="strategy.reduce",
                collective_seq=1,
                world_size=2,
            )
        )

    collectives = module.summarize(records)["collectives"]

    assert collectives[0]["collective_seq"] == 1
    assert collectives[0]["diagnostic_sort_metric"] == "rank_latency"
    assert collectives[1]["arrival_skew_ms"] == 100_000.0


def test_cli_selects_requested_run_and_rejects_unknown_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    records = [
        *_span_records(
            "raw_read",
            start_ms=0.0,
            duration_ms=10.0,
            operation_id="read-a",
            run_id="retry-a",
        ),
        *_span_records(
            "raw_read",
            start_ms=100.0,
            duration_ms=20.0,
            operation_id="read-b",
            run_id="retry-b",
        ),
    ]
    path = tmp_path / "audit.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert module.main([str(path), "--run-id", "retry-a", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["selected_run_id"] == "retry-a"
    assert output["ignored_record_count"] == 2

    with pytest.raises(SystemExit, match="2"):
        module.main([str(path), "--run-id", "missing"])
    assert "was not found" in capsys.readouterr().err


def test_text_render_compacts_large_sample_context_but_json_keeps_it() -> None:
    module = _module()
    sample_ids = [f"sample-{index}" for index in range(20)]
    summary = module.summarize(
        _span_records(
            "batch_fetch",
            start_ms=0.0,
            duration_ms=10.0,
            operation_id="fetch-many",
            component="anydataset",
            sample_ids=sample_ids,
        )
    )

    assert summary["top_foreground_spans"][0]["sample_ids"] == sample_ids
    rendered = module.render_text(summary)
    assert "sample-7" in rendered
    assert "sample-8" not in rendered
    assert "total=20" in rendered
