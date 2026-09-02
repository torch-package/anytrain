#!/usr/bin/env python3
"""Summarize AnyTrain/AnyDataset JSONL audit records.

The command intentionally uses only the Python standard library so it can be
run from a training host without importing torch or Lightning:

    python scripts/summarize_audit.py /path/to/audit --top 12

It reports open spans, the longest foreground blockers, and rank arrival skew
for collectives.  A report is evidence, not a profiler: a collective can be
slow because another rank arrived late, so the arrival skew and the preceding
span should be inspected together.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

_ENVELOPE_SPAN_NAMES = frozenset(
    {
        "checkpoint",
        "checkpoint_submit",
        "train_batch",
    }
)


@dataclass(frozen=True)
class Span:
    key: tuple[Any, ...]
    name: str
    start: Mapping[str, Any]
    terminal: Mapping[str, Any] | None
    duration_ms: float | None
    orphan_terminal: bool = False

    @property
    def rank(self) -> Any:
        return self.start.get("rank")

    @property
    def step(self) -> Any:
        for field in ("global_step", "step_hint", "step"):
            value = self.start.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    @property
    def blocking(self) -> bool:
        return bool(self.start.get("blocks_train_loop")) and self.start.get("execution_scope") in {
            "foreground",
            "mixed",
        }


def load_records(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load valid JSON objects from files or directories, ordered by time_ns."""

    paths: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.update(path.rglob("*.jsonl"))
        elif path.is_file():
            paths.add(path)
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            print(f"warning: cannot read {path}: {error}", file=sys.stderr)
            continue
        for line_number, line in enumerate(lines, 1):
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                print(f"warning: ignored invalid JSON {path}:{line_number}", file=sys.stderr)
                continue
            if isinstance(value, Mapping):
                record = dict(value)
                record["_source"] = str(path)
                records.append(record)
    return sorted(records, key=_time_key)


def pair_spans(records: Iterable[Mapping[str, Any]]) -> list[Span]:
    """Pair ``*_start`` records with ``*_done``/``*_error`` records."""

    pending: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    spans: list[Span] = []
    for record in sorted(records, key=_time_key):
        event = record.get("event")
        if not isinstance(event, str):
            continue
        kind = _event_kind(event)
        if kind is None:
            continue
        operation_id = record.get("operation_id")
        if operation_id is None:
            operation_id = record.get("checkpoint_id")
        # Lifecycle callbacks without an id are still useful locally.  Keep
        # their rank/process/worker in the key to avoid cross-rank pairing.
        key = (
            record.get("component"),
            record.get("run_id"),
            record.get("hostname"),
            record.get("pid"),
            record.get("rank"),
            record.get("worker"),
            operation_id,
            _span_name(event),
        )
        if kind == "start":
            pending[key].append(record)
            continue
        starts = pending.get(key)
        start = starts.pop(0) if starts else None
        if start is None:
            # A bounded log may retain a terminal after dropping its start, and
            # some producers use a *_done event as a point marker. Keep the
            # evidence visible, but never count its reported latency as a
            # paired blocker.
            spans.append(
                Span(
                    key=key,
                    name=_span_name(event),
                    start=record,
                    terminal=record,
                    duration_ms=None,
                    orphan_terminal=True,
                )
            )
            continue
        duration = _duration_ms(start, record)
        spans.append(
            Span(
                key=key, name=_span_name(event), start=start, terminal=record, duration_ms=duration
            )
        )
    for key, starts in pending.items():
        for start in starts:
            spans.append(
                Span(key=key, name=str(key[-1]), start=start, terminal=None, duration_ms=None)
            )
    return spans


def summarize(
    records: list[dict[str, Any]],
    *,
    top: int = 10,
    run_id: str | None = None,
) -> dict[str, Any]:
    selected_records, selected_run_id, run_ids = _select_run(records, run_id)
    spans = pair_spans(selected_records)
    blocking = sorted(
        (span for span in spans if span.blocking and span.duration_ms is not None),
        key=lambda span: span.duration_ms or 0.0,
        reverse=True,
    )
    foreground_envelopes = [span for span in blocking if _is_envelope(span.name)]
    foreground_spans = [span for span in blocking if not _is_envelope(span.name)]
    envelope_unattributed = {
        id(span): _envelope_unattributed_ms(span, foreground_spans)
        for span in foreground_envelopes
        if span.name == "train_batch"
    }
    completed = [
        span for span in spans if span.duration_ms is not None and not span.orphan_terminal
    ]
    worker_spans = sorted(
        (span for span in completed if _span_scope(span) == "worker"),
        key=lambda span: span.duration_ms or 0.0,
        reverse=True,
    )
    background_spans = sorted(
        (span for span in completed if _span_scope(span) == "background"),
        key=lambda span: span.duration_ms or 0.0,
        reverse=True,
    )
    open_spans = [span for span in spans if span.terminal is None]
    orphan_terminals = [span for span in spans if span.orphan_terminal]
    collectives = _collective_summary(selected_records)
    inclusive_categories: dict[str, float] = defaultdict(float)
    category_maxima: dict[str, float] = defaultdict(float)
    for span in blocking:
        category = _category(span.name)
        duration = span.duration_ms or 0.0
        inclusive_categories[category] += duration
        category_maxima[category] = max(category_maxima[category], duration)
    cause_category_maxima: dict[str, float] = defaultdict(float)
    for span in foreground_spans:
        category = _category(span.name)
        cause_category_maxima[category] = max(
            cause_category_maxima[category], span.duration_ms or 0.0
        )
    max_unattributed_train_ms = max(envelope_unattributed.values(), default=0.0)
    likely_scores = dict(cause_category_maxima)
    if max_unattributed_train_ms > 0:
        likely_scores["train_unattributed"] = max_unattributed_train_ms
    if not likely_scores:
        for span in foreground_envelopes:
            category = _category(span.name)
            likely_scores[category] = max(likely_scores.get(category, 0.0), span.duration_ms or 0.0)
    union_categories = _blocking_union_by_category(blocking)
    ordered_categories = sorted(union_categories.items(), key=lambda item: item[1], reverse=True)
    ordered_inclusive = sorted(inclusive_categories.items(), key=lambda item: item[1], reverse=True)
    ordered_maxima = sorted(category_maxima.items(), key=lambda item: item[1], reverse=True)
    ordered_cause_maxima = sorted(
        cause_category_maxima.items(), key=lambda item: item[1], reverse=True
    )
    likely = sorted(likely_scores.items(), key=lambda item: item[1], reverse=True)
    legacy_blocking = foreground_spans or foreground_envelopes
    return {
        "input_record_count": len(records),
        "record_count": len(selected_records),
        "span_count": len(spans),
        "open_span_count": len(open_spans),
        "orphan_terminal_count": len(orphan_terminals),
        "open_spans": [_span_json(span) for span in open_spans[:top]],
        "orphan_terminals": [_span_json(span) for span in orphan_terminals[:top]],
        "top_blocking_spans": [_span_json(span) for span in legacy_blocking[:top]],
        "top_foreground_spans": [_span_json(span) for span in foreground_spans[:top]],
        "top_foreground_envelopes": [
            _span_json(
                span,
                unattributed_ms=envelope_unattributed.get(id(span)),
            )
            for span in foreground_envelopes[:top]
        ],
        "top_worker_spans": [_span_json(span) for span in worker_spans[:top]],
        "top_background_spans": [_span_json(span) for span in background_spans[:top]],
        "collectives": collectives[:top],
        "blocking_time_by_category_ms": dict(ordered_categories),
        "inclusive_span_time_by_category_ms": dict(ordered_inclusive),
        "max_blocking_span_by_category_ms": dict(ordered_maxima),
        "max_foreground_cause_span_by_category_ms": dict(ordered_cause_maxima),
        "max_unattributed_train_ms": max_unattributed_train_ms,
        "likely_category_max_span_ms": dict(likely),
        "likely_categories": [name for name, _ in likely],
        "run_ids": run_ids,
        "selected_run_id": selected_run_id,
        "ignored_record_count": len(records) - len(selected_records),
    }


def _select_run(
    records: list[dict[str, Any]],
    requested: str | None,
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    run_ids = sorted(
        {str(record["run_id"]) for record in records if record.get("run_id") is not None}
    )
    if requested is not None:
        if requested not in run_ids:
            raise ValueError(f"run_id {requested!r} was not found; available runs: {run_ids!r}")
        selected = requested
    elif run_ids:
        latest = max(
            (record for record in records if record.get("run_id") is not None),
            key=_time_key,
        )
        selected = str(latest["run_id"])
    else:
        selected = None
    if selected is None:
        return list(records), None, run_ids
    return (
        [
            record
            for record in records
            if record.get("run_id") is not None and str(record["run_id"]) == selected
        ],
        selected,
        run_ids,
    )


def _blocking_union_by_category(spans: Iterable[Span]) -> dict[str, float]:
    intervals: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    fallback: dict[tuple[Any, ...], float] = defaultdict(float)
    for span in spans:
        duration = span.duration_ms
        if duration is None:
            continue
        category = _category(span.name)
        axis, started = _span_axis(span)
        context = (
            span.start.get("run_id"),
            span.start.get("hostname"),
            span.start.get("pid"),
            span.start.get("rank"),
            span.start.get("worker"),
            category,
            axis,
        )
        if started is None:
            fallback[context] = max(fallback[context], duration)
            continue
        intervals[context].append((started, started + max(0.0, duration) * 1_000_000.0))
    maxima: dict[str, float] = defaultdict(float)
    for context in set(intervals) | set(fallback):
        category = str(context[-2])
        total = _interval_union_ms(intervals.get(context, ()))
        total = max(total, fallback.get(context, 0.0))
        maxima[category] = max(maxima[category], total)
    return dict(maxima)


def _span_axis(span: Span) -> tuple[str, float | None]:
    monotonic_ns = span.start.get("monotonic_ns")
    if _is_number(monotonic_ns):
        return "monotonic", float(monotonic_ns)
    time_ns = span.start.get("time_ns")
    if _is_number(time_ns):
        return "wall", float(time_ns)
    return "missing", None


def _interval_union_ms(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return total / 1_000_000.0


def _envelope_unattributed_ms(
    envelope: Span,
    foreground_spans: Iterable[Span],
) -> float:
    duration = envelope.duration_ms
    if duration is None:
        return 0.0
    axis, started = _span_axis(envelope)
    if started is None:
        return duration
    finished = started + max(0.0, duration) * 1_000_000.0
    context = _span_process_context(envelope)
    attributed: list[tuple[float, float]] = []
    for span in foreground_spans:
        if _span_process_context(span) != context:
            continue
        child_axis, child_started = _span_axis(span)
        child_duration = span.duration_ms
        if child_axis != axis or child_started is None or child_duration is None:
            continue
        child_finished = child_started + max(0.0, child_duration) * 1_000_000.0
        overlap_start = max(started, child_started)
        overlap_end = min(finished, child_finished)
        if overlap_start < overlap_end:
            attributed.append((overlap_start, overlap_end))
    return max(0.0, duration - _interval_union_ms(attributed))


def _span_process_context(span: Span) -> tuple[Any, ...]:
    return (
        span.start.get("run_id"),
        span.start.get("hostname"),
        span.start.get("pid"),
        span.start.get("rank"),
        span.start.get("worker"),
    )


def render_text(summary: Mapping[str, Any]) -> str:
    lines = [
        f"records={summary['record_count']}/{summary.get('input_record_count', summary['record_count'])} "
        f"spans={summary['span_count']} "
        f"open_spans={summary['open_span_count']} "
        f"orphan_terminals={summary.get('orphan_terminal_count', 0)}",
    ]
    run_ids = summary.get("run_ids") or []
    if run_ids:
        lines.append(
            f"selected run: {summary.get('selected_run_id')} "
            f"available={','.join(map(str, run_ids))}"
        )
        if summary.get("ignored_record_count"):
            lines.append("use --run-id to inspect another run in this audit directory.")
    likely = summary.get("likely_categories") or []
    if likely:
        lines.append("likely blocking categories: " + ", ".join(map(str, likely[:4])))
    _append_span_section(
        lines,
        "open spans:",
        summary.get("open_spans") or [],
    )
    _append_span_section(
        lines,
        "top foreground cause spans:",
        summary.get("top_foreground_spans") or [],
    )
    _append_span_section(
        lines,
        "top foreground envelopes:",
        summary.get("top_foreground_envelopes") or [],
    )
    _append_span_section(
        lines,
        "top worker spans:",
        summary.get("top_worker_spans") or [],
    )
    _append_span_section(
        lines,
        "top background spans:",
        summary.get("top_background_spans") or [],
    )
    orphan_terminals = summary.get("orphan_terminals") or []
    if orphan_terminals:
        lines.append("orphan terminals (not counted as blockers):")
        for span in orphan_terminals:
            reported = span.get("reported_latency_ms")
            reported_text = "" if not _is_number(reported) else f" reported={reported:.1f}ms"
            lines.append(
                f"  {span.get('terminal_event')} run={span.get('run_id')} "
                f"rank={span.get('rank')} step={span.get('step')}"
                f"{reported_text} log={span.get('source')}"
            )
    lines.append("collective arrival skew:")
    collectives = summary.get("collectives") or []
    if not collectives:
        lines.append("  (none)")
    for item in collectives:
        origin = item.get("origin") or item.get("dataloader")
        origin_text = "" if origin is None else f" origin={origin}"
        world_size = item.get("world_size")
        rank_count = item.get("rank_count")
        ranks = str(rank_count) if world_size is None else f"{rank_count}/{world_size}"
        clock_text = (
            ""
            if item.get("arrival_skew_reliable")
            else f" clock={item.get('arrival_skew_clock_scope')}-wall-time-untrusted"
        )
        latest_rank = (
            str(item.get("latest_arrival_rank"))
            if item.get("latest_arrival_rank_reliable")
            else f"untrusted(raw={item.get('raw_latest_arrival_rank')})"
        )
        lines.append(
            f"  {item.get('collective')} seq={item.get('collective_seq')} "
            f"run={item.get('run_id')} step={item.get('step')} ranks={ranks} "
            f"skew={item.get('arrival_skew_ms', 0.0):.1f} ms "
            f"latest_rank={latest_rank} "
            f"max_latency_rank={item.get('max_latency_rank')} "
            f"max={item.get('max_latency_ms', 0.0):.1f} ms"
            f"{_collective_details(item)}{origin_text}{clock_text}"
        )
    if summary.get("open_span_count"):
        lines.append(
            "open spans usually mean the bounded log dropped a terminal event or the process exited while blocked."
        )
    return "\n".join(lines)


def _append_span_section(
    lines: list[str],
    title: str,
    spans: Iterable[Mapping[str, Any]],
) -> None:
    values = list(spans)
    lines.append(title)
    if not values:
        lines.append("  (none)")
        return
    for span in values:
        duration = span.get("duration_ms")
        duration_text = "open" if duration is None else f"{duration:.1f} ms"
        details = _span_details(span)
        lines.append(
            f"  {duration_text:>12} {span.get('name')} "
            f"run={span.get('run_id')} rank={span.get('rank')} step={span.get('step')} "
            f"{details}log={span.get('source')}"
        )


def _collective_summary(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    terminals: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        event = record.get("event")
        if event in {"collective_start", "collective_done", "collective_error"}:
            collective = record.get("collective")
            collective_seq = record.get("collective_seq")
        elif event in {"plan_sync_start", "plan_sync_done", "plan_sync_error"}:
            collective = "anydataset.plan_sync"
            collective_seq = record.get("chunk")
        else:
            continue
        if collective is None:
            continue
        key = (
            record.get("run_id"),
            collective,
            collective_seq,
            record.get("operation_id"),
            record.get("global_step", record.get("step_hint")),
        )
        (starts if event.endswith("_start") else terminals)[key].append(record)
    result: list[dict[str, Any]] = []
    for key, rank_starts in starts.items():
        rank_starts_by_rank = {record.get("rank"): record for record in rank_starts}
        unique_rank_starts = tuple(rank_starts_by_rank.values())
        duplicate_rank_start_count = len(rank_starts) - len(rank_starts_by_rank)
        times = [
            float(value)
            for record in unique_rank_starts
            if _is_number(value := record.get("time_ns"))
        ]
        if not times:
            continue
        terminal_by_rank = {record.get("rank"): record for record in terminals.get(key, [])}
        latencies: dict[Any, float] = {}
        for rank, start in rank_starts_by_rank.items():
            value = _latency_from_records(start, terminal_by_rank.get(rank))
            if value is not None:
                latencies[rank] = value
        timed_ranks = {
            rank: float(value)
            for rank, record in rank_starts_by_rank.items()
            if _is_number(value := record.get("time_ns"))
        }
        raw_latest_rank = max(timed_ranks, key=timed_ranks.__getitem__)
        rank_terminals = tuple(terminal_by_rank.values())
        hostname_values = [
            str(record["hostname"])
            for record in unique_rank_starts
            if record.get("hostname") is not None
        ]
        hostnames = set(hostname_values)
        complete_hostnames = len(hostname_values) == len(unique_rank_starts)
        complete_arrival_times = len(times) == len(unique_rank_starts)
        world_sizes = [
            int(value)
            for record in unique_rank_starts
            if _is_number(value := record.get("world_size"))
        ]
        world_size_values = set(world_sizes)
        world_size_consistent = len(world_size_values) == 1
        world_size = max(world_sizes, default=None)
        expected_ranks = (
            set(range(world_size))
            if world_size_consistent and world_size is not None and world_size > 0
            else set()
        )
        observed_ranks = {
            rank
            for rank in rank_starts_by_rank
            if isinstance(rank, int) and not isinstance(rank, bool)
        }
        complete_rank_coverage = (
            bool(expected_ranks)
            and observed_ranks == expected_ranks
            and len(observed_ranks) == len(rank_starts_by_rank)
        )
        clock_scope = (
            "same-host"
            if complete_hostnames and len(hostnames) == 1
            else "cross-host"
            if complete_hostnames and len(hostnames) > 1
            else "partial-host"
            if hostnames
            else "unknown-host"
        )
        arrival_skew_reliable = (
            complete_hostnames
            and complete_arrival_times
            and complete_rank_coverage
            and duplicate_rank_start_count == 0
            and len(hostnames) == 1
        )
        error_terminals = tuple(
            record for record in rank_terminals if str(record.get("event", "")).endswith("_error")
        )
        result.append(
            {
                "run_id": key[0],
                "collective": key[1],
                "collective_seq": key[2],
                "step": key[4],
                "rank_count": len(rank_starts_by_rank),
                "duplicate_rank_start_count": duplicate_rank_start_count,
                "world_size": world_size,
                "world_size_consistent": world_size_consistent,
                "rank_coverage_complete": complete_rank_coverage,
                "missing_ranks": sorted(expected_ranks - observed_ranks),
                "unexpected_ranks": sorted(observed_ranks - expected_ranks),
                "arrival_skew_ms": (max(times) - min(times)) / 1_000_000.0,
                "arrival_skew_clock_scope": clock_scope,
                "arrival_skew_reliable": arrival_skew_reliable,
                "hostname_count": len(hostnames),
                "arrival_time_count": len(times),
                "latency_spread_ms": (
                    max(latencies.values()) - min(latencies.values()) if latencies else 0.0
                ),
                "latest_arrival_rank": raw_latest_rank if arrival_skew_reliable else None,
                "raw_latest_arrival_rank": raw_latest_rank,
                "latest_arrival_rank_reliable": arrival_skew_reliable,
                "max_latency_rank": max(latencies, key=lambda rank: latencies[rank])
                if latencies
                else None,
                "max_latency_ms": max(latencies.values(), default=0.0),
                "origin": _first_value(unique_rank_starts, "source"),
                "dataloader": _first_value(unique_rank_starts, "dataloader"),
                "mode": _first_value(unique_rank_starts, "mode"),
                "window_id": _first_value(unique_rank_starts, "window_id"),
                "max_planning_ms": _max_numeric(rank_terminals, "planning_ms"),
                "max_plan_sync_ms": _max_numeric(rank_terminals, "plan_sync_ms"),
                "max_count_skew": _max_numeric(rank_terminals, "count_skew"),
                "error_ranks": sorted(
                    {
                        record.get("rank")
                        for record in error_terminals
                        if record.get("rank") is not None
                    },
                    key=str,
                ),
                "error_types": sorted(
                    {
                        str(record["error_type"])
                        for record in error_terminals
                        if record.get("error_type") is not None
                    }
                ),
                "error_messages": sorted(
                    {
                        _compact_message(record["error"])
                        for record in error_terminals
                        if record.get("error") is not None
                    }
                ),
                "error_stages": sorted(
                    {
                        str(record["stage"])
                        for record in error_terminals
                        if record.get("stage") is not None
                    }
                ),
                "rank_details": [
                    {
                        "rank": rank,
                        "terminal_event": record.get("event"),
                        "stage": record.get("stage"),
                        "local_count": record.get("local_count"),
                        "counts": record.get("counts"),
                        "kept_count": record.get("kept_count"),
                        "planning_ms": record.get("planning_ms"),
                        "plan_sync_ms": record.get("plan_sync_ms"),
                        "latency_ms": latencies.get(rank),
                    }
                    for rank, record in sorted(
                        terminal_by_rank.items(), key=lambda item: str(item[0])
                    )
                ],
            }
        )
    for item in result:
        if item["arrival_skew_reliable"]:
            item["diagnostic_score_ms"] = max(item["arrival_skew_ms"], item["max_latency_ms"])
            item["diagnostic_sort_metric"] = "arrival_skew_or_latency"
        else:
            item["diagnostic_score_ms"] = max(item["latency_spread_ms"], item["max_latency_ms"])
            item["diagnostic_sort_metric"] = "rank_latency"
    return sorted(
        result,
        key=lambda item: (item["diagnostic_score_ms"], item["max_latency_ms"]),
        reverse=True,
    )


def _span_json(
    span: Span,
    *,
    unattributed_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "name": span.name,
        "run_id": span.start.get("run_id"),
        "duration_ms": span.duration_ms,
        "unattributed_ms": unattributed_ms,
        "rank": span.rank,
        "step": span.step,
        "source": span.start.get("_source"),
        "origin": span.start.get("source"),
        "execution_scope": _span_scope(span),
        "blocks_train_loop": span.start.get("blocks_train_loop"),
        "hostname": span.start.get("hostname"),
        "pid": span.start.get("pid"),
        "worker": span.start.get("worker"),
        "operation_id": span.start.get("operation_id"),
        "checkpoint_key": _terminal_value(span, "checkpoint_key"),
        "path": _terminal_value(span, "path"),
        "trigger": span.start.get("trigger"),
        "dataloader": span.start.get("dataloader"),
        "iterator_sequence": _terminal_value(span, "iterator_sequence"),
        "num_workers": _terminal_value(span, "num_workers"),
        "persistent_workers": _terminal_value(span, "persistent_workers"),
        "prefetch_factor": _terminal_value(span, "prefetch_factor"),
        "pin_memory": _terminal_value(span, "pin_memory"),
        "worker_start_method": _terminal_value(span, "worker_start_method"),
        "worker_restart_expected": _terminal_value(span, "worker_restart_expected"),
        "timeout_seconds": _terminal_value(span, "timeout_seconds"),
        "dataset": span.start.get("dataset"),
        "planner": span.start.get("planner"),
        "collective": span.start.get("collective"),
        "phase": span.start.get("phase"),
        "stage": _terminal_value(span, "stage"),
        "reason": span.start.get("reason"),
        "latency_scope": _terminal_value(span, "latency_scope"),
        "operation": _terminal_value(span, "operation"),
        "index": _terminal_value(span, "index"),
        "indexes": _terminal_value(span, "indexes"),
        "global_index": _terminal_value(span, "global_index"),
        "global_indexes": _terminal_value(span, "global_indexes"),
        "sample_ids": _terminal_value(span, "sample_ids"),
        "input_count": _terminal_value(span, "input_count"),
        "planning_ms": _terminal_value(span, "planning_ms"),
        "plan_sync_ms": _terminal_value(span, "plan_sync_ms"),
        "counts": _terminal_value(span, "counts"),
        "local_count": _terminal_value(span, "local_count"),
        "kept_count": _terminal_value(span, "kept_count"),
        "chunk": _terminal_value(span, "chunk"),
        "window": _terminal_value(span, "window"),
        "window_id": _terminal_value(span, "window_id"),
        "mode": _terminal_value(span, "mode"),
        "count_skew": _terminal_value(span, "count_skew"),
        "queue_depth": _terminal_value(span, "queue_depth"),
        "queue_capacity": _terminal_value(span, "queue_capacity"),
        "backpressure_wait_ms": _terminal_value(span, "backpressure_wait_ms"),
        "completion_wait_ms": _terminal_value(span, "completion_wait_ms"),
        "checkpoint_bytes": _terminal_value(span, "checkpoint_bytes"),
        "error_type": _terminal_value(span, "error_type"),
        "error": _terminal_value(span, "error"),
        "status": _terminal_value(span, "status"),
        "terminal_event": None if span.terminal is None else span.terminal.get("event"),
        "reported_latency_ms": None if span.terminal is None else span.terminal.get("latency_ms"),
        "orphan_terminal": span.orphan_terminal,
    }


def _terminal_value(span: Span, name: str) -> Any:
    if span.terminal is not None and span.terminal.get(name) is not None:
        return span.terminal.get(name)
    return span.start.get(name)


def _span_details(span: Mapping[str, Any]) -> str:
    details: list[str] = []
    for field, label in (
        ("origin", "origin"),
        ("dataloader", "loader"),
        ("dataset", "dataset"),
        ("trigger", "trigger"),
        ("phase", "phase"),
        ("stage", "stage"),
        ("mode", "mode"),
        ("status", "status"),
        ("operation", "operation"),
        ("latency_scope", "scope"),
        ("checkpoint_key", "checkpoint"),
    ):
        if span.get(field) is not None:
            details.append(f"{label}={span[field]}")
    for field, label in (
        ("planning_ms", "planning"),
        ("plan_sync_ms", "sync"),
        ("backpressure_wait_ms", "backpressure"),
        ("completion_wait_ms", "wait"),
        ("unattributed_ms", "unattributed"),
    ):
        value = span.get(field)
        if _is_number(value) and float(value) > 0:
            details.append(f"{label}={float(value):.1f}ms")
    if span.get("queue_depth") is not None:
        details.append(f"queue={span.get('queue_depth')}/{span.get('queue_capacity')}")
    for field, label in (
        ("local_count", "local_count"),
        ("kept_count", "kept_count"),
        ("chunk", "chunk"),
        ("window", "window"),
    ):
        if span.get(field) is not None:
            details.append(f"{label}={span[field]}")
    if span.get("counts") is not None:
        details.append(f"counts={_compact_preview(span['counts'])}")
    if span.get("worker") not in (None, "main"):
        details.append(f"worker={span['worker']}")
    for field, label in (
        ("num_workers", "workers"),
        ("persistent_workers", "persistent"),
        ("prefetch_factor", "prefetch"),
        ("worker_start_method", "start_method"),
        ("worker_restart_expected", "restart_expected"),
    ):
        if span.get(field) is not None:
            details.append(f"{label}={span[field]}")
    for field, label in (
        ("sample_ids", "samples"),
        ("global_indexes", "global_indexes"),
        ("indexes", "indexes"),
        ("global_index", "global_index"),
        ("index", "index"),
    ):
        if span.get(field) is not None:
            details.append(f"{label}={_compact_preview(span[field])}")
    if span.get("error_type") is not None:
        details.append(f"error={span['error_type']}")
    if span.get("error") is not None:
        details.append(f"message={_compact_message(span['error'])}")
    return "" if not details else " ".join(details) + " "


def _collective_details(item: Mapping[str, Any]) -> str:
    details: list[str] = []
    for field, label in (
        ("max_planning_ms", "planning_max"),
        ("max_plan_sync_ms", "sync_max"),
    ):
        value = item.get(field)
        if _is_number(value):
            details.append(f"{label}={float(value):.1f}ms")
    if _is_number(item.get("max_count_skew")):
        details.append(f"count_skew={int(item['max_count_skew'])}")
    if not item.get("arrival_skew_reliable") and _is_number(item.get("latency_spread_ms")):
        details.append(f"latency_spread={float(item['latency_spread_ms']):.1f}ms")
    if item.get("error_ranks"):
        details.append(f"error_ranks={item['error_ranks']}")
    if item.get("error_types"):
        details.append(f"errors={item['error_types']}")
    if item.get("error_messages"):
        details.append(f"messages={item['error_messages']}")
    if item.get("error_stages"):
        details.append(f"stages={item['error_stages']}")
    if item.get("missing_ranks"):
        details.append(f"missing_ranks={item['missing_ranks']}")
    if item.get("unexpected_ranks"):
        details.append(f"unexpected_ranks={item['unexpected_ranks']}")
    rank_details = item.get("rank_details") or []
    if rank_details:
        local_counts = {
            detail.get("rank"): detail.get("local_count")
            for detail in rank_details
            if detail.get("local_count") is not None
        }
        if local_counts:
            details.append(f"local_counts={local_counts}")
        count_vectors = {
            repr(detail.get("counts"))
            for detail in rank_details
            if detail.get("counts") is not None
        }
        if len(count_vectors) == 1:
            details.append(f"counts={next(iter(count_vectors))}")
        elif count_vectors:
            details.append(f"count_vectors={sorted(count_vectors)}")
    if item.get("duplicate_rank_start_count"):
        details.append(f"duplicate_rank_starts={item['duplicate_rank_start_count']}")
    return "" if not details else " " + " ".join(details)


def _compact_message(value: Any, *, limit: int = 160) -> str:
    return " ".join(str(value).split())[:limit]


def _compact_preview(value: Any, *, limit: int = 8) -> str:
    if isinstance(value, (list, tuple)):
        preview = repr(value[:limit])
        if len(value) > limit:
            return f"{preview}... total={len(value)}"
        return preview
    return str(value)


def _first_value(records: Iterable[Mapping[str, Any]], name: str) -> Any:
    return next((record[name] for record in records if record.get(name) is not None), None)


def _max_numeric(records: Iterable[Mapping[str, Any]], name: str) -> float | None:
    values = [float(value) for record in records if _is_number(value := record.get(name))]
    return max(values, default=None)


def _event_kind(event: str) -> str | None:
    if event.endswith("_start"):
        return "start"
    if (
        event.endswith("_done")
        or event.endswith("_error")
        or event.endswith("_exhausted")
        or event.endswith("_end")
    ):
        return "terminal"
    return None


def _span_name(event: str) -> str:
    for suffix in ("_start", "_done", "_error", "_exhausted", "_end"):
        if event.endswith(suffix):
            return event[: -len(suffix)]
    return event


def _duration_ms(start: Mapping[str, Any], terminal: Mapping[str, Any]) -> float | None:
    latency = terminal.get("latency_ms")
    if _is_number(latency):
        return float(latency)
    start_ns, terminal_ns = start.get("time_ns"), terminal.get("time_ns")
    if _is_number(start_ns) and _is_number(terminal_ns):
        return max(0.0, (float(terminal_ns) - float(start_ns)) / 1_000_000.0)
    return None


def _latency_from_records(
    start: Mapping[str, Any], terminal: Mapping[str, Any] | None
) -> float | None:
    return None if terminal is None else _duration_ms(start, terminal)


def _time_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    time_ns = record.get("time_ns")
    monotonic_ns = record.get("monotonic_ns")
    return (
        int(time_ns) if _is_number(time_ns) else 0,
        int(monotonic_ns) if _is_number(monotonic_ns) else 0,
        str(record.get("_source", "")),
    )


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _span_scope(span: Span) -> str:
    value = span.start.get("execution_scope")
    if isinstance(value, str) and value:
        return value
    worker = span.start.get("worker")
    if worker not in (None, "main"):
        return "worker"
    return "unknown"


def _is_envelope(name: str) -> bool:
    return name in _ENVELOPE_SPAN_NAMES


def _category(name: str) -> str:
    lowered = name.lower()
    if lowered == "train_batch":
        return "train_envelope"
    if lowered == "validation_batch":
        return "validation"
    if "checkpoint" in lowered:
        return "checkpoint"
    if "collective" in lowered or "strategy." in lowered:
        return "collective"
    if "batch_transfer" in lowered or "host_to_device" in lowered:
        return "transfer"
    if any(
        token in lowered
        for token in (
            "collate",
            "dataloader",
            "dataset",
            "fetch",
            "parse",
            "raw_read",
        )
    ):
        return "dataloader"
    if "plan" in lowered or "cost" in lowered:
        return "planner"
    if "validation" in lowered:
        return "validation"
    if (
        "backward" in lowered
        or "optimizer" in lowered
        or "training_step" in lowered
        or "train_batch" in lowered
    ):
        return "train"
    return "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSONL file(s) or audit directorie(s)")
    parser.add_argument("--top", type=int, default=10, help="number of entries per section")
    parser.add_argument("--run-id", help="inspect one run id (defaults to the latest run)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    records = load_records(args.paths)
    try:
        summary = summarize(records, top=args.top, run_id=args.run_id)
    except ValueError as error:
        parser.error(str(error))
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
