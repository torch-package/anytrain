# Stall Audit

`anytrain` and `anydataset` provide opt-in, task-agnostic JSONL audit events for
long-running distributed jobs. Set both repositories to the same directory and
run id so their records can be joined by `time_ns`, `global_step`, rank, and
operation id:

```bash
export ANY_AUDIT_RUN_ID="${JOB_ID}"
export ANYTRAIN_AUDIT_DIR="${RUN_DIR}/audit"
export ANYDATASET_AUDIT_DIR="${RUN_DIR}/audit"
export ANYTRAIN_AUDIT_KEEP_STEPS=128
export ANYDATASET_AUDIT_KEEP_STEPS=128
```

Add the callback to the Lightning trainer:

```python
from anytrain.lightning import TrainingAuditCallback

trainer = Trainer(
    callbacks=[
        *callbacks,
        TrainingAuditCallback(),
    ],
)
```

When using an explicit sink instead of `ANYTRAIN_AUDIT_DIR`, pass the same
`AuditSink` to both callbacks so synchronous and asynchronous checkpoint
phases join the training stream:

```python
sink = JsonlAuditSink(run_dir / "audit")
trainer = Trainer(
    callbacks=[
        TrainingAuditCallback(sink=sink),
        ModelCheckpoint(audit_sink=sink),
    ],
)
```

If Lightning ends a fit before a batch-end hook runs, the audit callback emits
`<phase>_end` with `status="fit_end"` and `stage="fit_end"`. These forced
terminals keep the span pairable while preserving that the phase did not reach
its normal hook.

Build loaders through anydataset's public `LoadingPlan` API. Its internal loader
emits audit records when the audit environment is enabled. Give every logical
loader a stable name that is identical across ranks and unique within the run:

```python
loader = (
    dataset.loading("cost_aware")
    .costs(costs, max_batch_memory=budget)
    .runtime(collate_fn=collate, num_workers=workers, audit_name="s2tt")
    .build()
)
```

The audit is disabled when the directory variables are unset. Each process uses
a fixed-capacity in-memory deque, so emitting a record is an O(1) append and
cannot scan the retained step window. Step retention is applied during the
asynchronous flush, and a worker crash can still leave the preceding events on
disk. `batch_report` on the callback or `LoadingPlan.runtime(...)` may add
compact task fields such as route, token count, or decode mode without putting
task semantics into either repository.

Environment-backed sinks are initialized once per process under a lock. This
matters when checkpoint and training threads emit their first event at the same
time: both records are retained by the same sink and closed together.

For explicit wiring, pass one `AuditSink` to both `TrainingAuditCallback(sink=...)`
and `ModelCheckpoint(audit_sink=...)`. The callbacks keep a small shared lifetime
lease, so callback ordering cannot close the sink before asynchronous checkpoint
copy/publication events are emitted. `JsonlAuditSink` preserves its directory
configuration across spawn-based workers; custom sinks are process-local and
should be rebound in each worker.

To inspect a live or completed run:

```bash
python scripts/summarize_audit.py "${RUN_DIR}/audit" --top 20
python scripts/summarize_audit.py "${RUN_DIR}/audit" --run-id "${ANY_AUDIT_RUN_ID}"
python scripts/summarize_audit.py "${RUN_DIR}/audit" --json > "${RUN_DIR}/audit-summary.json"
```

When a directory contains several retries, the summarizer selects the run id
with the latest record by default and reports how many records it ignored. Use
`--run-id` to inspect an older attempt; starts and terminals are never paired
across runs.

The text report keeps different execution contexts separate:

- `top foreground cause spans` are direct training-loop waits such as
  `batch_fetch`, `batch_transfer`, `training_step`, `backward`, collectives, and
  checkpoint phases.
- `top foreground envelopes` are enclosing `train_batch` or checkpoint
  lifecycles. A `train_batch` also reports `unattributed_ms`, the
  part not covered by finer foreground spans from the same process.
- `top worker spans` show reads, parsing, transforms, and collation performed by
  DataLoader workers, with sample/index context when the producer supplies it.
- `top background spans` show asynchronous checkpoint copy/publication and
  other work that does not directly block the training loop.

`blocking_time_by_category_ms` is an interval union per process and category,
then the maximum across ranks/processes. Nested phases and identical work on
eight ranks therefore count once on the critical rank. The explicitly
inclusive `inclusive_span_time_by_category_ms` field is also available when the
sum of all observed span durations is useful. Neither aggregate controls
`likely_categories`: likely causes are ordered by the longest single
non-envelope foreground span in each category (plus unexplained train-envelope
residual), so many normal short steps cannot outrank one long fetch stall.

Interpret the report in this order:

- An open `batch_fetch`, `dataset_read`, `collate`, `checkpoint`, or planner
  span means the process exited while that operation was active, or its
  terminal record was outside the bounded window.
- A large foreground `batch_fetch`/`dataset_read` span points to worker,
  decode, storage, or prefetch starvation. Join it with the worker id and
  sample/index fields in the raw JSONL.
- A large `training_step` span covers the callback-visible forward interval from
  train-batch start to the first backward hook. A large `train_batch`
  `unattributed_ms` means the enclosing stall is real but finer instrumentation
  has not yet named it.
- An `anydataset.plan_sync` entry reports rank coverage and arrival skew.
  `planning_max` is rank-local cost/planner work; `sync_max` is time inside the
  count synchronization. A large planning value identifies cold cost planning,
  while a large sync value with high arrival skew identifies a straggling rank.
  Arrival skew is marked reliable only with complete world-size coverage on one
  host. Failure entries retain the producer's `stage`, per-rank `local_count`,
  and gathered `counts`, so planning, count validation, and rank imbalance are
  distinguishable without reopening every JSONL file.
- A large `checkpoint_serialize` span is synchronous foreground serialization;
  `checkpoint_copy` and publication are background phases. `queue_depth`,
  `backpressure_wait_ms`, and `blocks_train_loop` distinguish a full async queue
  from a direct training-loop stall. Every phase carries a stable
  `checkpoint_key` (`step:trigger:kind:path`) so the same logical checkpoint can
  be aligned across ranks even though process-local operation ids differ.
- For a collective, compare `arrival_skew_ms` with `max_latency_ms`. A high
  arrival skew means the latest rank was delayed before entering the collective;
  inspect that rank's preceding DataLoader/planner/checkpoint span. Similar
  arrival times with a high collective latency points to the communication or
  backend itself. Wall-clock arrival skew is reliable only when all records are
  from the same host (or the hosts have an independently verified synchronized
  clock). The report marks cross-host wall-clock skew as untrusted and also
  exposes the clock-independent latency spread as supporting evidence.
- The `origin` field names the owner of a collective. Strategy collectives infer
  their caller while native paths identify ADT statistic sync, VQ EMA/dead-code
  sync, auto-cost calibration, or text evaluator corpus gathering explicitly.

The summarizer includes `run_id` in every pairing key. If several job attempts
share a directory, their starts, terminals, and collective ranks cannot be
joined into a false span. An orphan terminal is shown for context but is never
counted as blocking time because its matching start is absent from the bounded
log. The compatibility `top_blocking_spans` field contains direct cause spans
when available; enclosing spans remain visible in
`top_foreground_envelopes`.

Lightning calls batch transfer after a DataLoader fetch but before
`on_train_batch_start`, and the installed Callback API has no before/after
transfer hooks. Without task-side instrumentation, the interval from
`batch_fetch_done` to `train_batch_start` is therefore a useful gap but not a
named span. Instrument a custom module transfer hook when CPU-to-device copy,
pinning, or device-side preprocessing is a possible cause:

```python
from anytrain.lightning import audit_operation

def transfer_batch_to_device(self, batch, device, dataloader_idx):
    transfer = super().transfer_batch_to_device
    return audit_operation(
        "batch_transfer",
        lambda: transfer(batch, device, dataloader_idx),
        execution_scope="foreground",
        blocks_train_loop=True,
        stage="train" if self.training else "eval",
        dataloader_idx=dataloader_idx,
        target_device=str(device),
    )
```

The event schema includes `schema_version`, `component`, `run_id`, `hostname`,
`node_rank`, `local_rank`, `rank`, `world_size`, `pid`, `device`, and step hints.
These are intentionally infrastructure facts. Video decoding, ASR, UTMOS, and
task-loss meanings remain owned by the downstream project and can be attached
through `batch_report` or explicit `audit_event` fields.
