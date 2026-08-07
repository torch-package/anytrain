# Refactor Debt

This file tracks runnable but poorly designed code in `third_party/anytrain`.

Keep this as an index of unfinished refactor debt. Move completed design notes into nearby `docs/` files or leave them to git history instead of keeping done items here.

## Scope

Use this file for:

- Interface debt: public APIs, module contracts, config surfaces, or caller expectations that should change.
- Boundary debt: logic living in the wrong layer, duplicated runtime constraints, or scripts carrying reusable rules.
- Implementation debt: ugly internal code that is stable enough to defer.

Do not use this file for experiment logs, benchmark results, private paths, remote training notes, checkpoints, or completed history.

## Priority

| Priority | Meaning |
|---|---|
| P0 | Currently blocks development, correctness, or result trust. |
| P1 | Likely to be touched soon or spreading into other modules. |
| P2 | Runnable and stable, but should be fixed when this area is next changed. |
| P3 | Mostly aesthetic; do not schedule proactively. |

## Debt Index

| ID | Type | Area | Problem | Risk | Priority | Status | Next Action | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | Interface Debt | `anytrain.audio.codec` registry | Codec capabilities and default loaders now live in per-wrapper descriptors; the central registry only indexes entries and enforces public errors. | New codec wrappers still need an `_registry.py` descriptor, but no shared loader branch needs to change. | P1 | resolved | Keep new codec registration descriptor-driven. | Evidence: `src/anytrain/audio/codec/_registry.py`; `src/anytrain/audio/codec/*/_registry.py`; `src/anytrain/audio/codec/registry.py`. |
| R002 | Boundary Debt | Speech ASR evaluator | `WhisperASREvaluator` now depends on a narrow transcriber protocol; OpenAI Whisper loading, short-batch decode optimization, option coercion, fallback logic, and text extraction live in a private adapter. | New ASR backend work should add or swap transcriber adapters without growing the public evaluator. | P2 | resolved | Keep evaluator composition thin and backend-specific decode behavior private. | Evidence: `src/anytrain/evaluator/speech/asr.py`; `src/anytrain/evaluator/speech/_whisper.py`. |
| R004 | Boundary Debt | Public package root | The empty root package is documented as an intentional strict non-export contract with submodule import examples. | Future root exports should be treated as an explicit API change rather than an ad hoc convenience. | P3 | resolved | Keep `anytrain.__all__ == []` unless package-root policy changes deliberately. | Evidence: `src/anytrain/__init__.py`; `docs/modules/package-root.md`; `README.md`. |
| R005 | Boundary Debt | Lightning callback protocols | Lightning Trainer/logger/strategy capability access now goes through a private helper with explicit fallback and error semantics. | New callbacks can still bypass the helper if not kept aligned during review. | P2 | resolved | Keep rank, step, logger, reduce, and barrier access routed through `lightning/_capability.py`. | Evidence: `src/anytrain/lightning/_capability.py`; `src/anytrain/lightning/callback/loss.py`; `src/anytrain/lightning/callback/performance.py`; `src/anytrain/lightning/experiment.py`; `tests/test_lightning_capability.py`. |

## Item Notes

### R001 Codec registry should be data-driven

Resolved: callers still ask for capabilities or default frame, semantic-acoustic, and semantic-global loaders by name, but capability declaration and lazy loader dispatch now live in one `CodecEntry` descriptor per codec wrapper.

Current policy:

1. Add `src/anytrain/audio/codec/<name>/_registry.py`.
2. Export one `ENTRY = CodecEntry(...)` with `name`, `capabilities`, and matching lazy loader callables.
3. Import the entry in `src/anytrain/audio/codec/registry.py` and add it to `_ENTRIES`.
4. Keep codec-specific optional imports inside descriptor loader functions.

Verification:

- Focused codec registry tests should continue to cover capabilities, lazy loaders, and stable public errors.

### R002 Whisper ASR backend needs a thinner evaluator boundary

Resolved: `WhisperASREvaluator` now owns public evaluation concerns only and delegates transcription to a narrow backend protocol. The OpenAI Whisper adapter owns model loading, wave normalization, short-batch decode gating, temperature fallback, and result text extraction.

Current policy:

1. Keep public constructor options and return normalization in `asr.py`.
2. Keep OpenAI Whisper-specific behavior in `_whisper.py`.
3. Add future ASR backends by satisfying the transcriber protocol instead of expanding the evaluator.
4. Keep backend modules private until there is a stable public backend API.

Verification:

- Focused speech evaluator tests should continue to cover single-sample fallback, batch decode gating, and public evaluator behavior.

### R004 Make the empty root package a deliberate contract

Resolved: `anytrain.__all__` remains empty by policy, and users import from stable submodules. Discoverability lives in `docs/modules/package-root.md` and README examples rather than root-level aliases.

Current policy:

1. Keep `anytrain.__all__ == []`.
2. Import from explicit submodules such as `anytrain.lightning`, `anytrain.loss`, and `anytrain.optim`.
3. Treat any future package-root export as a deliberate public API change with regression tests.

Verification:

- Package-root docs and README continue to show explicit submodule imports.

### R005 Centralize Lightning capability probing

Resolved: callbacks and helper modules now route Trainer/logger/strategy capability access through `lightning/_capability.py`. Missing distributed reduce/barrier support has explicit errors, while optional no-logger paths keep their prior graceful fallback.

Current policy:

1. Use `_capability.global_step`, `global_rank`, `world_size`, and `is_global_zero` for Trainer rank/step access.
2. Use `_capability.logger_experiment`, `loggers`, and `log_hyperparams` for logger access.
3. Use `_capability.strategy_reduce`, `strategy_reduce_or_none`, and `strategy_barrier` for distributed strategy access.
4. Keep helper behavior covered by no-logger and fake-strategy tests.

Verification:

- `tests/test_lightning_capability.py` covers logger fallback, missing reduce behavior, fake strategy reduce/barrier delegation, and hyperparameter logger filtering.

## Item Template

Use a separate `docs/refactor/R###-short-name.md` file only when an item needs more detail.

```md
# R### Short title

## Current Behavior

Describe what works today.

## Design Problem

Explain why the current shape is costly or risky.

## Why Not Now

State why this remains deferred instead of being fixed immediately.

## Target Contract

Define the desired public interface, module boundary, or invariant.

## Smallest Safe Path

1. Add or identify a behavior-preserving check.
2. Move one responsibility to the target owner.
3. Replace callers.
4. Remove the old path.

## Acceptance

- List the exact lint, type, test, smoke, or training command that proves the change.
```
