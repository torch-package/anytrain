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
| R001 | Interface Debt | `anytrain.codec` registry | Codec capabilities and default loaders now live in per-wrapper descriptors; the central registry only indexes entries and enforces public errors. | New codec wrappers still need an `_registry.py` descriptor, but no shared loader branch needs to change. | P1 | resolved | Keep new codec registration descriptor-driven. | Evidence: `src/anytrain/codec/_registry.py`; `src/anytrain/codec/*/_registry.py`; `src/anytrain/codec/registry.py`. |
| R002 | Boundary Debt | Speech ASR evaluator | `WhisperASREvaluator` backend code embeds openai-whisper loading, short-batch decode optimization, option coercion, fallback logic, and text extraction in one evaluator path. | Upstream whisper API changes or new ASR backends will force edits in a large evaluator module instead of isolated backend adapters. | P2 | pending | Split the whisper backend into a stable transcriber interface plus private optimization adapter; keep evaluator composition thin. | Evidence: `src/anytrain/evaluator/speech/asr.py`. |
| R003 | Implementation Debt | Buffer compatibility shim | `anytrain._buffer.register_buffer` exists to bridge PyTorch versions and is imported by reusable modules, even though the preferred modern shape is direct `nn.Buffer` assignment. | Once the minimum Torch version moves past 2.5, the helper will keep legacy API shape alive and encourage new call sites to copy it. | P2 | pending | After the project raises the minimum Torch version, replace helper usage with direct `nn.Buffer` assignment and delete the shim. | Evidence: `pyproject.toml` currently allows `torch>=2.4`; `_buffer.py`; `codec/_module.py`; `loss/spectral/transform.py`. |
| R004 | Boundary Debt | Public package root | The empty root package is documented as an intentional strict non-export contract with submodule import examples. | Future root exports should be treated as an explicit API change rather than an ad hoc convenience. | P3 | resolved | Keep `anytrain.__all__ == []` unless package-root policy changes deliberately. | Evidence: `src/anytrain/__init__.py`; `docs/modules/package-root.md`; `README.md`. |
| R005 | Boundary Debt | Lightning callback protocols | Lightning callbacks repeatedly probe `trainer`, `strategy`, logger, and module capabilities with `getattr` instead of a small local protocol layer. | Callback behavior is flexible but hard to reason about, and missing capabilities can degrade silently in new Trainer/strategy combinations. | P2 | pending | Centralize Trainer/logger/strategy capability accessors with explicit fallback semantics and focused tests for DDP and no-logger cases. | Evidence: repeated probes in `lightning/callback/loss.py`, `lightning/callback/performance.py`, `lightning/experiment.py`, and `lightning/callback/grad.py`. |

## Item Notes

### R001 Codec registry should be data-driven

Resolved: callers still ask for capabilities or default frame/semantic-acoustic loaders by name, but capability declaration and lazy loader dispatch now live in one `CodecEntry` descriptor per codec wrapper.

Current policy:

1. Add `src/anytrain/codec/<name>/_registry.py`.
2. Export one `ENTRY = CodecEntry(...)` with `name`, `capabilities`, and matching lazy loader callables.
3. Import the entry in `src/anytrain/codec/registry.py` and add it to `_ENTRIES`.
4. Keep codec-specific optional imports inside descriptor loader functions.

Verification:

- Focused codec registry tests should continue to cover capabilities, lazy loaders, and stable public errors.

### R002 Whisper ASR backend needs a thinner evaluator boundary

Current behavior works for openai-whisper and includes a useful short-batch path. The design problem is that evaluator composition and backend-specific optimization live together. The target contract should let `WhisperASREvaluator` depend on a backend object with `transcribe(audio, sample_rate, **options)`, while the openai-whisper adapter owns private decode details.

Smallest safe path:

1. Add behavior tests for short single-sample fallback and batch decode gating using fakes.
2. Extract `_OpenAIWhisperBackend` into a focused module.
3. Move short-batch decode helpers behind one private adapter class or method group.
4. Keep public evaluator constructor options unchanged.

Acceptance:

- `python -m ruff check src examples`
- `python -m basedpyright`
- `python -m pytest`

### R003 Drop buffer shim after the Torch floor moves

Current behavior is justified by the current `torch>=2.4` dependency. The design problem is the long-term public shape: new modules import a helper named `register_buffer` rather than using modern `nn.Buffer` assignment. This should remain deferred until the dependency floor changes.

Smallest safe path:

1. Raise the Torch minimum version in a dedicated dependency-change commit.
2. Replace `register_buffer(self, name, tensor, persistent=...)` with `self.name = nn.Buffer(tensor, persistent=...)` at each call site.
3. Delete `anytrain._buffer` if no public callers remain.
4. Add a note in the migration docs if users imported the helper.

Acceptance:

- `python -m ruff check src examples`
- `python -m basedpyright`
- `python -m pytest`

### R004 Make the empty root package a deliberate contract

Resolved: `anytrain.__all__` remains empty by policy, and users import from stable submodules. Discoverability lives in `docs/modules/package-root.md` and README examples rather than root-level aliases.

Current policy:

1. Keep `anytrain.__all__ == []`.
2. Import from explicit submodules such as `anytrain.lightning`, `anytrain.loss`, and `anytrain.optim`.
3. Treat any future package-root export as a deliberate public API change with regression tests.

Verification:

- Package-root docs and README continue to show explicit submodule imports.

### R005 Centralize Lightning capability probing

Current behavior works across several Lightning versions and strategy configurations. The design problem is that each callback decides its own `getattr` fallback policy. The target contract should keep Lightning optional flexibility but make missing-capability behavior visible and consistent.

Smallest safe path:

1. Inventory Trainer/logger/strategy probes in callbacks.
2. Add a private `lightning/_capability.py` module with typed helpers for global step, rank-zero, logger experiment, reduce, and barrier.
3. Replace probes in one callback first, preserving behavior.
4. Extend tests to no-logger, single-process, and fake-strategy cases.

Acceptance:

- `python -m ruff check src examples`
- `python -m basedpyright`
- `python -m pytest`

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
