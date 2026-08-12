# Translation

`anytrain.translation` defines a dataset-independent translation capability. It
uses strings and position-aligned sequences; it does not import dataset roles,
workspace paths, or materialization contracts.

The stable runtime protocol remains:

```python
from anytrain.translation import UniversalTranslator

translated = translator.translate(
    ["你好", "hello"],
    ["Chinese", "English"],
    ["English", "Chinese"],
)
```

All three sequences must be non-empty and have the same length. Results are a
plain `list[str]` in the same order.

## Ownership

- `protocol.py` owns the dataset-independent `UniversalTranslator` protocol and aligned
  input/output batch contract.
- `_validation.py` is the private owner for non-empty strings, string sequences, and positive
  integer controls shared by the protocol and concrete backends.
- `qwen.py` owns one complete pinned backend lifecycle: semantic spec, physical load policy,
  pickle-safe factory, tokenizer-only estimator, prompt/decoder protocol, and runtime adapter.

The Qwen lifecycle remains together because its semantic identity, prompt, generation options,
token planning, and decoder must change coherently. Splitting it only by file length would add
cross-file navigation without creating an independent contract owner.

## Pinned Qwen semantics

Qwen configuration is split into output semantics and physical loading policy:

```python
from anytrain.translation.qwen import (
    QwenTranslationLoadOptions,
    QwenTranslationSpec,
)

spec = QwenTranslationSpec(
    model="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    dtype="bfloat16",
    max_new_tokens=256,
)
load = QwenTranslationLoadOptions(
    cache_dir="storage/huggingface",
    local_files_only=True,
)
factory = spec.factory(load)
translator = factory("cuda:0")
```

`QwenTranslationSpec` is frozen and requires an explicit 40-character model commit. Its
semantic identity includes model, revision, dtype, deterministic generation
options, and versioned prompt/decoder protocols. `cache_dir`, offline policy,
device, batch size, and worker count are not semantic inputs.

`QwenTranslatorFactory` is a frozen, pickle-safe lazy factory. Constructing or
serializing it imports no Transformers backend and loads no tokenizer or model;
the selected worker loads one backend when it calls the factory. The resulting
`QwenTranslator.config_hash()` and `semantic_id()` are derived only from the
spec.

The public boundaries reject implicit coercion: model/device/cache strings must be non-empty,
token counts are strict positive integers, and boolean loading controls do not accept `0` or `1`.
At runtime the adapter validates callable tokenizer/model methods, rank-2 integer token ids,
batch alignment, prompt-prefix length, and non-empty decoded strings before returning results.

Generation is deterministic: `do_sample=False`, `use_cache=True`, and the
explicit `max_new_tokens` contract cannot be overridden through
`generation_options`. Unsupported prompt or decoder protocol versions fail at
spec construction instead of silently reusing current code under a new identity.

## Token planning

The same spec creates a tokenizer-only estimator:

```python
estimator = spec.token_estimator(load)

budget = estimator.planned_tokens(
    "需要翻译的文本",
    source_language="Chinese",
    target_language="English",
)
decoded = estimator.decoded_target_tokens("translated text")
```

`planned_tokens()` is the exact v1 prompt token count plus
`spec.max_new_tokens`. `decoded_target_tokens()` retokenizes decoded text as a
post-output work proxy; it is not the number of token ids emitted by
`generate()`. The estimator lazily loads only `AutoTokenizer`, never the model.

Install the optional backend with `anytrain[translation]`. Dataset-specific row
selection, language/role mapping, durable provider-id compatibility, source
scheduling, and quality filtering stay outside AnyTrain.
