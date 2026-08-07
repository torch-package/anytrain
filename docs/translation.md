# Translation

`anytrain.translation` defines a dataset-independent universal translation
capability. It uses only strings and position-aligned sequences; it does not
import language enums, canonical samples, workspace paths, or synthesis plans.

The stable protocol is:

```python
from anytrain.translation import UniversalTranslator

translated = translator.translate(
    ["你好", "hello"],
    ["Chinese", "English"],
    ["English", "Chinese"],
)
```

All three sequences must be non-empty and have the same length. Results are a
plain `list[str]` in the same order. `config_hash()` identifies the model and
generation semantics used to produce durable data.

The optional Qwen backend is loaded separately:

```python
from anytrain.translation.qwen import QwenTranslator

translator = QwenTranslator.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device="cuda:0",
)
```

Install it with `anytrain[translation]`. The adapter performs deterministic
batched generation and returns only translated text. Dataset-specific prompt
selection, source scheduling, snapshot publication, and quality filtering stay
outside anytrain.
