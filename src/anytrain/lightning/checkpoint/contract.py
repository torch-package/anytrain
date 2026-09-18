"""State-dict schemas and compatible shared contract exports."""
from __future__ import annotations

from typing import TypedDict

from torch import Tensor, nn

from anytrain._contract import (
    Difference as Difference,
)
from anytrain._contract import (
    canonical_sha256 as canonical_sha256,
)
from anytrain._contract import (
    canonical_value as canonical_value,
)
from anytrain._contract import (
    first_difference as first_difference,
)
from anytrain._contract import (
    stable_json as stable_json,
)

STATE_DICT_SCHEMA_GRAMMAR = "torch-state-dict-schema-v1"
class StateDictSchema(TypedDict):
    grammar: str
    entry_count: int
    parameter_entries: int
    buffer_entries: int
    schema_sha256: str


def state_dict_schema(module: nn.Module) -> StateDictSchema:
    """Describe the registered tensor names and shapes in a module state dict."""
    if not isinstance(module, nn.Module):
        raise TypeError("state-dict schema requires a torch module.")
    state = module.state_dict(keep_vars=True)
    parameter_names = {
        name for name, _ in module.named_parameters(remove_duplicate=False)
    }
    buffer_names = {name for name, _ in module.named_buffers(remove_duplicate=False)}
    entries: list[dict[str, object]] = []
    parameter_entries = 0
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"state entry {name!r} must be a tensor for checkpoint schemas."
            )
        is_parameter = name in parameter_names
        is_buffer = name in buffer_names
        if is_parameter == is_buffer:
            raise TypeError(
                f"state entry {name!r} must resolve to one parameter or buffer."
            )
        kind = "parameter" if is_parameter else "buffer"
        if is_parameter:
            parameter_entries += 1
        entries.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(value.shape),
            }
        )
    return {
        "grammar": STATE_DICT_SCHEMA_GRAMMAR,
        "entry_count": len(entries),
        "parameter_entries": parameter_entries,
        "buffer_entries": len(entries) - parameter_entries,
        "schema_sha256": canonical_sha256(entries),
    }

__all__ = [
    "STATE_DICT_SCHEMA_GRAMMAR",
    "Difference",
    "StateDictSchema",
    "canonical_sha256",
    "canonical_value",
    "first_difference",
    "stable_json",
    "state_dict_schema",
]
