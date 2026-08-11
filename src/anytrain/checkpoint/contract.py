from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, NamedTuple, TypedDict

import torch
from torch import Tensor, nn

STATE_DICT_SCHEMA_GRAMMAR = "torch-state-dict-schema-v1"
_MISSING = "<missing>"


class Difference(NamedTuple):
    path: str
    actual: object
    expected: object


class StateDictSchema(TypedDict):
    grammar: str
    entry_count: int
    parameter_entries: int
    buffer_entries: int
    schema_sha256: str


def canonical_value(value: Any) -> Any:
    """Convert a contract value into deterministic JSON-safe data."""
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint contract mapping keys must be strings.")
        return {key: canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item) for item in value]
        return sorted(items, key=stable_json)
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint contract floats must be finite.")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported checkpoint contract value: {type(value).__name__}.")


def stable_json(value: object) -> str:
    """Serialize a supported value as deterministic compact JSON."""
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    """Hash the stable JSON representation of a supported value."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def first_difference(
    actual: object,
    expected: object,
    *,
    path: str = "value",
    key_order: Mapping[str, Sequence[str]] | None = None,
) -> Difference | None:
    """Return the first deterministic difference between canonical values."""
    if not isinstance(path, str) or not path:
        raise ValueError("difference path must be a non-empty string.")
    order = {} if key_order is None else key_order
    return _first_difference(
        canonical_value(actual),
        canonical_value(expected),
        path=path,
        key_order=order,
    )


def _first_difference(
    actual: Any,
    expected: Any,
    *,
    path: str,
    key_order: Mapping[str, Sequence[str]],
) -> Difference | None:
    if type(actual) is not type(expected):
        return Difference(path, actual, expected)
    if isinstance(actual, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        preferred = {
            key: index for index, key in enumerate(key_order.get(path, ()))
        }
        for key in sorted(
            actual_keys | expected_keys,
            key=lambda item: (preferred.get(item, len(preferred)), item),
        ):
            if key not in actual:
                return Difference(f"{path}.{key}", _MISSING, expected[key])
            if key not in expected:
                return Difference(f"{path}.{key}", actual[key], _MISSING)
            difference = _first_difference(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
                key_order=key_order,
            )
            if difference is not None:
                return difference
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return Difference(f"{path}.length", len(actual), len(expected))
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            difference = _first_difference(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                key_order=key_order,
            )
            if difference is not None:
                return difference
        return None
    if actual != expected:
        return Difference(path, actual, expected)
    return None


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
