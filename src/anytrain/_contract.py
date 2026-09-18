"""Deterministic contract values shared by modules and checkpoint adapters."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, NamedTuple

import torch

_MISSING = "<missing>"
class Difference(NamedTuple):
    path: str
    actual: object
    expected: object


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
