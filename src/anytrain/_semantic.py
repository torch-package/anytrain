from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


def semantic_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        semantic_mapping(payload, name="semantic payload"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def semantic_id(prefix: str, payload: Mapping[str, object]) -> str:
    _non_empty_string(prefix, name="semantic identity prefix")
    return f"{prefix}:{semantic_hash(payload)}"


def semantic_mapping(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    output: dict[str, object] = {}
    for key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
        _non_empty_string(key, name=f"{name} key")
        output[key] = semantic_value(item, name=f"{name}[{key!r}]")
    return output


def semantic_options(
    value: tuple[tuple[str, object], ...],
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of key/value pairs.")
    output: dict[str, object] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{name}[{index}] must be a key/value tuple.")
        key, item = entry
        _non_empty_string(key, name=f"{name}[{index}] key")
        if key in output:
            raise ValueError(f"{name} contains duplicate key {key!r}.")
        output[key] = semantic_value(item, name=f"{name}[{key!r}]")
    return output


def semantic_value(value: object, *, name: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value
    if isinstance(value, Mapping):
        return semantic_mapping(value, name=name)
    if isinstance(value, (tuple, list)):
        return [
            semantic_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{name} must contain only JSON scalar, mapping, list, or tuple values."
    )


def _non_empty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


__all__ = ["semantic_hash", "semantic_id", "semantic_mapping", "semantic_options"]
