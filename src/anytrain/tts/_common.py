from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def config_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def hashable_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: hashable_value(item) for key, item in sorted(value.items())}


def hashable_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): hashable_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [hashable_value(item) for item in value]
    return str(value)


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def validate_mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings.")
