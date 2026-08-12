from __future__ import annotations

from collections.abc import Sequence


def non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def non_empty_strings(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    output = tuple(values)
    if not output:
        raise ValueError(f"{name} must not be empty.")
    for index, value in enumerate(output):
        non_empty_string(value, f"{name}[{index}]")
    return output


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


__all__ = ["non_empty_string", "non_empty_strings", "positive_int"]
