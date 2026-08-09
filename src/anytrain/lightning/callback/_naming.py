from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

ValueT = TypeVar("ValueT")


def join_tag(*parts: object) -> str:
    """Join logger tag parts with / while preserving dot-style names."""

    cleaned = [str(part).strip("/") for part in parts if str(part).strip("/")]
    return "/".join(cleaned)


def validate_tag_part(value: object, name: str = "tag") -> str:
    """Normalize and validate one logger tag component."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    text = value.strip("/")
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def unique_tag_items[ValueT](
    items: Iterable[tuple[object, ValueT]],
    *,
    name: str = "monitor name",
) -> list[tuple[str, ValueT]]:
    """Normalize named values and reject collisions introduced by normalization."""

    result: list[tuple[str, ValueT]] = []
    source_by_name: dict[str, object] = {}
    for source, value in items:
        normalized = validate_tag_part(source, name)
        if normalized in source_by_name:
            previous = source_by_name[normalized]
            raise ValueError(
                f"{name} values {previous!r} and {source!r} normalize to "
                f"the same tag {normalized!r}."
            )
        source_by_name[normalized] = source
        result.append((normalized, value))
    return result


__all__ = [
    "join_tag",
    "unique_tag_items",
    "validate_tag_part",
]
