from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import Enum
from itertools import zip_longest
from typing import TypeVar

from typing_extensions import Self

T1 = TypeVar("T1")
T2 = TypeVar("T2")

__all__ = ["Self", "StrEnum", "strict_zip"]


class StrEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        return name.lower()


_MISSING = object()


def strict_zip(first: Iterable[T1], second: Iterable[T2]) -> Iterator[tuple[T1, T2]]:
    for left, right in zip_longest(first, second, fillvalue=_MISSING):
        if left is _MISSING or right is _MISSING:
            raise ValueError("zip() argument lengths differ")
        yield left, right
