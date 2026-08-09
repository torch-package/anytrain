from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import StrEnum
from itertools import zip_longest
from typing import Self

__all__ = ["Self", "StrEnum", "strict_zip"]

_MISSING = object()


def strict_zip[T1, T2](first: Iterable[T1], second: Iterable[T2]) -> Iterator[tuple[T1, T2]]:
    for left, right in zip_longest(first, second, fillvalue=_MISSING):
        if left is _MISSING or right is _MISSING:
            raise ValueError("zip() argument lengths differ")
        yield left, right
