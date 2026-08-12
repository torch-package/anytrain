from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ._validation import non_empty_strings, positive_int


class UniversalTranslator(Protocol):
    """Batch text translation without dataset or task-specific types."""

    name: str

    def config_hash(self) -> str: ...

    def translate(
        self,
        texts: Sequence[str],
        source_languages: Sequence[str],
        target_languages: Sequence[str],
    ) -> list[str]: ...


def validate_translation_batch(
    texts: Sequence[str],
    source_languages: Sequence[str],
    target_languages: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate and freeze one position-aligned translation batch."""

    values = non_empty_strings(texts, "texts")
    sources = non_empty_strings(source_languages, "source_languages")
    targets = non_empty_strings(target_languages, "target_languages")
    if len(sources) != len(values) or len(targets) != len(values):
        raise ValueError(
            "texts, source_languages, and target_languages must have the same length."
        )
    return values, sources, targets


def validate_translation_outputs(values: Sequence[str], *, expected: int) -> list[str]:
    """Validate position-aligned translated text returned by a backend."""

    expected = positive_int(expected, "expected")
    outputs = list(non_empty_strings(values, "translation outputs"))
    if len(outputs) != expected:
        raise ValueError(
            "translation output count must match the input batch length: "
            f"{len(outputs)} != {expected}."
        )
    return outputs


__all__ = [
    "UniversalTranslator",
    "validate_translation_batch",
    "validate_translation_outputs",
]
