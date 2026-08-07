from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


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

    values = _strings("texts", texts)
    sources = _strings("source_languages", source_languages)
    targets = _strings("target_languages", target_languages)
    if len(sources) != len(values) or len(targets) != len(values):
        raise ValueError(
            "texts, source_languages, and target_languages must have the same length."
        )
    return values, sources, targets


def validate_translation_outputs(values: Sequence[str], *, expected: int) -> list[str]:
    """Validate position-aligned translated text returned by a backend."""

    outputs = list(_strings("translation outputs", values))
    if len(outputs) != expected:
        raise ValueError(
            "translation output count must match the input batch length: "
            f"{len(outputs)} != {expected}."
        )
    return outputs


def _strings(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    output = tuple(values)
    if not output:
        raise ValueError(f"{name} must not be empty.")
    for index, value in enumerate(output):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a string.")
        if not value.strip():
            raise ValueError(f"{name}[{index}] must not be empty.")
    return output


__all__ = [
    "UniversalTranslator",
    "validate_translation_batch",
    "validate_translation_outputs",
]
