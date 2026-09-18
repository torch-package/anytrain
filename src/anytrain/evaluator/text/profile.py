from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from .normalization import TextNormalizationConfig
from .scores import bleu_signature, chrf_signature

if TYPE_CHECKING:
    from .evaluator import TextComparisonEvaluator


@dataclass(frozen=True)
class TextMetricProfile:
    """Complete text scoring semantics, reusable by domain evaluators."""

    normalization: TextNormalizationConfig = field(default_factory=TextNormalizationConfig)
    bleu_smoothing: bool = True
    bleu_tokenize: str = "none"
    chrf_word_order: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.normalization, TextNormalizationConfig):
            raise TypeError("normalization must be a TextNormalizationConfig.")
        if not isinstance(self.bleu_smoothing, bool):
            raise TypeError("bleu_smoothing must be a boolean.")
        bleu_signature(tokenize=self.bleu_tokenize, smooth=self.bleu_smoothing)
        chrf_signature(word_order=self.chrf_word_order)

    def semantic_payload(self) -> dict[str, object]:
        return asdict(self)

    def evaluator(self) -> TextComparisonEvaluator:
        from .evaluator import TextComparisonEvaluator

        return TextComparisonEvaluator(
            **asdict(self.normalization),
            bleu_smoothing=self.bleu_smoothing,
            bleu_tokenize=self.bleu_tokenize,
            chrf_word_order=self.chrf_word_order,
        )
