from __future__ import annotations

from collections.abc import Sequence

import torch.distributed as dist

from .._corpus import compute_on_root
from ..abc import EvaluatorABC
from ..types import MetricDict
from .normalization import TextNormalizationConfig, normalize_text_batch
from .profile import TextMetricProfile
from .scores import (
    bleu_signature,
    character_error_rate,
    chrf_signature,
    corpus_bleu_score,
    corpus_chrf_score,
    word_error_rate,
)

type _TextSemanticKey = tuple[TextNormalizationConfig, bool, str, int]
type _TextCorpusPayload = tuple[_TextSemanticKey, tuple[str, ...], tuple[str, ...]]


class TextComparisonEvaluator(EvaluatorABC):
    """Lightweight text comparison evaluator.

    BLEU and chrF/chrF++ are backed by sacreBLEU, and WER/CER are backed by jiwer. A
    small private fallback keeps the evaluator usable in minimal environments.
    """

    def __init__(
        self,
        *,
        strip: bool = True,
        collapse_whitespace: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = False,
        chinese: str | None = "simplified",
        bleu_smoothing: bool = True,
        bleu_tokenize: str = "none",
        chrf_word_order: int = 0,
    ) -> None:
        super().__init__()
        self.normalization = TextNormalizationConfig(
            strip=strip,
            collapse_whitespace=collapse_whitespace,
            remove_punctuation=remove_punctuation,
            lowercase=lowercase,
            chinese=chinese,
        )
        TextMetricProfile(self.normalization, bleu_smoothing, bleu_tokenize, chrf_word_order)
        self.bleu_smoothing = bleu_smoothing
        self.bleu_tokenize = bleu_tokenize
        self.chrf_word_order = chrf_word_order
        self._predictions: list[str] = []
        self._targets: list[str] = []

    @property
    def profile(self) -> TextMetricProfile:
        return TextMetricProfile(
            self.normalization, self.bleu_smoothing, self.bleu_tokenize, self.chrf_word_order
        )

    def evaluate(
        self,
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> MetricDict:
        predictions, targets = self._normalize_pair(prediction_text, target_text)
        return self._scores(predictions, targets)

    def update(
        self,
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> None:
        predictions, targets = self._normalize_pair(prediction_text, target_text)
        self._predictions.extend(predictions)
        self._targets.extend(targets)

    def compute(self) -> MetricDict:
        local = self._local_corpus()
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return self._compute_corpus(*local)
        return self._compute_distributed(local)

    def reset(self) -> None:
        self._predictions.clear()
        self._targets.clear()

    def metric_signatures(self) -> dict[str, str]:
        """Return reproducible one-reference BLEU and chrF profile signatures."""
        return {
            "bleu": bleu_signature(
                tokenize=self.bleu_tokenize,
                smooth=self.bleu_smoothing,
            ),
            "chrf": chrf_signature(word_order=self.chrf_word_order),
        }

    def _normalize_pair(
        self,
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        predictions = normalize_text_batch(
            prediction_text,
            name="prediction_text",
            config=self.normalization,
        )
        targets = normalize_text_batch(
            target_text,
            name="target_text",
            config=self.normalization,
        )
        if len(predictions) != len(targets):
            raise ValueError(
                "prediction_text and target_text must have the same batch length: "
                f"got {len(predictions)} and {len(targets)}."
            )
        if not predictions:
            raise ValueError("prediction_text and target_text must contain at least one item.")
        return predictions, targets

    def _scores(self, predictions: Sequence[str], targets: Sequence[str]) -> MetricDict:
        return {
            "bleu": corpus_bleu_score(
                predictions,
                targets,
                smooth=self.bleu_smoothing,
                tokenize=self.bleu_tokenize,
            ),
            "wer": word_error_rate(predictions, targets),
            "chrf": corpus_chrf_score(
                predictions,
                targets,
                word_order=self.chrf_word_order,
            ),
            "cer": character_error_rate(predictions, targets),
        }

    def _local_corpus(self) -> tuple[list[str], list[str]]:
        return list(self._predictions), list(self._targets)

    def _semantic_key(self) -> _TextSemanticKey:
        return (
            self.normalization,
            self.bleu_smoothing,
            self.bleu_tokenize,
            self.chrf_word_order,
        )

    def _compute_corpus(
        self,
        predictions: Sequence[str],
        targets: Sequence[str],
    ) -> MetricDict:
        if not predictions:
            raise ValueError("No text pairs have been recorded.")
        return self._scores(predictions, targets)

    def _compute_distributed(
        self,
        local: tuple[list[str], list[str]],
    ) -> MetricDict:
        """Gather text to rank zero and broadcast one corpus result."""

        payload: _TextCorpusPayload = (
            self._semantic_key(), tuple(local[0]), tuple(local[1]),
        )

        def compute(gathered: Sequence[_TextCorpusPayload | None]) -> dict[str, object]:
            predictions, targets = _merge_text_corpora(
                gathered, semantic_key=self._semantic_key(),
            )
            return {"metrics": self._compute_corpus(predictions, targets)}

        result = compute_on_root(
            payload, compute,
            source="TextComparisonEvaluator._compute_distributed",
            label="text",
            invalid_envelope="Distributed text compute returned an invalid result envelope.",
            local_item_count=len(local[0]),
        )
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("Distributed text compute returned invalid metrics.")
        return self._validate_metric_dict(metrics)


def _merge_text_corpora(
    gathered: Sequence[_TextCorpusPayload | None],
    *,
    semantic_key: _TextSemanticKey,
) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    targets: list[str] = []
    for rank, corpus in enumerate(gathered):
        if corpus is None:
            raise RuntimeError(
                "Distributed text corpus gathering returned no value for rank "
                f"{rank}."
            )
        if not isinstance(corpus, tuple) or len(corpus) != 3:
            raise RuntimeError("Distributed text corpus payload has an invalid shape.")
        rank_semantic_key, rank_predictions, rank_targets = corpus
        if rank_semantic_key != semantic_key:
            raise RuntimeError(
                "Distributed text evaluators have different semantic configurations."
            )
        if not isinstance(rank_predictions, tuple) or not isinstance(rank_targets, tuple):
            raise RuntimeError("Distributed text corpus payload has invalid text fields.")
        if len(rank_predictions) != len(rank_targets):
            raise RuntimeError("Distributed text corpus payloads are not aligned.")
        predictions.extend(rank_predictions)
        targets.extend(rank_targets)
    return predictions, targets
