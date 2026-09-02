from __future__ import annotations

from collections.abc import Sequence

import torch.distributed as dist

from ..abc import EvaluatorABC
from ..types import MetricDict
from .normalization import TextNormalizationConfig, normalize_text_batch
from .scores import corpus_bleu_score, corpus_chrf_score, word_error_rate


class TextComparisonEvaluator(EvaluatorABC):
    """Lightweight text comparison evaluator.

    BLEU and chrF are backed by sacreBLEU, and WER is backed by jiwer. A small
    private fallback keeps the evaluator usable in minimal environments.
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
    ) -> None:
        super().__init__()
        self.normalization = TextNormalizationConfig(
            strip=strip,
            collapse_whitespace=collapse_whitespace,
            remove_punctuation=remove_punctuation,
            lowercase=lowercase,
            chinese=chinese,
        )
        if not isinstance(bleu_smoothing, bool):
            raise TypeError("bleu_smoothing must be a boolean.")
        self.bleu_smoothing = bleu_smoothing
        self._predictions: list[str] = []
        self._targets: list[str] = []

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
        predictions, targets = self._corpus()
        if not predictions:
            raise ValueError("No text pairs have been recorded.")
        return self._scores(predictions, targets)

    def reset(self) -> None:
        self._predictions.clear()
        self._targets.clear()

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
            "bleu": corpus_bleu_score(predictions, targets, smooth=self.bleu_smoothing),
            "wer": word_error_rate(predictions, targets),
            "chrf": corpus_chrf_score(predictions, targets),
        }

    def _corpus(self) -> tuple[list[str], list[str]]:
        if not dist.is_available() or not dist.is_initialized():
            return list(self._predictions), list(self._targets)

        local = (tuple(self._predictions), tuple(self._targets))
        gathered: list[tuple[tuple[str, ...], tuple[str, ...]] | None]
        gathered = [None] * dist.get_world_size()
        from anytrain.lightning.audit import audit_collective

        audit_collective(
            "torch.distributed.all_gather_object",
            lambda: dist.all_gather_object(gathered, local),
            source="TextComparisonEvaluator._corpus",
            object_type=type(local).__name__,
            local_item_count=len(local[0]),
            world_size=dist.get_world_size(),
        )

        predictions: list[str] = []
        targets: list[str] = []
        for corpus in gathered:
            if corpus is None:
                raise RuntimeError(
                    "Distributed text corpus gathering returned no value for a rank."
                )
            rank_predictions, rank_targets = corpus
            predictions.extend(rank_predictions)
            targets.extend(rank_targets)
        return predictions, targets
