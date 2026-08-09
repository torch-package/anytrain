from __future__ import annotations

from collections.abc import Sequence
from warnings import warn

from . import _fallback

TEXT_INSTALL_HINT = (
    'Install text dependencies with `python -m pip install "jiwer>=4.0" "sacrebleu>=2.0"`.'
)


def corpus_bleu_score(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    max_order: int = 4,
    smooth: bool = True,
    smooth_value: float = 0.1,
) -> float:
    """Return a sacreBLEU effective-order BLEU score on the 0-100 scale."""
    _validate_paired_corpus(predictions, targets)
    _validate_positive_integer(max_order, name="max_order")
    if smooth:
        _validate_positive_float(smooth_value, name="smooth_value")

    if _all_empty(predictions) and _all_empty(targets):
        return 100.0
    if _all_empty(predictions) or _all_empty(targets):
        return 0.0

    try:
        from sacrebleu.metrics.bleu import BLEU
    except ImportError as exc:
        _warn_fallback("BLEU", "sacrebleu", exc)
        return _clip_percent(
            _fallback.corpus_bleu_score(
                predictions,
                targets,
                max_order=max_order,
                smooth=smooth,
                smooth_value=smooth_value,
            )
        )

    scorer = BLEU(
        tokenize="none",
        smooth_method="floor" if smooth else "none",
        smooth_value=smooth_value if smooth else None,
        max_ngram_order=max_order,
        effective_order=True,
    )
    return _clip_percent(float(scorer.corpus_score(list(predictions), [list(targets)]).score))


def word_error_rate(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Return corpus WER using jiwer when available."""
    _validate_paired_corpus(predictions, targets)

    try:
        from jiwer import wer
    except ImportError as exc:
        _warn_fallback("WER", "jiwer", exc)
        return _fallback.word_error_rate(predictions, targets)

    return float(wer(reference=list(targets), hypothesis=list(predictions)))


def corpus_chrf_score(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Return a sacreBLEU chrF score on the 0-100 scale."""
    _validate_paired_corpus(predictions, targets)
    _validate_positive_integer(max_order, name="max_order")
    _validate_positive_float(beta, name="beta")

    if _all_empty(predictions) and _all_empty(targets):
        return 100.0
    if _all_empty(predictions) or _all_empty(targets):
        return 0.0

    try:
        from sacrebleu.metrics.chrf import CHRF
    except ImportError as exc:
        _warn_fallback("chrF", "sacrebleu", exc)
        return _clip_percent(
            _fallback.corpus_chrf_score(
                predictions,
                targets,
                max_order=max_order,
                beta=beta,
            )
        )

    scorer = CHRF(char_order=max_order, beta=int(beta))
    return _clip_percent(float(scorer.corpus_score(list(predictions), [list(targets)]).score))


def _validate_paired_corpus(predictions: Sequence[str], targets: Sequence[str]) -> None:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    if not predictions:
        raise ValueError("predictions and targets must contain at least one item.")


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a float.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _all_empty(texts: Sequence[str]) -> bool:
    return all(text == "" for text in texts)


def _clip_percent(value: float) -> float:
    return min(max(value, 0.0), 100.0)


def _warn_fallback(metric: str, package: str, exc: ImportError) -> None:
    warn(
        f"Text {metric} is using the in-package fallback because {package!r} is not "
        f"installed. {TEXT_INSTALL_HINT}",
        RuntimeWarning,
        stacklevel=3,
    )
