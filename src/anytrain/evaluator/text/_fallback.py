from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import exp, log

from anytrain._compat import strict_zip


def corpus_bleu_score(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    max_order: int,
    smooth: bool,
    smooth_value: float,
) -> float:
    prediction_length = 0
    target_length = 0
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for prediction, target in strict_zip(predictions, targets):
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        prediction_length += len(prediction_tokens)
        target_length += len(target_tokens)

        for order in range(1, max_order + 1):
            prediction_ngrams = _ngram_counts(prediction_tokens, order)
            target_ngrams = _ngram_counts(target_tokens, order)
            overlap = prediction_ngrams & target_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(prediction_tokens) - order + 1, 0)

    if prediction_length == 0:
        return 100.0 if target_length == 0 else 0.0
    if target_length == 0:
        return 0.0

    precisions: list[float] = []
    for matches, possible in strict_zip(matches_by_order, possible_matches_by_order):
        if possible == 0:
            continue
        if matches == 0:
            if not smooth:
                return 0.0
            precisions.append(smooth_value / possible)
        else:
            precisions.append(matches / possible)

    if not precisions:
        return 0.0

    brevity_penalty = 1.0
    if prediction_length < target_length:
        brevity_penalty = exp(1.0 - target_length / prediction_length)

    mean_log_precision = sum(log(precision) for precision in precisions) / len(precisions)
    return 100.0 * brevity_penalty * exp(mean_log_precision)


def word_error_rate(predictions: Sequence[str], targets: Sequence[str]) -> float:
    total_distance = 0
    total_reference_tokens = 0
    total_empty_reference_errors = 0
    total_empty_references = 0
    for prediction, target in strict_zip(predictions, targets):
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        if not target_tokens:
            total_empty_references += 1
            if prediction_tokens:
                total_empty_reference_errors += 1
            continue

        total_distance += _levenshtein_distance(prediction_tokens, target_tokens)
        total_reference_tokens += len(target_tokens)

    if total_reference_tokens == 0:
        if total_empty_reference_errors == 0:
            return 0.0
        return float(total_empty_reference_errors / total_empty_references)
    return float((total_distance + total_empty_reference_errors) / total_reference_tokens)


def character_error_rate(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Return corpus CER with jiwer-compatible character and empty-reference policy."""
    total_distance = 0
    total_reference_chars = 0
    for prediction, target in strict_zip(predictions, targets):
        prediction_chars = tuple(prediction.strip())
        target_chars = tuple(target.strip())
        total_distance += _levenshtein_distance(prediction_chars, target_chars)
        total_reference_chars += len(target_chars)

    if total_reference_chars == 0:
        # jiwer reports the total number of insertions when every reference is empty.
        return float(total_distance)
    return float(total_distance / total_reference_chars)


def corpus_chrf_score(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    max_order: int,
    beta: float,
) -> float:
    stats = [0] * (3 * max_order)
    for prediction, target in strict_zip(predictions, targets):
        prediction_ngrams = _all_char_ngram_counts(prediction, max_order)
        target_ngrams = _all_char_ngram_counts(target, max_order)
        for index, (prediction_order, target_order) in enumerate(
            strict_zip(prediction_ngrams, target_ngrams)
        ):
            offset = 3 * index
            prediction_count, target_count, match_count = _match_statistics(
                prediction_order,
                target_order,
            )
            stats[offset] += prediction_count
            stats[offset + 1] += target_count
            stats[offset + 2] += match_count

    return _chrf_score_from_stats(stats, beta=beta)


def _chrf_score_from_stats(stats: Sequence[int], *, beta: float) -> float:
    effective_order = 0
    factor = beta * beta
    avg_precision = 0.0
    avg_recall = 0.0

    for index in range(len(stats) // 3):
        prediction_count, target_count, match_count = stats[3 * index : 3 * index + 3]
        if prediction_count > 0 and target_count > 0:
            avg_precision += match_count / prediction_count
            avg_recall += match_count / target_count
            effective_order += 1

    if effective_order == 0:
        return 0.0

    avg_precision /= effective_order
    avg_recall /= effective_order
    if avg_precision + avg_recall == 0.0:
        return 0.0

    score = (1.0 + factor) * avg_precision * avg_recall
    return 100.0 * score / (factor * avg_precision + avg_recall)


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _all_char_ngram_counts(text: str, max_order: int) -> list[Counter[str]]:
    text = "".join(text.split())
    return [
        Counter(text[index : index + order] for index in range(len(text) - order + 1))
        for order in range(1, max_order + 1)
    ]


def _match_statistics(
    prediction_ngrams: Counter[str],
    target_ngrams: Counter[str],
) -> tuple[int, int, int]:
    prediction_count = 0
    match_count = 0
    for ngram, count in prediction_ngrams.items():
        prediction_count += count
        if ngram in target_ngrams:
            match_count += min(count, target_ngrams[ngram])
    target_count = sum(target_ngrams.values())
    return prediction_count if target_count > 0 else 0, target_count, match_count


def _levenshtein_distance(prediction_tokens: Sequence[str], target_tokens: Sequence[str]) -> int:
    previous = list(range(len(target_tokens) + 1))
    for prediction_index, prediction_token in enumerate(prediction_tokens, start=1):
        current = [prediction_index]
        for target_index, target_token in enumerate(target_tokens, start=1):
            substitution_cost = 0 if prediction_token == target_token else 1
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
