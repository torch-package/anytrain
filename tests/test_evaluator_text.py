import sys
import types
import unittest
from typing import Any, cast
from unittest.mock import patch

from anytrain.evaluator import EvaluatorABC, EvaluatorGroup, MetricDict
from anytrain.evaluator.text import TextComparisonEvaluator, TextNormalizationConfig
from anytrain.evaluator.text.scores import (
    character_error_rate,
    corpus_bleu_score,
    corpus_chrf_score,
    word_error_rate,
)


class TextComparisonEvaluatorTest(unittest.TestCase):
    def test_text_configuration_rejects_non_boolean_flags(self):
        for name in (
            "strip",
            "collapse_whitespace",
            "remove_punctuation",
            "lowercase",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, "boolean"):
                cast(Any, TextNormalizationConfig)(**{name: 1})

        with self.assertRaisesRegex(TypeError, "boolean"):
            TextComparisonEvaluator(bleu_smoothing=cast(Any, 1))

    def test_text_evaluator_inherits_evaluator_abc(self):
        self.assertIsInstance(TextComparisonEvaluator(), EvaluatorABC)

    def test_single_string_returns_bleu_wer_chrf_cer_metrics(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("the quick brown fox", "the quick brown fox")

        self.assertEqual(set(metrics), {"bleu", "wer", "chrf", "cer"})
        self.assertEqual(metrics["bleu"], 100.0)
        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)
        self.assertEqual(metrics["cer"], 0.0)

    def test_evaluate_remains_rank_local_and_stateless(self):
        evaluator = TextComparisonEvaluator()

        with patch(
            "anytrain.evaluator.text.evaluator.dist.is_available",
            side_effect=AssertionError("evaluate must not inspect distributed state"),
        ):
            metrics = evaluator.evaluate("hello world", "hello world")

        self.assertEqual(metrics["wer"], 0.0)
        with self.assertRaisesRegex(ValueError, "No text pairs"):
            evaluator.compute()

    def test_batch_scores_are_corpus_scores(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator(
            ["the quick brown fox", "hello world"],
            ["the quick brown fox", "hello there"],
        )

        self.assertGreater(metrics["bleu"], 50.0)
        self.assertLess(metrics["bleu"], 100.0)
        self.assertAlmostEqual(metrics["wer"], 1.0 / 6.0)
        self.assertGreater(metrics["chrf"], 50.0)
        self.assertLess(metrics["chrf"], 100.0)
        self.assertGreater(metrics["cer"], 0.0)
        self.assertLess(metrics["cer"], 1.0)

    def test_running_scores_match_direct_corpus_across_uneven_batches(self):
        correct_predictions = ["the quick brown fox"] * 10
        correct_targets = ["the quick brown fox"] * 10
        wrong_predictions = ["slow turtle sleeps now"]
        wrong_targets = ["the quick brown fox"]
        evaluator = TextComparisonEvaluator()

        direct = evaluator(
            correct_predictions + wrong_predictions,
            correct_targets + wrong_targets,
        )
        evaluator.update(correct_predictions, correct_targets)
        evaluator.update(wrong_predictions, wrong_targets)
        running = evaluator.compute()

        for name, expected in direct.items():
            self.assertAlmostEqual(running[name], expected)
        running_wer: Any = running["wer"]
        self.assertAlmostEqual(running_wer, 1.0 / 11.0)

    def test_running_scores_reset_clears_corpus(self):
        evaluator = TextComparisonEvaluator()
        evaluator.update("hello world", "hello world")

        self.assertEqual(evaluator.compute()["wer"], 0.0)
        evaluator.reset()

        with self.assertRaisesRegex(ValueError, "No text pairs"):
            evaluator.compute()

    def test_evaluator_group_delegates_text_corpus_lifecycle(self):
        correct_predictions = ["the quick brown fox"] * 10
        correct_targets = ["the quick brown fox"] * 10
        wrong_predictions = ["slow turtle sleeps now"]
        wrong_targets = ["the quick brown fox"]
        expected = TextComparisonEvaluator()(
            correct_predictions + wrong_predictions,
            correct_targets + wrong_targets,
        )
        evaluator = EvaluatorGroup({"text": TextComparisonEvaluator()})

        evaluator.update(correct_predictions, correct_targets)
        evaluator.update(wrong_predictions, wrong_targets)
        running = evaluator.compute()

        for name, value in expected.items():
            self.assertAlmostEqual(running[f"text/{name}"], value)
        evaluator.reset()
        with self.assertRaisesRegex(ValueError, "No text pairs"):
            evaluator.compute()

    def test_group_rejects_mixed_lifecycle_before_updating_stateful_evaluator(self):
        class StatelessEvaluator(EvaluatorABC):
            def evaluate(self, prediction_text, target_text) -> MetricDict:
                del prediction_text, target_text
                return {"score": 1.0}

        text = TextComparisonEvaluator()
        evaluator = EvaluatorGroup({"text": text, "stateless": StatelessEvaluator()})

        with self.assertRaisesRegex(NotImplementedError, "complete stateful lifecycle"):
            evaluator.update("hello", "hello")

        with self.assertRaisesRegex(ValueError, "No text pairs"):
            text.compute()

    def test_distributed_compute_scores_on_root_once_and_broadcasts_result(self):
        evaluator = TextComparisonEvaluator(lowercase=True)
        evaluator.update(" Local, TEXT! ", "local text")
        semantic_key = evaluator._semantic_key()
        remote = (semantic_key, ("wrong words",), ("right words",))
        expected = evaluator(
            ["local text", "wrong words"],
            ["local text", "right words"],
        )

        def gather(payload, gathered, *, dst):
            self.assertEqual(dst, 0)
            self.assertEqual(
                payload,
                (semantic_key, ("local text",), ("local text",)),
            )
            self.assertIsNotNone(gathered)
            assert gathered is not None
            gathered[:] = [payload, remote]

        def broadcast(envelope, *, src):
            self.assertEqual(src, 0)
            self.assertEqual(envelope[0], {"status": "ok", "metrics": expected})

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=0),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ) as gather_object,
            patch(
                "anytrain.evaluator.text.evaluator.dist.broadcast_object_list",
                side_effect=broadcast,
            ) as broadcast_object_list,
            patch.object(evaluator, "_scores", wraps=evaluator._scores) as scores,
        ):
            running = evaluator.compute()

        gather_object.assert_called_once()
        broadcast_object_list.assert_called_once()
        scores.assert_called_once_with(
            ["local text", "wrong words"],
            ["local text", "right words"],
        )
        self.assertEqual(running, expected)

    def test_distributed_compute_non_root_uses_broadcast_without_scoring(self):
        evaluator = TextComparisonEvaluator()
        evaluator.update("local text", "local text")
        semantic_key = evaluator._semantic_key()
        expected = {"bleu": 75.0, "wer": 0.25, "chrf": 80.0, "cer": 0.1}

        def gather(payload, gathered, *, dst):
            self.assertEqual(dst, 0)
            self.assertEqual(
                payload,
                (semantic_key, ("local text",), ("local text",)),
            )
            self.assertIsNone(gathered)

        def broadcast(envelope, *, src):
            self.assertEqual(src, 0)
            envelope[0] = {"status": "ok", "metrics": expected}

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=1),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ),
            patch(
                "anytrain.evaluator.text.evaluator.dist.broadcast_object_list",
                side_effect=broadcast,
            ),
            patch.object(
                evaluator,
                "_scores",
                side_effect=AssertionError("non-root must not score the global corpus"),
            ),
        ):
            running = evaluator.compute()

        self.assertEqual(running, expected)

    def test_distributed_compute_accepts_empty_local_corpus_on_root(self):
        evaluator = TextComparisonEvaluator()
        semantic_key = evaluator._semantic_key()
        remote = (semantic_key, ("hello world",), ("hello world",))
        expected = evaluator(remote[1], remote[2])

        def gather(payload, gathered, *, dst):
            self.assertEqual(dst, 0)
            self.assertEqual(payload, (semantic_key, (), ()))
            assert gathered is not None
            gathered[:] = [payload, remote]

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=0),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ),
            patch("anytrain.evaluator.text.evaluator.dist.broadcast_object_list"),
        ):
            running = evaluator.compute()

        self.assertEqual(running, expected)

    def test_distributed_compute_broadcasts_root_scoring_error(self):
        evaluator = TextComparisonEvaluator()
        evaluator.update("hello world", "hello world")

        def gather(payload, gathered, *, dst):
            del dst
            assert gathered is not None
            gathered[:] = [payload, payload]

        def broadcast(envelope, *, src):
            self.assertEqual(src, 0)
            self.assertEqual(envelope[0]["status"], "error")
            self.assertEqual(envelope[0]["error_type"], "ValueError")
            self.assertEqual(envelope[0]["error"], "scoring exploded")

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=0),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ),
            patch(
                "anytrain.evaluator.text.evaluator.dist.broadcast_object_list",
                side_effect=broadcast,
            ),
            patch.object(evaluator, "_scores", side_effect=ValueError("scoring exploded")),
            self.assertRaisesRegex(RuntimeError, "ValueError: scoring exploded") as raised,
        ):
            evaluator.compute()

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_distributed_compute_rejects_mismatched_semantic_config(self):
        evaluator = TextComparisonEvaluator()
        evaluator.update("hello world", "hello world")
        remote_key = TextComparisonEvaluator(lowercase=True)._semantic_key()

        def gather(payload, gathered, *, dst):
            del dst
            assert gathered is not None
            gathered[:] = [payload, (remote_key, ("hello",), ("hello",))]

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=0),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ),
            patch("anytrain.evaluator.text.evaluator.dist.broadcast_object_list"),
            self.assertRaisesRegex(RuntimeError, "different semantic configurations"),
        ):
            evaluator.compute()

    def test_distributed_compute_non_root_raises_broadcast_root_error(self):
        evaluator = TextComparisonEvaluator()

        def gather(payload, gathered, *, dst):
            del payload
            self.assertEqual(dst, 0)
            self.assertIsNone(gathered)

        def broadcast(envelope, *, src):
            self.assertEqual(src, 0)
            envelope[0] = {
                "status": "error",
                "error_type": "ValueError",
                "error": "No text pairs have been recorded.",
            }

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=1),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ),
            patch(
                "anytrain.evaluator.text.evaluator.dist.broadcast_object_list",
                side_effect=broadcast,
            ),
            self.assertRaisesRegex(RuntimeError, "ValueError: No text pairs"),
        ):
            evaluator.compute()

    def test_distributed_compute_rejects_globally_empty_corpus(self):
        evaluator = TextComparisonEvaluator()

        def gather(payload, gathered, *, dst):
            del dst
            assert gathered is not None
            gathered[:] = [payload, payload]

        with (
            patch("anytrain.evaluator.text.evaluator.dist.is_available", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.is_initialized", return_value=True),
            patch("anytrain.evaluator.text.evaluator.dist.get_world_size", return_value=2),
            patch("anytrain.evaluator.text.evaluator.dist.get_rank", return_value=0),
            patch(
                "anytrain.evaluator.text.evaluator.dist.gather_object",
                side_effect=gather,
            ),
            patch("anytrain.evaluator.text.evaluator.dist.broadcast_object_list"),
            self.assertRaisesRegex(RuntimeError, "ValueError: No text pairs"),
        ):
            evaluator.compute()

    def test_normalization_defaults_strip_and_collapse_whitespace(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("  hello \n\t world  ", "hello world")

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_normalization_removes_punctuation_by_default(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("Please call Stella", "Please call Stella.")

        self.assertEqual(metrics["bleu"], 100.0)
        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_punctuation_removal_can_be_disabled(self):
        evaluator = TextComparisonEvaluator(remove_punctuation=False)

        metrics = evaluator("Please call Stella", "Please call Stella.")

        self.assertGreater(metrics["wer"], 0.0)
        self.assertLess(metrics["chrf"], 100.0)

    def test_smoothed_bleu_reports_short_partial_match(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("Please cool Stella", "Please call Stella.")

        self.assertGreater(metrics["bleu"], 0.0)
        self.assertLess(metrics["bleu"], 100.0)
        self.assertAlmostEqual(metrics["wer"], 1.0 / 3.0)

    def test_bleu_smoothing_can_be_disabled(self):
        evaluator = TextComparisonEvaluator(bleu_smoothing=False)

        metrics = evaluator("Please cool Stella", "Please call Stella.")

        self.assertEqual(metrics["bleu"], 0.0)

    def test_chinese_bleu_requires_the_explicit_zh_tokenizer(self):
        predictions = ["今天天气不错我们去公园"]
        targets = ["今天天气很好我们去公园"]

        untokenized = TextComparisonEvaluator(bleu_tokenize="none")(
            predictions,
            targets,
        )
        chinese = TextComparisonEvaluator(bleu_tokenize="zh")(
            predictions,
            targets,
        )

        self.assertEqual(untokenized["bleu"], 0.0)
        self.assertGreater(chinese["bleu"], 0.0)
        self.assertLess(chinese["bleu"], 100.0)

    def test_metric_signatures_record_bleu_tokenizer_and_chrfpp_profile(self):
        signatures = TextComparisonEvaluator(
            bleu_tokenize="zh",
            chrf_word_order=2,
        ).metric_signatures()

        self.assertIn("tok:zh", signatures["bleu"])
        self.assertIn("nc:6", signatures["chrf"])
        self.assertIn("nw:2", signatures["chrf"])

    def test_metric_profile_options_are_validated(self):
        with self.assertRaisesRegex(ValueError, "tokenize"):
            TextComparisonEvaluator(bleu_tokenize="auto")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TextComparisonEvaluator(chrf_word_order=-1)

    def test_normalization_lowercase_is_explicit(self):
        evaluator = TextComparisonEvaluator(lowercase=True)

        metrics = evaluator("Hello WORLD", "hello world")

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_normalization_preserves_case_by_default(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("Hello WORLD", "hello world")

        self.assertEqual(metrics["wer"], 1.0)
        self.assertLess(metrics["chrf"], 100.0)

    def test_normalization_converts_traditional_chinese_to_simplified_by_default(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("從一流的媒體實驗室", "从一流的媒体实验室")

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_chinese_normalization_can_be_disabled(self):
        evaluator = TextComparisonEvaluator(chinese=None)

        metrics = evaluator("從一流的媒體實驗室", "从一流的媒体实验室")

        self.assertGreater(metrics["wer"], 0.0)
        self.assertLess(metrics["chrf"], 100.0)

    def test_invalid_chinese_normalization_raises(self):
        evaluator = TextComparisonEvaluator(chinese="traditional")

        with self.assertRaisesRegex(ValueError, "simplified"):
            evaluator("從", "从")

    def test_batch_length_mismatch_raises(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(ValueError, "same batch length"):
            evaluator(["hello"], ["hello", "world"])

    def test_text_input_rejects_bytes(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(TypeError, "prediction_text"):
            evaluator(b"hello", "hello")

    def test_text_input_rejects_non_string_items(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(TypeError, r"prediction_text\[0\]"):
            evaluator([1], ["hello"])

    def test_empty_reference_wer_policy(self):
        evaluator = TextComparisonEvaluator()

        empty_metrics = evaluator("", "")
        non_empty_prediction_metrics = evaluator("hello", "")

        self.assertEqual(empty_metrics["wer"], 0.0)
        self.assertEqual(empty_metrics["bleu"], 100.0)
        self.assertEqual(empty_metrics["chrf"], 100.0)
        self.assertEqual(empty_metrics["cer"], 0.0)
        self.assertEqual(non_empty_prediction_metrics["wer"], 1.0)
        self.assertEqual(non_empty_prediction_metrics["bleu"], 0.0)
        self.assertEqual(non_empty_prediction_metrics["chrf"], 0.0)
        self.assertEqual(non_empty_prediction_metrics["cer"], 5.0)

    def test_metric_keys_are_stable(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("a b c d", "a b c d")

        self.assertEqual(list(metrics), ["bleu", "wer", "chrf", "cer"])

    def test_empty_batch_raises(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(ValueError, "at least one item"):
            evaluator([], [])

    def test_wer_warns_and_uses_fallback_when_jiwer_is_missing(self):
        real_import = __import__

        def fail_jiwer_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jiwer":
                raise ImportError("missing jiwer")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fail_jiwer_import),
            self.assertWarnsRegex(RuntimeWarning, "fallback"),
        ):
            score = word_error_rate(["hello world"], ["hello there"])

        self.assertEqual(score, 0.5)

    def test_cer_uses_corpus_reference_character_denominator(self):
        predictions = ["ab", "x"]
        targets = ["ac", ""]

        # One substitution plus one insertion over two reference characters.
        self.assertEqual(character_error_rate(predictions, targets), 1.0)

    def test_cer_warns_and_uses_fallback_when_jiwer_is_missing(self):
        predictions = ["ab", "x"]
        targets = ["ac", ""]
        expected = character_error_rate(predictions, targets)
        real_import = __import__

        def fail_jiwer_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jiwer":
                raise ImportError("missing jiwer")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fail_jiwer_import),
            self.assertWarnsRegex(RuntimeWarning, "fallback"),
        ):
            fallback = character_error_rate(predictions, targets)

        self.assertEqual(fallback, expected)

    def test_chrf_fallback_matches_sacrebleu_for_corpus_examples(self):
        examples = [
            (["Please cool Stella"], ["Please call Stella"]),
            (["the quick brown fox", "hello world"], ["the quick brown fox", "hello there"]),
        ]

        real_import = __import__

        def fail_sacrebleu_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sacrebleu"):
                raise ImportError("missing sacrebleu")
            return real_import(name, globals, locals, fromlist, level)

        for predictions, targets in examples:
            expected = corpus_chrf_score(predictions, targets)
            with (
                patch("builtins.__import__", side_effect=fail_sacrebleu_import),
                self.assertWarnsRegex(RuntimeWarning, "fallback"),
            ):
                fallback = corpus_chrf_score(predictions, targets)

            self.assertAlmostEqual(fallback, expected)

    def test_chrf_preserves_fractional_beta_and_rejects_nonfinite_values(self):
        predictions = ["ab"]
        targets = ["ac"]
        observed: list[float] = []

        class FakeCHRF:
            def __init__(self, *, char_order: int, word_order: int, beta: float):
                self.char_order = char_order
                self.word_order = word_order
                observed.append(beta)

            def corpus_score(self, _predictions, _targets):
                return types.SimpleNamespace(score=37.0)

        fake_chrf = types.ModuleType("sacrebleu.metrics.chrf")
        cast(Any, fake_chrf).CHRF = FakeCHRF
        with patch.dict(
            sys.modules,
            {
                "sacrebleu": types.ModuleType("sacrebleu"),
                "sacrebleu.metrics": types.ModuleType("sacrebleu.metrics"),
                "sacrebleu.metrics.chrf": fake_chrf,
            },
        ):
            self.assertEqual(corpus_chrf_score(predictions, targets, beta=2.5), 37.0)
        self.assertEqual(observed, [2.5])

        for value in (float("nan"), float("inf"), 0.0):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite|positive"
            ):
                corpus_chrf_score(predictions, targets, beta=value)

    def test_zh_bleu_fallback_matches_sacrebleu(self):
        predictions = ["今天天气不错我们去公园", "机器翻译质量很好"]
        targets = ["今天天气很好我们去公园", "机器翻译质量不错"]
        expected = corpus_bleu_score(predictions, targets, tokenize="zh")
        real_import = __import__

        def fail_sacrebleu_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sacrebleu"):
                raise ImportError("missing sacrebleu")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fail_sacrebleu_import),
            self.assertWarnsRegex(RuntimeWarning, "fallback"),
        ):
            fallback = corpus_bleu_score(predictions, targets, tokenize="zh")

        self.assertAlmostEqual(fallback, expected)


if __name__ == "__main__":
    unittest.main()
