import inspect
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

from anytrain.audio.bpe import CodecBPE, CodecBPETrainerSpec

try:
    import tokenizers
except ImportError:
    tokenizers = None


def state():
    return {
        "codebook_sizes": [16],
        "tokens": {
            "0": [[1]],
            "1": [[2]],
            "2": [[3]],
            "3": [[1], [2]],
            "4": [[1], [2], [3]],
        },
        "merges": [
            {"left": 0, "right": 1, "token_id": 3},
            {"left": 3, "right": 2, "token_id": 4},
        ],
    }


def load_state(value):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "codec_bpe.json").write_text(json.dumps(value), encoding="utf-8")
        return CodecBPE.from_pretrained(path)


def replay(corpus):
    return lambda: corpus


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class BPETest(unittest.TestCase):
    def test_trainer_spec_freezes_output_affecting_contract(self):
        trainer = CodecBPETrainerSpec(tokenizers_version="0.22.2")

        self.assertEqual(trainer.vocab_size, 30_000)
        self.assertEqual(trainer.min_frequency, 0)
        self.assertIsNone(trainer.max_token_length)
        self.assertEqual(trainer.max_frames, 1_000_000_000)
        self.assertEqual(
            trainer.identity,
            "codec-bpe-trainer-v1:"
            "6da79642fd164dc58c82ef0bd096f4a00cc1679f84f742c8e3a788756dc59901",
        )
        for changed in (
            replace(trainer, vocab_size=30_001),
            replace(trainer, min_frequency=1),
            replace(trainer, max_token_length=8),
            replace(trainer, max_frames=None),
            replace(trainer, tokenizers_version="0.22.3"),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(changed.identity, trainer.identity)

    def test_train_from_spec_uses_the_frozen_contract(self):
        trainer = CodecBPETrainerSpec(
            vocab_size=18,
            min_frequency=0,
            max_token_length=2,
            max_frames=3,
        )
        bpe = CodecBPE.train_from_spec(
            replay([[[1], [2], [1]], [[2], [3], [4]]]),
            codebook_sizes=(16,),
            trainer=trainer,
            show_progress=False,
        )

        token_lengths = (len(bpe.decode([token])) for token in range(bpe.vocab_size))
        self.assertLessEqual(max(token_lengths), 2)

    def test_train_from_spec_rejects_tokenizers_contract_drift(self):
        trainer = CodecBPETrainerSpec(tokenizers_version="0.0.0")

        with self.assertRaisesRegex(RuntimeError, "tokenizers==0.0.0"):
            CodecBPE.train_from_spec(
                replay([[[1], [2]]]),
                codebook_sizes=(16,),
                trainer=trainer,
                show_progress=False,
            )

    def test_trainer_spec_rejects_invalid_semantic_parameters(self):
        for kwargs in (
            {"vocab_size": 0},
            {"min_frequency": -1},
            {"max_token_length": 0},
            {"max_frames": 0},
            {"max_frames": True},
            {"tokenizers_version": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                CodecBPETrainerSpec(**cast(Any, kwargs))

    def test_train_defaults_to_one_billion_frames(self):
        parameter = inspect.signature(CodecBPE.train).parameters["max_frames"]

        self.assertEqual(parameter.default, 1_000_000_000)

    def test_train_reports_missing_tokenizer_extra(self):
        with (
            patch.dict(sys.modules, {"tokenizers": None, "tokenizers.models": None}),
            self.assertRaisesRegex(ImportError, r'pip install "tokenizers>=0.20"'),
        ):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(4,),
                show_progress=False,
            )

    def test_from_pretrained_accepts_frame_tokens_and_merges(self):
        bpe = load_state(state())

        self.assertEqual(bpe.encode([[1], [2], [3]]), [4])
        self.assertEqual(bpe.decode([4]), [(1,), (2,), (3,)])

    def test_from_pretrained_round_trip(self):
        bpe = load_state(state())

        with tempfile.TemporaryDirectory() as tmp:
            out = bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

            self.assertTrue((out / "codec_bpe.json").exists())

        self.assertEqual(loaded.codebook_sizes, bpe.codebook_sizes)
        self.assertEqual(loaded.vocab_size, bpe.vocab_size)
        self.assertEqual(loaded.encode([[1], [2], [3]]), [4])
        self.assertEqual(loaded.decode([4]), bpe.decode([4]))

    def test_train_merges_single_codebook_frames_and_round_trips(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=18,
        )

        token_ids = bpe.encode([[1], [2], [1], [2], [3]])

        self.assertEqual(bpe.vocab_size, 18)
        self.assertEqual(token_ids, [16, 17])
        self.assertEqual(bpe.decode(token_ids), [(1,), (2,), (1,), (2,), (3,)])

    def test_train_supports_longcat_scale_frame_vocab(self):
        frames = [[value] for value in range(8192)]

        bpe = CodecBPE.train(replay([frames]), codebook_sizes=(8192,), vocab_size=8192)

        encoded = bpe.encode([[0], [4096], [8191]])
        self.assertEqual(encoded, [0, 4096, 8191])
        self.assertEqual(bpe.decode(encoded), [(0,), (4096,), (8191,)])

    def test_train_keeps_alphabet_when_vocab_size_is_smaller(self):
        bpe = CodecBPE.train(
            replay([[[100], [101], [102]]]),
            codebook_sizes=(128,),
            vocab_size=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 128)
        self.assertEqual(bpe.encode([[100], [101], [102]]), [100, 101, 102])

    def test_train_respects_min_frequency(self):
        bpe = CodecBPE.train(
            replay([[[1], [2]]]),
            codebook_sizes=(16,),
            vocab_size=3,
            min_frequency=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 16)
        self.assertEqual(bpe.encode([[1], [2]]), [1, 2])

    def test_train_respects_max_token_length(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=18,
            max_token_length=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 17)
        self.assertEqual(bpe.encode([[1], [2], [3]]), [16, 3])

    def test_train_stops_after_full_sequence_at_max_frames(self):
        bpe = CodecBPE.train(
            replay(
                [
                    [[1], [2]],
                    [[4], [5], [4], [5]],
                    [[16]],
                ]
            ),
            codebook_sizes=(16,),
            vocab_size=17,
            max_frames=3,
            show_progress=False,
        )

        token_ids = bpe.encode([[4], [5]])

        self.assertEqual(len(token_ids), 1)
        self.assertEqual(bpe.decode(token_ids), [(4,), (5,)])

    def test_train_does_not_pull_after_max_frames_boundary(self):
        def corpus():
            yield [[1], [2]]
            yield [[3], [4]]
            raise AssertionError("corpus was read past max_frames")

        bpe = CodecBPE.train(
            corpus,
            codebook_sizes=(16,),
            max_frames=4,
            show_progress=False,
        )

        self.assertEqual(bpe.decode(bpe.encode([[3], [4]])), [(3,), (4,)])

    def test_train_max_frames_none_uses_full_corpus(self):
        consumed = []

        def corpus():
            for frames in ([[1], [2]], [[3], [4]]):
                consumed.append(frames)
                yield frames

        bpe = CodecBPE.train(
            corpus,
            codebook_sizes=(16,),
            max_frames=None,
            show_progress=False,
        )

        self.assertEqual(consumed, [[[1], [2]], [[3], [4]]])
        self.assertEqual(bpe.decode(bpe.encode([[3], [4]])), [(3,), (4,)])

    def test_train_max_frames_progress_tracks_frame_limit(self):
        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=True),
            patch("tqdm.auto.tqdm") as tqdm,
        ):
            CodecBPE.train(
                replay(
                    [
                        [[1], [2]],
                        [[3], [4], [5], [6]],
                    ]
                ),
                codebook_sizes=(16,),
                max_frames=3,
                show_progress=True,
            )

        tqdm.assert_called_once_with(
            total=3,
            desc="CodecBPE frames",
            unit="frame",
            unit_scale=True,
        )
        tqdm.return_value.update.assert_has_calls([call(2), call(1)])
        tqdm.return_value.close.assert_called_once_with()
        tqdm.write.assert_any_call("CodecBPE corpus: 6 frames in 2 sequences; frame limit reached")

    def test_train_max_frames_none_keeps_default_progress(self):
        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=True),
            patch("tqdm.auto.tqdm") as tqdm,
        ):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(16,),
                max_frames=None,
                show_progress=True,
            )

        tqdm.assert_not_called()
        tqdm.write.assert_any_call("CodecBPE trainer: started (corpus, pair counts, merges)")
        tqdm.write.assert_any_call("CodecBPE trainer: completed")

    def test_train_max_frames_progress_reports_early_corpus_exhaustion(self):
        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=True),
            patch("tqdm.auto.tqdm") as tqdm,
        ):
            CodecBPE.train(
                replay([[[1], [2]], [[3], [4]]]),
                codebook_sizes=(16,),
                max_frames=10,
                show_progress=True,
            )

        tqdm.return_value.update.assert_has_calls([call(2), call(2)])
        tqdm.write.assert_any_call("CodecBPE corpus: 4 frames in 2 sequences; corpus exhausted")

    def test_train_max_frames_progress_respects_show_progress(self):
        with patch("tqdm.auto.tqdm") as tqdm:
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(16,),
                max_frames=2,
                show_progress=False,
            )

        tqdm.assert_not_called()
        tqdm.write.assert_not_called()

    def test_train_progress_uses_static_logs_without_dynamic_bars_non_interactive(self):
        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=False),
            patch("anytrain.audio.bpe._core.STATIC_PROGRESS_INTERVAL_SECONDS", 0),
            patch("builtins.print") as output,
            patch("tqdm.auto.tqdm") as tqdm,
        ):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(16,),
                max_frames=2,
                show_progress=True,
            )

        tqdm.assert_not_called()
        output.assert_any_call(
            "CodecBPE alphabet: skipped for single codebook",
            file=sys.stderr,
            flush=True,
        )
        output.assert_any_call(
            "CodecBPE corpus: 2/2 frames (100.0%) in 1 sequences processed",
            file=sys.stderr,
            flush=True,
        )
        output.assert_any_call(
            "CodecBPE corpus: 2 frames in 1 sequences; frame limit reached",
            file=sys.stderr,
            flush=True,
        )
        output.assert_any_call(
            "CodecBPE trainer: started (corpus, pair counts, merges)",
            file=sys.stderr,
            flush=True,
        )
        output.assert_any_call(
            "CodecBPE trainer: completed",
            file=sys.stderr,
            flush=True,
        )

    def test_train_disables_tokenizers_progress_non_interactive(self):
        from tokenizers.trainers import BpeTrainer

        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=False),
            patch("tokenizers.trainers.BpeTrainer", wraps=BpeTrainer) as trainer,
        ):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(16,),
                show_progress=True,
            )

        self.assertFalse(trainer.call_args.kwargs["show_progress"])

    def test_multi_codebook_alphabet_uses_periodic_static_logs_non_interactive(self):
        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=False),
            patch("anytrain.audio.bpe._core.STATIC_PROGRESS_INTERVAL_SECONDS", 0),
            patch("builtins.print") as output,
        ):
            CodecBPE.train(
                replay([[[1, 2], [2, 3]]]),
                codebook_sizes=(4, 4),
                max_frames=2,
                show_progress=True,
            )

        output.assert_any_call(
            "CodecBPE alphabet: 1 sequences processed",
            file=sys.stderr,
            flush=True,
        )

    def test_train_max_frames_progress_closes_on_corpus_error(self):
        def corpus():
            yield [[1], [2]]
            raise RuntimeError("corpus failed")

        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=True),
            patch("tqdm.auto.tqdm") as tqdm,
            self.assertRaisesRegex(RuntimeError, "corpus failed"),
        ):
            CodecBPE.train(
                corpus,
                codebook_sizes=(16,),
                max_frames=10,
                show_progress=True,
            )

        tqdm.return_value.close.assert_called_once_with()
        self.assertFalse(
            any(
                progress_call.args and progress_call.args[0].startswith("CodecBPE corpus:")
                for progress_call in tqdm.write.call_args_list
            )
        )
        self.assertNotIn(call("CodecBPE trainer: completed"), tqdm.write.call_args_list)

    def test_multi_codebook_frame_progress_only_wraps_training_pass(self):
        frame_bar = MagicMock()

        def progress(iterable=None, **_):
            return iterable if iterable is not None else frame_bar

        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=True),
            patch("tqdm.auto.tqdm", side_effect=progress) as tqdm,
        ):
            CodecBPE.train(
                replay([[[1, 2], [2, 3]]]),
                codebook_sizes=(4, 4),
                max_frames=2,
                show_progress=True,
            )

        frame_bars = [
            progress_call
            for progress_call in tqdm.call_args_list
            if progress_call.kwargs.get("desc") == "CodecBPE frames"
        ]
        self.assertEqual(
            frame_bars,
            [
                call(
                    total=2,
                    desc="CodecBPE frames",
                    unit="frame",
                    unit_scale=True,
                )
            ],
        )

    def test_train_rejects_non_positive_max_frames(self):
        for max_frames in (0, -1):
            with (
                self.subTest(max_frames=max_frames),
                self.assertRaisesRegex(ValueError, "max_frames"),
            ):
                CodecBPE.train(
                    replay([[[1], [2]]]),
                    codebook_sizes=(16,),
                    max_frames=max_frames,
                    show_progress=False,
                )

    def test_train_rejects_non_integer_max_frames(self):
        for max_frames in (True, 1.5):
            with (
                self.subTest(max_frames=max_frames),
                self.assertRaisesRegex(TypeError, "max_frames"),
            ):
                CodecBPE.train(
                    replay([[[1], [2]]]),
                    codebook_sizes=(16,),
                    max_frames=cast(Any, max_frames),
                    show_progress=False,
                )

    def test_train_requires_callable_corpus(self):
        with self.assertRaisesRegex(TypeError, "corpus must be callable"):
            CodecBPE.train(
                cast(Any, [[[1], [2]]]),
                codebook_sizes=(16,),
                show_progress=False,
            )
        with self.assertRaisesRegex(TypeError, "show_progress must be a bool"):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(16,),
                show_progress=cast(Any, 1),
            )

    def test_codebook_sizes_require_positive_integers(self):
        for sizes in ((True,), (16.0,), ("16",)):
            with (
                self.subTest(sizes=sizes),
                self.assertRaisesRegex(TypeError, "codebook sizes must be integers"),
            ):
                CodecBPE.train(
                    replay([[[1], [2]]]),
                    codebook_sizes=cast(Any, sizes),
                    show_progress=False,
                )

    def test_train_merges_multi_codebook_frames_and_round_trips(self):
        corpus = [
            [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
            [[1, 4], [2, 7], [3, 8]],
        ]

        bpe = CodecBPE.train(replay(corpus), codebook_sizes=(4, 16), vocab_size=5)

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(bpe.encode([[1, 4], [2, 7], [3, 8]]), [4])
        self.assertEqual(bpe.decode([4]), [(1, 4), (2, 7), (3, 8)])

    def test_train_progress_keeps_same_result(self):
        corpus = [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

        plain = CodecBPE.train(
            replay(corpus),
            codebook_sizes=(16,),
            vocab_size=18,
            show_progress=False,
        )
        with_progress = CodecBPE.train(
            replay(corpus),
            codebook_sizes=(16,),
            vocab_size=18,
            show_progress=True,
        )

        self.assertEqual(with_progress.vocab_size, plain.vocab_size)
        self.assertEqual(
            with_progress.encode([[1], [2], [1], [2], [3]]),
            plain.encode([[1], [2], [1], [2], [3]]),
        )

    def test_single_codebook_train_calls_corpus_once(self):
        calls = 0

        def corpus():
            nonlocal calls
            calls += 1
            return iter([[[1], [2], [1], [2], [3]], [[1], [2], [3]]])

        bpe = CodecBPE.train(corpus, codebook_sizes=(16,), vocab_size=18, show_progress=False)

        self.assertEqual(calls, 1)
        self.assertEqual(bpe.decode(bpe.encode([[1], [2], [3]])), [(1,), (2,), (3,)])

    def test_multi_codebook_train_calls_corpus_twice(self):
        calls = 0

        def corpus():
            nonlocal calls
            calls += 1
            return iter([[[1, 2], [2, 3]]])

        CodecBPE.train(corpus, codebook_sizes=(4, 4), show_progress=False)

        self.assertEqual(calls, 2)

    def test_multi_codebook_train_replays_same_max_frames_prefix(self):
        pulled = []
        sequences = (
            [[1, 4], [2, 7]],
            [[3, 8], [0, 9]],
            [[4, 10]],
        )

        def corpus():
            current = []
            pulled.append(current)
            for index, frames in enumerate(sequences):
                current.append(index)
                yield frames

        bpe = CodecBPE.train(
            corpus,
            codebook_sizes=(4, 16),
            max_frames=3,
            show_progress=False,
        )
        prefix = [[1, 4], [2, 7], [3, 8], [0, 9]]

        self.assertEqual(pulled, [[0, 1], [0, 1]])
        self.assertEqual(
            bpe.decode(bpe.encode(prefix)),
            [(1, 4), (2, 7), (3, 8), (0, 9)],
        )

    def test_large_single_codebook_falls_back_to_alphabet_scan(self):
        calls = 0

        def corpus():
            nonlocal calls
            calls += 1
            return iter([[[1], [2]]])

        bpe = CodecBPE.train(
            corpus,
            codebook_sizes=(200_000,),
            show_progress=False,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(bpe.decode(bpe.encode([[1], [2]])), [(1,), (2,)])

    def test_single_codebook_reports_skipped_alphabet(self):
        output = io.StringIO()

        with redirect_stderr(output):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(4,),
                show_progress=True,
            )

        self.assertIn("alphabet: skipped for single codebook", output.getvalue())

    def test_rejects_invalid_frames(self):
        with self.assertRaisesRegex(ValueError, "number of codebooks"):
            CodecBPE.train(replay([[[1, 4], [2, 7, 9]]]), codebook_sizes=(4, 16))
        with self.assertRaisesRegex(ValueError, "codebook 1"):
            CodecBPE.train(replay([[[1, 4], [2, 17]]]), codebook_sizes=(4, 16))
        with self.assertRaisesRegex(TypeError, "use \\[id\\]"):
            CodecBPE.train(replay([[1, 2, 3]]), codebook_sizes=(16,))
        with self.assertRaisesRegex(TypeError, "integers"):
            CodecBPE.train(replay([[[1.5]]]), codebook_sizes=(16,))

    def test_rejects_empty_corpus_and_sequences(self):
        with self.assertRaisesRegex(ValueError, "corpus"):
            CodecBPE.train(replay([]), codebook_sizes=(16,))
        with self.assertRaisesRegex(ValueError, "empty sequences"):
            CodecBPE.train(replay([[]]), codebook_sizes=(16,))

    def test_encode_rejects_empty_and_unknown_frames(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "frames"):
            bpe.encode([])
        self.assertEqual(bpe.decode(bpe.encode([[4]])), [(4,)])
        with self.assertRaisesRegex(ValueError, "must be in"):
            bpe.encode([[16]])

    def test_decode_rejects_empty_and_unknown_token_ids(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "token_ids"):
            bpe.decode([])
        with self.assertRaisesRegex(ValueError, "unknown CodecBPE token id"):
            bpe.decode([bpe.vocab_size])
        for token_id in (True, 1.0, "1"):
            with (
                self.subTest(token_id=token_id),
                self.assertRaisesRegex(TypeError, "token ids must be integers"),
            ):
                bpe.decode(cast(Any, [token_id]))

    def test_eval_reports_expected_compression(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=18,
        )

        stats = bpe.evaluate(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            show_progress=False,
        )

        self.assertEqual(stats["num_sequences"], 2)
        self.assertEqual(stats["original_frames"], 8)
        self.assertEqual(stats["encoded_tokens"], 3)
        self.assertEqual(stats["mean_original_length"], 4.0)
        self.assertEqual(stats["mean_encoded_length"], 1.5)
        self.assertAlmostEqual(stats["compression_ratio"], 3 / 8)
        self.assertAlmostEqual(stats["compression_factor"], 8 / 3)
        self.assertAlmostEqual(stats["compression_gain"], 5 / 8)

    def test_eval_reports_token_distributions(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=18,
        )

        stats = bpe.evaluate(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            show_progress=False,
        )

        self.assertEqual(stats["encoded_tokens"], 3)
        self.assertEqual(stats["token_count_histogram"], {1: 1, 2: 1})
        self.assertEqual([top["count"] for top in stats["top_token_counts"]], [2, 1])
        self.assertEqual(stats["num_used_tokens"], 2)
        self.assertAlmostEqual(
            sum(top["frequency"] for top in stats["top_token_counts"]),
            1.0,
        )
        self.assertAlmostEqual(stats["vocab_coverage"], stats["num_used_tokens"] / bpe.vocab_size)
        self.assertGreater(stats["entropy"], 0.0)
        self.assertEqual(stats["used_token_length_counts"], (0, 0, 1, 2))
        self.assertEqual(stats["used_token_length_frequencies"], (0.0, 0.0, 1 / 3, 2 / 3))
        self.assertEqual(sum(stats["vocab_token_length_counts"]), bpe.vocab_size)
        self.assertEqual(stats["mean_used_token_length"], 8 / 3)
        self.assertEqual(stats["max_used_token_length"], 3)
        self.assertEqual(stats["used_token_length_quantiles"]["p50"], 3.0)

    def test_eval_limits_top_token_counts(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=18,
        )

        stats = bpe.evaluate(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            show_progress=False,
            top_k=1,
        )

        self.assertEqual(len(stats["top_token_counts"]), 1)
        self.assertEqual(stats["top_token_counts"][0]["count"], 2)

    def test_eval_rejects_invalid_top_k(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "top_k"):
            bpe.evaluate([[[1], [2], [3]]], show_progress=False, top_k=-1)
        for top_k in (True, 1.0, "1"):
            with (
                self.subTest(top_k=top_k),
                self.assertRaisesRegex(TypeError, "top_k must be an integer"),
            ):
                bpe.evaluate(
                    [[[1], [2], [3]]],
                    show_progress=False,
                    top_k=cast(Any, top_k),
                )
        with self.assertRaisesRegex(TypeError, "show_progress must be a bool"):
            bpe.evaluate(
                [[[1], [2], [3]]],
                show_progress=cast(Any, 1),
            )

    def test_eval_progress_keeps_same_result(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=False,
        )
        corpus = [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

        plain = bpe.evaluate(corpus, show_progress=False)
        with_progress = bpe.evaluate(corpus, show_progress=True)

        self.assertEqual(with_progress, plain)

    def test_eval_progress_does_not_render_dynamic_bar_non_interactive(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=False,
        )

        with (
            patch("anytrain.audio.bpe._core.sys.stderr.isatty", return_value=False),
            patch("anytrain.audio.bpe._core.STATIC_PROGRESS_INTERVAL_SECONDS", 0),
            patch("builtins.print") as output,
            patch("tqdm.auto.tqdm") as tqdm,
        ):
            bpe.evaluate([[[1], [2], [3]]], show_progress=True)

        tqdm.assert_not_called()
        output.assert_any_call(
            "CodecBPE eval: 1 sequences processed",
            file=sys.stderr,
            flush=True,
        )

    def test_eval_rejects_empty_corpus(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "corpus"):
            bpe.evaluate([], show_progress=False)


if __name__ == "__main__":
    unittest.main()
