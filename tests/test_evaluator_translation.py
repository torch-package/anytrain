import unittest

from anytrain.evaluator.translation import (
    CometEvaluator,
    CometLoadOptions,
    CometResult,
    CometSample,
    CometSemanticConfig,
)

_CONFIG = CometSemanticConfig(
    model="Unbabel/wmt22-comet-da",
    revision="0123456789abcdef0123456789abcdef01234567",
    weights_sha256="0" * 64,
    backend_version="2.2.0",
)


class FakeBackend:
    def __init__(self, score: float = 0.75) -> None:
        self.score = score
        self.calls: list[tuple[tuple[CometSample, ...], int]] = []

    def predict(self, samples, *, batch_size: int) -> CometResult:
        frozen = tuple(samples)
        self.calls.append((frozen, batch_size))
        return CometResult(
            tuple(self.score for _ in frozen),
            self.score,
        )


class CometEvaluatorTest(unittest.TestCase):
    def test_evaluate_uses_aligned_source_hypothesis_reference(self) -> None:
        backend = FakeBackend()
        evaluator = CometEvaluator(_CONFIG, backend=backend)

        metrics = evaluator(
            [" source ", "another source"],
            [" output ", "another output"],
            [" reference ", "another reference"],
        )

        self.assertEqual(metrics, {"comet": 0.75})
        self.assertEqual(backend.calls[0][1], 8)
        self.assertEqual(
            backend.calls[0][0],
            (
                CometSample("source", "output", "reference"),
                CometSample("another source", "another output", "another reference"),
            ),
        )

    def test_stateful_compute_scores_one_complete_corpus(self) -> None:
        backend = FakeBackend(0.8)
        evaluator = CometEvaluator(_CONFIG, backend=backend)

        evaluator.update("source one", "output one", "reference one")
        evaluator.update(["source two", "source three"], ["output two", "output three"], [
            "reference two",
            "reference three",
        ])

        self.assertEqual(evaluator.compute(), {"comet": 0.8})
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][0]), 3)
        evaluator.reset()
        with self.assertRaisesRegex(ValueError, "No translation triples"):
            evaluator.compute()

    def test_rejects_misaligned_or_empty_inputs(self) -> None:
        evaluator = CometEvaluator(_CONFIG, backend=FakeBackend())

        with self.assertRaisesRegex(ValueError, "same batch length"):
            evaluator("source", ["output", "extra"], "reference")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            evaluator("source", "", "reference")

    def test_semantic_identity_changes_with_checkpoint_inputs(self) -> None:
        changed = CometSemanticConfig(
            model=_CONFIG.model,
            revision=_CONFIG.revision,
            weights_sha256="1" * 64,
            backend_version=_CONFIG.backend_version,
        )
        self.assertNotEqual(_CONFIG.semantic_id(), changed.semantic_id())
        self.assertEqual(_CONFIG.semantic_id(), _CONFIG.semantic_id())

    def test_backend_result_must_match_segment_count(self) -> None:
        class WrongCountBackend:
            def predict(self, samples, *, batch_size: int) -> CometResult:
                del samples, batch_size
                return CometResult((0.5,), 0.5)

        evaluator = CometEvaluator(_CONFIG, backend=WrongCountBackend())
        with self.assertRaisesRegex(ValueError, "segment score count"):
            evaluator(["one", "two"], ["a", "b"], ["x", "y"])

    def test_custom_backend_is_not_combined_with_load_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "custom backend"):
            CometEvaluator(
                _CONFIG,
                backend=FakeBackend(),
                load_options=CometLoadOptions("checkpoint.ckpt"),
            )

    def test_default_backend_is_lazy_and_validates_checkpoint_on_first_score(self) -> None:
        evaluator = _CONFIG.evaluator(CometLoadOptions("/tmp/missing-comet.ckpt"))
        backend = evaluator.backend
        self.assertIsNone(backend.model)  # type: ignore[attr-defined]
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint does not exist"):
            evaluator("source", "output", "reference")


if __name__ == "__main__":
    unittest.main()
