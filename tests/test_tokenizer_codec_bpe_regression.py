import tempfile
import unittest

from anytrain.audio.bpe import CodecBPE

try:
    import tokenizers
except ImportError:
    tokenizers = None


def corpus():
    return [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class CodecBPERegressionTest(unittest.TestCase):
    def test_from_pretrained_round_trip(self):
        bpe = CodecBPE.train(corpus, codebook_sizes=(16,), vocab_size=18)

        with tempfile.TemporaryDirectory() as tmp:
            bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

        self.assertEqual(loaded.vocab_size, bpe.vocab_size)
        self.assertEqual(loaded.encode([[1], [2], [3]]), bpe.encode([[1], [2], [3]]))

    def test_progress_keeps_same_result(self):
        plain = CodecBPE.train(corpus, codebook_sizes=(16,), vocab_size=18, show_progress=False)
        with_progress = CodecBPE.train(
            corpus,
            codebook_sizes=(16,),
            vocab_size=18,
            show_progress=True,
        )

        self.assertEqual(with_progress.vocab_size, plain.vocab_size)
        self.assertEqual(
            with_progress.encode([[1], [2], [3]]),
            plain.encode([[1], [2], [3]]),
        )


if __name__ == "__main__":
    unittest.main()
