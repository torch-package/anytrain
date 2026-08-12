import unittest

import anytrain
import anytrain.audio
import anytrain.chat
import anytrain.checkpoint
import anytrain.evaluator
import anytrain.example
import anytrain.framework
import anytrain.lightning
import anytrain.loss
import anytrain.module
import anytrain.observation
import anytrain.optim
import anytrain.translation
import anytrain.tts


class PackageRootTest(unittest.TestCase):
    def test_root_is_an_empty_lightweight_namespace(self) -> None:
        self.assertEqual(anytrain.__all__, [])
        self.assertFalse(hasattr(anytrain, "LightningModule"))
        self.assertFalse(hasattr(anytrain, "run_train"))

    def test_top_level_package_exports_are_unique_and_resolvable(self) -> None:
        packages = (
            anytrain.audio,
            anytrain.chat,
            anytrain.checkpoint,
            anytrain.evaluator,
            anytrain.example,
            anytrain.framework,
            anytrain.lightning,
            anytrain.loss,
            anytrain.module,
            anytrain.observation,
            anytrain.optim,
            anytrain.translation,
            anytrain.tts,
        )
        for package in packages:
            with self.subTest(package=package.__name__):
                self.assertEqual(len(package.__all__), len(set(package.__all__)))
                for name in package.__all__:
                    self.assertTrue(hasattr(package, name), name)


if __name__ == "__main__":
    unittest.main()
