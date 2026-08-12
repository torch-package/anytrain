import unittest

import anytrain.lightning as lightning_api
import anytrain.lightning.callback as callback_api


class LightningAPITest(unittest.TestCase):
    def test_public_exports_are_unique_and_resolvable(self) -> None:
        for module in (lightning_api, callback_api):
            with self.subTest(module=module.__name__):
                self.assertEqual(len(module.__all__), len(set(module.__all__)))
                for name in module.__all__:
                    self.assertTrue(hasattr(module, name), name)

    def test_root_callbacks_share_the_callback_package_owner(self) -> None:
        callback_names = {
            "DebugCallback",
            "EMACallback",
            "GradientProbeLoggerCallback",
            "LossSummaryCallback",
            "ModelCheckpoint",
            "ObservationCallback",
            "ParameterPolicyCallback",
            "PerformanceCallback",
        }

        for name in callback_names:
            with self.subTest(name=name):
                self.assertIn(name, callback_api.__all__)
                self.assertIs(getattr(lightning_api, name), getattr(callback_api, name))


if __name__ == "__main__":
    unittest.main()
