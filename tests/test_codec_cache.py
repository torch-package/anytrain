import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.codec._cache import anytrain_cache_dir, hf_cache_dir
from anytrain.env import (
    ANYTRAIN_HOME_ENV,
    DEFAULT_ANYTRAIN_HOME,
    DEFAULT_HF_HOME,
    HF_HOME_ENV,
)


class CodecCacheTest(unittest.TestCase):
    def test_explicit_cache_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertEqual(hf_cache_dir("codec", root / "hf"), root / "hf")
            self.assertEqual(anytrain_cache_dir("codec", root / "anytrain"), root / "anytrain")

    def test_environment_controls_default_cache_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                ANYTRAIN_HOME_ENV: str(root / "anytrain"),
                HF_HOME_ENV: str(root / "hf"),
            }

            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(hf_cache_dir("codec"), root / "hf" / "codec")
                self.assertEqual(
                    anytrain_cache_dir("codec"),
                    root / "anytrain" / "codec",
                )

    def test_packaged_defaults_are_used_without_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(hf_cache_dir("codec"), DEFAULT_HF_HOME / "codec")
            self.assertEqual(
                anytrain_cache_dir("codec"),
                DEFAULT_ANYTRAIN_HOME / "codec",
            )

    def test_empty_environment_values_fail(self):
        with (
            patch.dict(os.environ, {HF_HOME_ENV: ""}, clear=True),
            self.assertRaisesRegex(ValueError, HF_HOME_ENV),
        ):
            hf_cache_dir("codec")
        with (
            patch.dict(os.environ, {ANYTRAIN_HOME_ENV: ""}, clear=True),
            self.assertRaisesRegex(ValueError, ANYTRAIN_HOME_ENV),
        ):
            anytrain_cache_dir("codec")


if __name__ == "__main__":
    unittest.main()
