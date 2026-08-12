import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anytrain
import anytrain.env as env_api
from anytrain.env import (
    ANYTRAIN_HOME_ENV,
    DEFAULT_ANYTRAIN_HOME,
    HF_HOME_ENV,
    TORCH_HOME_ENV,
    WHISPER_ROOT_ENV,
    anytrain_home,
    hf_home,
    torch_home,
    whisper_root,
)


class EnvTest(unittest.TestCase):
    def test_package_root_does_not_export_env_helpers(self):
        self.assertEqual(anytrain.__all__, [])

    def test_env_public_exports_are_unique_and_resolvable(self):
        self.assertEqual(len(env_api.__all__), len(set(env_api.__all__)))
        for name in env_api.__all__:
            self.assertTrue(hasattr(env_api, name), name)

    def test_anytrain_home_defaults_to_user_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(anytrain_home(), DEFAULT_ANYTRAIN_HOME)

    def test_anytrain_home_env_overrides_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {ANYTRAIN_HOME_ENV: str(root)}, clear=True):
                self.assertEqual(anytrain_home(), root)

    def test_default_cache_envs_are_created_under_anytrain_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {ANYTRAIN_HOME_ENV: str(root)}, clear=True):
                self.assertEqual(hf_home(), root / "huggingface")
                self.assertEqual(torch_home(), root / "torch")
                self.assertEqual(whisper_root(), root / "whisper")
                self.assertEqual(os.environ[HF_HOME_ENV], str(root / "huggingface"))
                self.assertEqual(os.environ[TORCH_HOME_ENV], str(root / "torch"))
                self.assertEqual(os.environ[WHISPER_ROOT_ENV], str(root / "whisper"))

    def test_existing_cache_envs_are_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                ANYTRAIN_HOME_ENV: str(root / "anytrain"),
                HF_HOME_ENV: str(root / "hf"),
                TORCH_HOME_ENV: str(root / "torch"),
                WHISPER_ROOT_ENV: str(root / "whisper"),
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(hf_home(), root / "hf")
                self.assertEqual(torch_home(), root / "torch")
                self.assertEqual(whisper_root(), root / "whisper")

    def test_empty_env_fails(self):
        for name in (ANYTRAIN_HOME_ENV, HF_HOME_ENV, TORCH_HOME_ENV, WHISPER_ROOT_ENV):
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: ""}, clear=True),
                self.assertRaisesRegex(ValueError, name),
            ):
                if name == ANYTRAIN_HOME_ENV:
                    anytrain_home()
                elif name == HF_HOME_ENV:
                    hf_home()
                elif name == TORCH_HOME_ENV:
                    torch_home()
                else:
                    whisper_root()


if __name__ == "__main__":
    unittest.main()
