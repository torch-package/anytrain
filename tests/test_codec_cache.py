import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.audio.codec._cache import anytrain_cache_dir, hf_cache_dir, hf_file, hf_snapshot
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

    def test_hf_file_requires_revision_for_remote_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "model.bin"

            with self.assertRaisesRegex(ValueError, "pinned Hugging Face revision"):
                hf_file(
                    component="test component",
                    repo_id="org/repo",
                    filename="model.bin",
                    cache_dir=root,
                    target=target,
                    local_files_only=False,
                    force_download=True,
                )

    def test_download_controls_require_bool_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "model.bin"

            for name, value in (
                ("local_files_only", 0),
                ("force_download", 1),
                ("allow_unpinned_revision", "yes"),
            ):
                kwargs = {
                    "component": "test component",
                    "repo_id": "org/repo",
                    "filename": "model.bin",
                    "cache_dir": root,
                    "target": target,
                    "revision": "abc123",
                    "local_files_only": False,
                    "force_download": False,
                    "allow_unpinned_revision": False,
                }
                kwargs[name] = value
                with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                    hf_file(**kwargs)

    def test_hf_file_passes_revision_to_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "model.bin"
            calls = {}

            def hf_hub_download(**kwargs):
                calls.update(kwargs)
                target.write_text("checkpoint")
                return str(target)

            fake_hub = types.SimpleNamespace(hf_hub_download=hf_hub_download)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                downloaded = hf_file(
                    component="test component",
                    repo_id="org/repo",
                    filename="model.bin",
                    cache_dir=root,
                    target=target,
                    revision="abc123",
                    local_files_only=False,
                    force_download=True,
                )

            self.assertEqual(downloaded, target)
            self.assertEqual(calls["repo_id"], "org/repo")
            self.assertEqual(calls["filename"], "model.bin")
            self.assertEqual(calls["local_dir"], str(root))
            self.assertEqual(calls["revision"], "abc123")
            self.assertFalse(calls["local_files_only"])
            self.assertTrue(calls["force_download"])

    def test_hf_snapshot_passes_revision_to_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = {}

            def snapshot_download(**kwargs):
                calls.update(kwargs)
                return str(root)

            fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                snapshot = hf_snapshot(
                    component="test snapshot",
                    repo_id="org/repo",
                    cache_dir=root,
                    allow_patterns=("*.json",),
                    revision="abc123",
                    local_files_only=False,
                    force_download=True,
                )

            self.assertEqual(snapshot, root)
            self.assertEqual(calls["repo_id"], "org/repo")
            self.assertEqual(calls["local_dir"], str(root))
            self.assertEqual(calls["allow_patterns"], ("*.json",))
            self.assertEqual(calls["revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
