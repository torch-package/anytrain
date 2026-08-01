from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.codec.bicodec.assets import (
    DEFAULT_HF_REPO_ID,
    SNAPSHOT_PATTERNS,
    ensure_bicodec_assets,
)


class BiCodecAssetsTest(unittest.TestCase):
    def test_ensure_bicodec_assets_downloads_minimal_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "BiCodec").mkdir()
            (root / "wav2vec2-large-xlsr-53").mkdir()
            (root / "config.yaml").touch()
            (root / "BiCodec" / "config.yaml").touch()
            (root / "BiCodec" / "model.safetensors").touch()
            (root / "wav2vec2-large-xlsr-53" / "config.json").touch()

            calls = {}

            def snapshot_download(**kwargs):
                calls.update(kwargs)
                return str(root)

            with patch(
                "anytrain.codec._cache.require_hf_function",
                return_value=snapshot_download,
            ):
                assets = ensure_bicodec_assets(
                    root,
                    revision="abc123",
                    local_files_only=True,
                    force_download=True,
                )

        self.assertEqual(assets["cache_dir"], root)
        self.assertEqual(assets["model_dir"], root)
        self.assertEqual(calls["repo_id"], DEFAULT_HF_REPO_ID)
        self.assertEqual(calls["local_dir"], str(root))
        self.assertEqual(calls["allow_patterns"], SNAPSHOT_PATTERNS)
        self.assertEqual(calls["revision"], "abc123")
        self.assertTrue(calls["local_files_only"])
        self.assertTrue(calls["force_download"])

    def test_ensure_bicodec_assets_rejects_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def snapshot_download(**kwargs):
                return str(root)

            with (
                patch(
                    "anytrain.codec._cache.require_hf_function",
                    return_value=snapshot_download,
                ),
                self.assertRaisesRegex(FileNotFoundError, "incomplete"),
            ):
                ensure_bicodec_assets(root, allow_unpinned_revision=True)


if __name__ == "__main__":
    unittest.main()
