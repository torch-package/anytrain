import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anytrain.audio.codec.unicodec.assets as unicodec_assets
from anytrain.audio.codec.unicodec.assets import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_HF_REPO_ID,
    ensure_unicodec_assets,
)


class UniCodecAssetsTest(unittest.TestCase):
    def test_ensure_assets_downloads_checkpoint_and_uses_packaged_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("model: {}")

            def fake_download(
                *,
                repo_id,
                filename,
                local_dir,
                local_files_only,
                force_download,
                revision,
            ):
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(repo_id)
                return str(path)

            with (
                patch(
                    "anytrain.audio.codec._cache.require_hf_function",
                    return_value=fake_download,
                ),
                patch.object(unicodec_assets, "_default_config_path", return_value=config),
            ):
                assets = ensure_unicodec_assets(cache_dir=root, revision="abc123")

            self.assertEqual(assets["cache_dir"], root)
            self.assertEqual(assets["config"], config)
            self.assertEqual(assets["checkpoint"], root / DEFAULT_CHECKPOINT_FILENAME)
            self.assertEqual(assets["checkpoint"].read_text(), DEFAULT_HF_REPO_ID)

    def test_ensure_assets_reuses_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / DEFAULT_CHECKPOINT_FILENAME
            checkpoint.write_text("existing")
            config = root / "config.yaml"
            config.write_text("model: {}")

            with (
                patch(
                    "anytrain.audio.codec._cache.require_hf_function",
                    side_effect=AssertionError("unexpected download"),
                ),
                patch.object(unicodec_assets, "_default_config_path", return_value=config),
            ):
                assets = ensure_unicodec_assets(cache_dir=root)

            self.assertEqual(assets["checkpoint"], checkpoint)
            self.assertEqual(assets["checkpoint"].read_text(), "existing")

    def test_ensure_assets_requires_revision_for_remote_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("model: {}")
            with (
                patch.object(unicodec_assets, "_default_config_path", return_value=config),
                self.assertRaisesRegex(ValueError, "pinned Hugging Face revision"),
            ):
                ensure_unicodec_assets(cache_dir=root)


if __name__ == "__main__":
    unittest.main()
