from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import anytrain.audio.codec.dac.assets as dac_assets
from anytrain.audio.codec.dac import ensure_dac_assets


class DACAssetsTest(unittest.TestCase):
    def test_ensure_assets_downloads_to_explicit_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_download(url: str, target: Path, **kwargs) -> None:
                target.write_text(url)

            with patch.object(dac_assets, "_download", side_effect=fake_download):
                assets = ensure_dac_assets(
                    cache_dir=root,
                    model_type="24khz",
                    allow_unsafe_download=True,
                )

            self.assertEqual(assets["cache_dir"], root)
            self.assertEqual(assets["tag"], "0.0.4")
            self.assertEqual(
                assets["checkpoint"],
                root / "weights_24khz_8kbps_0.0.4.pth",
            )
            self.assertIn("weights_24khz.pth", assets["checkpoint"].read_text())

    def test_ensure_assets_reuses_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "weights_44khz_8kbps_0.0.1.pth"
            checkpoint.write_text("existing")

            with patch.object(
                dac_assets,
                "_download",
                side_effect=AssertionError("unexpected download"),
            ):
                assets = ensure_dac_assets(cache_dir=root)

            self.assertEqual(assets["checkpoint"], checkpoint)
            self.assertEqual(checkpoint.read_text(), "existing")

    def test_local_files_only_exposes_missing_checkpoint(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(FileNotFoundError, "not available locally"),
        ):
            ensure_dac_assets(cache_dir=tmp, local_files_only=True)

    def test_remote_download_requires_checksum_metadata_by_default(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "SHA256 metadata"),
        ):
            ensure_dac_assets(cache_dir=tmp)

    def test_download_validates_checksum_and_removes_partial_file(self):
        payload = b"bad-checkpoint"

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, size=-1):
                if self.consumed:
                    return b""
                self.consumed = True
                return payload

            consumed = False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "weights.pth"
            with (
                patch.object(dac_assets, "urlopen", return_value=Response()),
                self.assertRaisesRegex(ValueError, "SHA256 mismatch"),
            ):
                dac_assets._download(
                    "https://example.test/weights.pth",
                    target,
                    expected_sha256="0" * 64,
                )

            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_download_writes_file_when_checksum_matches(self):
        payload = b"checkpoint"

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, size=-1):
                if self.consumed:
                    return b""
                self.consumed = True
                return payload

            consumed = False

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.pth"
            expected = sha256(payload).hexdigest()
            with patch.object(dac_assets, "urlopen", return_value=Response()) as urlopen:
                dac_assets._download(
                    "https://example.test/weights.pth",
                    target,
                    expected_sha256=expected,
                    timeout=12.0,
                )

            urlopen.assert_called_once_with("https://example.test/weights.pth", timeout=12.0)
            self.assertEqual(target.read_bytes(), payload)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no latest checkpoint"):
            ensure_dac_assets(model_type="24khz", model_bitrate="16kbps")


if __name__ == "__main__":
    unittest.main()
