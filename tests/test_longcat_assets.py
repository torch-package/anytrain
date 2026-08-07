import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anytrain.audio.codec.longcat.assets as longcat_assets
from anytrain.audio.codec.longcat.assets import (
    CHECKPOINT_FILENAMES,
    DECODER_CONFIG_KEYS,
    ensure_longcat_assets,
)

YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(YAML_AVAILABLE, "PyYAML is not installed")
class LongCatAssetsTest(unittest.TestCase):
    def test_ensure_assets_downloads_checkpoints_and_writes_patched_configs(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

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

            def fake_config(stem):
                return {"codec_config": {"name": stem, "ckpt_path": "ckpts/old.pt"}}

            with (
                patch(
                    "anytrain.audio.codec._cache.require_hf_function",
                    return_value=fake_download,
                ),
                patch.object(longcat_assets, "_read_default_config", side_effect=fake_config),
            ):
                assets = ensure_longcat_assets(cache_dir=root, revision="abc123")

            self.assertEqual(assets.cache_dir, root)
            self.assertEqual(assets.ckpt_dir, root / "ckpts")
            self.assertEqual(set(assets.checkpoints), set(CHECKPOINT_FILENAMES))

            for key, filename in CHECKPOINT_FILENAMES.items():
                self.assertEqual(assets.checkpoints[key], root / "ckpts" / filename)
                self.assertTrue(assets.checkpoints[key].exists())

            encoder_config = yaml.safe_load(assets.configs.encoder.read_text())
            self.assertEqual(
                encoder_config["codec_config"]["ckpt_path"],
                str(assets.checkpoints["encoder"]),
            )

            for decoder, key in DECODER_CONFIG_KEYS.items():
                decoder_config = yaml.safe_load(assets.configs.decoder(decoder).read_text())
                self.assertEqual(
                    decoder_config["codec_config"]["ckpt_path"],
                    str(assets.checkpoints[key]),
                )

    def test_ensure_assets_only_requires_requested_decoders(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = []

            def fake_download(
                *,
                repo_id,
                filename,
                local_dir,
                local_files_only,
                force_download,
                revision,
            ):
                downloads.append(filename)
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(repo_id)
                return str(path)

            def fake_config(stem):
                return {"codec_config": {"name": stem, "ckpt_path": "ckpts/old.pt"}}

            with (
                patch(
                    "anytrain.audio.codec._cache.require_hf_function",
                    return_value=fake_download,
                ),
                patch.object(longcat_assets, "_read_default_config", side_effect=fake_config),
            ):
                assets = ensure_longcat_assets(
                    cache_dir=root,
                    decoders=("16k_4codebooks",),
                    revision="abc123",
                )

            self.assertEqual(
                set(assets.checkpoints),
                {"encoder", "encoder_cmvn", "decoder_16k_4codebooks"},
            )
            self.assertEqual(
                set(downloads),
                {
                    "ckpts/LongCatAudioCodec_encoder.pt",
                    "ckpts/LongCatAudioCodec_encoder_cmvn.npy",
                    "ckpts/LongCatAudioCodec_decoder_16k_4codebooks.pt",
                },
            )
            self.assertTrue(assets.configs.encoder.exists())
            self.assertTrue(assets.configs.decoder_16k_4codebooks.exists())
            self.assertFalse(assets.configs.decoder_24k_2codebooks.exists())
            self.assertFalse(assets.configs.decoder_24k_4codebooks.exists())

            decoder_config = yaml.safe_load(assets.configs.decoder_16k_4codebooks.read_text())
            self.assertEqual(
                decoder_config["codec_config"]["ckpt_path"],
                str(assets.checkpoints["decoder_16k_4codebooks"]),
            )


if __name__ == "__main__":
    unittest.main()
