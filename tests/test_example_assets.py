import unittest
from pathlib import Path
from unittest import mock

import torch


class ExampleAssetsTest(unittest.TestCase):
    def test_paths_exist(self):
        from anytrain.example.resources import (
            color_your_night_path,
            vctk_path,
        )

        speech = vctk_path()
        music = color_your_night_path()

        self.assertEqual(speech.suffix, ".flac")
        self.assertEqual(music.suffix, ".mp3")
        self.assertTrue(speech.is_file())
        self.assertTrue(music.is_file())
        self.assertGreater(speech.stat().st_size, 1024)
        self.assertGreater(music.stat().st_size, 1024)

    def test_list_example_audio(self):
        from anytrain.example.resources import (
            ExampleAudio,
            list_example_audio,
        )

        self.assertEqual(
            list_example_audio(),
            (ExampleAudio.VCTK, ExampleAudio.COLOR_YOUR_NIGHT),
        )

    def test_package_exports_only_example_loaders(self):
        import anytrain.example as example

        self.assertEqual(example.__all__, ["color_your_night", "vctk"])

    def test_unknown_audio_name_raises(self):
        from anytrain.example.resources import example_audio_path

        with self.assertRaisesRegex(ValueError, "Unknown example audio"):
            example_audio_path("missing")

    def test_fallback_to_torchaudio_warns(self):
        from anytrain.example.audio import load_example_audio

        waveform = torch.zeros(1, 16)
        with (
            mock.patch(
                "anytrain.example.audio._load_audio_with_torchcodec",
                side_effect=RuntimeError("codec unavailable"),
            ),
            mock.patch(
                "anytrain.example.audio._load_audio_with_torchaudio",
                return_value=(waveform, 48000),
            ) as torchaudio_loader,
            self.assertWarnsRegex(RuntimeWarning, "falling back to torchaudio"),
        ):
            loaded, sample_rate = load_example_audio(Path("dummy.wav"), duration=0.1)

        self.assertIs(loaded, waveform)
        self.assertEqual(sample_rate, 48000)
        torchaudio_loader.assert_called_once()

    def test_all_decoders_fail_with_install_hint(self):
        from anytrain.example.audio import load_example_audio

        with (
            mock.patch(
                "anytrain.example.audio._load_audio_with_torchcodec",
                side_effect=RuntimeError("codec unavailable"),
            ),
            mock.patch(
                "anytrain.example.audio._load_audio_with_torchaudio",
                side_effect=RuntimeError("torchaudio unavailable"),
            ),
            self.assertWarns(RuntimeWarning),
            self.assertRaisesRegex(
                RuntimeError,
                r'pip install "torchcodec>=0.2" "torchaudio>=2.0"',
            ),
        ):
            load_example_audio(Path("dummy.wav"), duration=0.1)

    def test_invalid_time_range_raises_before_decoder(self):
        from anytrain.example.audio import load_example_audio

        with self.assertRaisesRegex(ValueError, "start_seconds"):
            load_example_audio(Path("dummy.wav"), start_seconds=-1)

        with self.assertRaisesRegex(ValueError, "duration"):
            load_example_audio(Path("dummy.wav"), duration=0)

    def test_audio_range_error_does_not_fallback(self):
        from anytrain.example.audio import (
            _AudioRangeError,
            load_example_audio,
        )

        with (
            mock.patch(
                "anytrain.example.audio._load_audio_with_torchcodec",
                side_effect=_AudioRangeError("range overflow"),
            ),
            mock.patch("anytrain.example.audio._load_audio_with_torchaudio") as torchaudio_loader,
            self.assertRaisesRegex(ValueError, "range overflow"),
        ):
            load_example_audio(Path("dummy.wav"), duration=0.1)

        torchaudio_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
