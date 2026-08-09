from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from anytrain._compat import StrEnum


class ExampleAudio(StrEnum):
    VCTK = "vctk"
    COLOR_YOUR_NIGHT = "color_your_night"


def list_example_audio() -> tuple[ExampleAudio, ...]:
    return tuple(ExampleAudio)


def example_audio_path(name: ExampleAudio | str) -> Path:
    audio = _audio(name)
    resource = files("anytrain.example").joinpath(*_relative_parts(audio))
    return Path(str(resource))


def vctk_path() -> Path:
    return example_audio_path(ExampleAudio.VCTK)


def color_your_night_path() -> Path:
    return example_audio_path(ExampleAudio.COLOR_YOUR_NIGHT)


def _audio(name: ExampleAudio | str) -> ExampleAudio:
    if isinstance(name, ExampleAudio):
        return name
    if not isinstance(name, str):
        raise TypeError("name must be an ExampleAudio or string.")
    try:
        return ExampleAudio(name)
    except ValueError as exc:
        supported = ", ".join(item.value for item in ExampleAudio)
        raise ValueError(f"Unknown example audio {name!r}. Supported: {supported}.") from exc


def _relative_parts(name: ExampleAudio) -> tuple[str, ...]:
    if name == ExampleAudio.VCTK:
        return ("assets", "speech", "p225_001_mic1.flac")
    if name == ExampleAudio.COLOR_YOUR_NIGHT:
        return ("assets", "music", "color_your_night.mp3")
    raise ValueError(f"Unknown example audio {name!r}.")
