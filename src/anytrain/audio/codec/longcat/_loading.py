"""Optional upstream imports and checkpoint environment during model loading."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


def load_loaders():
    try:
        from longcat_audio_codec import (  # pyright: ignore[reportMissingImports]
            load_decoder,
            load_encoder,
        )
    except ImportError as exc:
        raise ImportError(
            "LongCat requires the installable LongCat fork. Install it with "
            '`python -m pip install "longcat-audio-codec @ git+https://github.com/rookiejune/LongCat-Audio-Codec.git"`.'
        ) from exc
    return load_encoder, load_decoder


@contextmanager
def checkpoint_env(ckpt_dir: Path):
    name = "LONGCAT_AUDIO_CODEC_CKPT_DIR"
    old = os.environ.get(name)
    os.environ[name] = str(ckpt_dir)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old
