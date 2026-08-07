from __future__ import annotations

from typing import Any

INSTALL_HINT = (
    "Install BPE dependencies with "
    '`python -m pip install "anytrain[bpe]"` '
    'or `python -m pip install "tokenizers>=0.20"`.'
)


def bpe_class() -> type[Any]:
    try:
        from tokenizers.models import BPE
    except ImportError as exc:
        raise ImportError(f"`CodecBPE` requires `tokenizers`. {INSTALL_HINT}") from exc
    return BPE


def training_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise ImportError(f"`CodecBPE` requires `tokenizers`. {INSTALL_HINT}") from exc
    return Tokenizer, BPE, BpeTrainer
