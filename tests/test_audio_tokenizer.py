from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from anytrain.audio.tokenizer import AudioCodeSpec, AudioTokenizer


class _ExampleTokenizer(AudioTokenizer):
    _code_spec = AudioCodeSpec(
        names=("semantic", "rvq_0"),
        vocab_sizes=(4, 8),
        sample_rate=16_000,
        frame_rates=(50.0, 50.0),
    )

    def __init__(self, output: Mapping[str, Tensor]) -> None:
        self.output = output

    @property
    def code_spec(self) -> AudioCodeSpec:
        return self._code_spec

    def _tokenize(self, audio: Tensor, sample_rate: int) -> Mapping[str, Tensor]:
        del audio, sample_rate
        return self.output


def test_audio_tokenizer_orders_and_validates_named_streams() -> None:
    tokenizer = _ExampleTokenizer(
        {
            "rvq_0": torch.tensor([[3, 4]]),
            "semantic": torch.tensor([[1, 2]]),
        }
    )

    codes = tokenizer.tokenize(torch.zeros((1, 1, 8)), 16_000)

    assert tuple(codes) == ("semantic", "rvq_0")


def test_audio_tokenizer_requires_exact_stream_inventory() -> None:
    tokenizer = _ExampleTokenizer({"semantic": torch.tensor([[1]])})

    with pytest.raises(ValueError, match="missing=.*rvq_0"):
        tokenizer.tokenize(torch.zeros((1, 1, 8)), 16_000)


def test_audio_tokenizer_rejects_invalid_code_ids() -> None:
    tokenizer = _ExampleTokenizer(
        {
            "semantic": torch.tensor([[4]]),
            "rvq_0": torch.tensor([[0]]),
        }
    )

    with pytest.raises(ValueError, match="invalid code ID"):
        tokenizer.tokenize(torch.zeros((1, 1, 8)), 16_000)


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.uint8, torch.uint16, torch.uint32, torch.uint64],
)
def test_audio_tokenizer_requires_signed_integer_streams(dtype: torch.dtype) -> None:
    tokenizer = _ExampleTokenizer(
        {
            "semantic": torch.zeros((1, 2), dtype=dtype),
            "rvq_0": torch.zeros((1, 2), dtype=torch.long),
        }
    )

    with pytest.raises(TypeError, match="signed integer IDs"):
        tokenizer.tokenize(torch.zeros((1, 1, 8)), 16_000)
