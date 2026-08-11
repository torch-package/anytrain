from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor

from anytrain.audio.codec import AudioCodeStreamSpec
from anytrain.audio.tokenizer import AudioCodeSpec, AudioTokenizer, PaddedAudioCodes


class _ExampleTokenizer(AudioTokenizer):
    _code_spec = AudioCodeSpec(
        sample_rate=16_000,
        streams=(
            AudioCodeStreamSpec("semantic", 4, frame_rate=50.0),
            AudioCodeStreamSpec("rvq_0", 8, frame_rate=50.0),
        ),
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


def test_padded_audio_codes_accepts_aligned_streams() -> None:
    padded = PaddedAudioCodes(
        codes={
            "rvq_0": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "rvq_1": torch.tensor([[6, 0], [7, 8]]),
        },
        lengths={
            "rvq_0": torch.tensor([2, 3]),
            "rvq_1": torch.tensor([1, 2]),
        },
    )

    assert tuple(padded.codes) == ("rvq_0", "rvq_1")


@pytest.mark.parametrize(
    ("codes", "error", "match"),
    [
        ([[1, 2]], ValueError, "shape"),
        (torch.ones((1, 2), dtype=torch.float32), TypeError, "signed integer IDs"),
        (torch.ones((1, 2, 1), dtype=torch.long), ValueError, "shape"),
        (torch.empty((1, 0), dtype=torch.long), ValueError, "must not be empty"),
    ],
)
def test_padded_audio_codes_rejects_invalid_code_tensor(
    codes: Any,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        PaddedAudioCodes(
            codes={"rvq_0": codes},
            lengths={"rvq_0": torch.tensor([1])},
        )


def test_padded_audio_codes_requires_codes_and_lengths_to_align_on_batch() -> None:
    with pytest.raises(ValueError, match="align on batch"):
        PaddedAudioCodes(
            codes={"rvq_0": torch.ones((2, 1), dtype=torch.long)},
            lengths={"rvq_0": torch.tensor([1])},
        )


def test_padded_audio_code_streams_align_on_batch() -> None:
    with pytest.raises(ValueError, match="streams must align on batch"):
        PaddedAudioCodes(
            codes={
                "rvq_0": torch.ones((2, 1), dtype=torch.long),
                "rvq_1": torch.ones((1, 1), dtype=torch.long),
            },
            lengths={
                "rvq_0": torch.tensor([1, 1]),
                "rvq_1": torch.tensor([1]),
            },
        )


@pytest.mark.parametrize("length", [0, 3])
def test_padded_audio_codes_rejects_invalid_valid_length(length: int) -> None:
    with pytest.raises(ValueError, match="positive and not exceed"):
        PaddedAudioCodes(
            codes={"rvq_0": torch.ones((1, 2), dtype=torch.long)},
            lengths={"rvq_0": torch.tensor([length])},
        )
