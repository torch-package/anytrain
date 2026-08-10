from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

import anytrain.audio.codec as codec_api
from anytrain.audio.codec import (
    AudioCapability,
    AudioCodeSpec,
    audio_backend_identity,
    audio_capabilities,
    audio_spec,
    can_load_audio_detokenizer,
    can_load_audio_tokenizer,
    clear_audio_backend_cache,
    load_audio_detokenizer,
    load_audio_tokenizer,
)
from anytrain.audio.codec._module import validate_frame_codes
from anytrain.audio.codec.longcat import DEFAULT_LONGCAT_REVISION
from anytrain.audio.detokenizer import AudioDetokenizer


@pytest.fixture(autouse=True)
def _isolated_audio_backend_cache():
    clear_audio_backend_cache()
    yield
    clear_audio_backend_cache()


def test_audio_capability_values_are_stable() -> None:
    assert AudioCapability.TOKENIZER.value == "tokenizer"
    assert AudioCapability.DETOKENIZER.value == "detokenizer"
    assert codec_api.__all__ == [
        "AudioBackendIdentity",
        "AudioCapability",
        "AudioCodeSpec",
        "audio_backend_identity",
        "audio_capabilities",
        "audio_spec",
        "can_load_audio_detokenizer",
        "can_load_audio_tokenizer",
        "clear_audio_backend_cache",
        "load_audio_detokenizer",
        "load_audio_tokenizer",
    ]


def test_audio_specs_are_static_named_stream_inventories() -> None:
    assert audio_spec("bicodec") == AudioCodeSpec(
        names=("semantic", "global"),
        vocab_sizes=(8192, 4096),
        sample_rate=16_000,
        frame_rates=(50.0, None),
    )
    assert audio_spec("longcat").names == ("semantic", "rvq_0", "rvq_1", "rvq_2")
    assert audio_spec("dac").names == tuple(f"rvq_{index}" for index in range(9))


def test_audio_code_spec_selects_and_excludes_streams() -> None:
    spec = audio_spec("longcat")
    assert spec.select(("rvq_2", "semantic")).names == ("rvq_2", "semantic")
    assert spec.exclude(("semantic",)).names == ("rvq_0", "rvq_1", "rvq_2")


class _ExampleDetokenizer(AudioDetokenizer):
    _code_spec = AudioCodeSpec(
        names=("left", "right"),
        vocab_sizes=(4, 8),
        sample_rate=16_000,
        frame_rates=(50.0, 50.0),
    )

    @property
    def code_spec(self) -> AudioCodeSpec:
        return self._code_spec

    def codes_to_features(self, stream: str, codes: torch.Tensor) -> torch.Tensor:
        return codes.float().unsqueeze(-1) * (1 if stream == "left" else 2)

    def decode_features(self, features: Mapping[str, torch.Tensor | None]) -> torch.Tensor:
        values = self._ordered(features, value_name="features")
        left = values["left"]
        right = values["right"]
        assert left is not None and right is not None
        return left + right


def test_audio_detokenizer_abc_composes_codebook_features() -> None:
    output = _ExampleDetokenizer().detokenize(
        {"left": torch.tensor([[1, 2]]), "right": torch.tensor([[3, 4]])}
    )
    torch.testing.assert_close(output, torch.tensor([[[7.0], [10.0]]]))


def test_audio_detokenizer_abc_requires_exact_stream_inventory() -> None:
    detokenizer = _ExampleDetokenizer()
    with pytest.raises(ValueError, match="missing=.*right"):
        detokenizer.detokenize({"left": torch.tensor([[1]])})
    with pytest.raises(ValueError, match="unexpected=.*extra"):
        detokenizer.detokenize(
            {
                "left": torch.tensor([[1]]),
                "right": torch.tensor([[1]]),
                "extra": torch.tensor([[1]]),
            }
        )


def test_frame_code_validation_remains_backend_internal() -> None:
    with pytest.raises(ValueError, match="outside the configured codebook ranges"):
        validate_frame_codes(torch.tensor([[[-1, 0]]]), (2, 3))


def test_directional_capabilities_and_identity() -> None:
    assert audio_capabilities("glm4") == frozenset({AudioCapability.TOKENIZER})
    assert can_load_audio_tokenizer("glm4")
    assert not can_load_audio_detokenizer("glm4")
    identity = audio_backend_identity("bicodec")
    assert identity.preset == "bicodec"
    assert identity.revision == "642071559bfc6346c2359d19dcb6be3f9dd8a05d"
    assert DEFAULT_LONGCAT_REVISION == "5c2493c6c68959bd9ad4b06cc3df9292fc1d670f"


class _FrameBackend:
    name = "unicodec"
    sample_rate = 24_000
    frame_rate = 75.0

    def __init__(self) -> None:
        self.encoded: tuple[torch.Tensor, int] | None = None
        self.decoded: tuple[torch.Tensor, ...] | None = None

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.encoded = (audio, sample_rate)
        return torch.ones((audio.size(0), 2, 1), dtype=torch.long)

    def codebook_codes_to_features(self, index: int, codes: torch.Tensor) -> torch.Tensor:
        assert index == 0
        return codes.float().unsqueeze(-1)

    def decode_codebook_features(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.decoded = features
        return torch.zeros((features[0].size(0), 1, 4))


def test_directional_views_share_one_backend() -> None:
    backend = _FrameBackend()
    with patch(
        "anytrain.audio.codec.unicodec.UniCodec.from_pretrained", return_value=backend
    ) as load:
        tokenizer = load_audio_tokenizer("unicodec", device="cpu")
        detokenizer = load_audio_detokenizer("unicodec", device=torch.device("cpu"))
    load.assert_called_once_with(device="cpu")
    assert cast(Any, tokenizer).backend is cast(Any, detokenizer).backend is backend
    audio = torch.zeros((2, 1, 8))
    codes = tokenizer.tokenize(audio, 24_000)
    waveform = detokenizer.detokenize(codes)
    assert backend.encoded == (audio, 24_000)
    assert backend.decoded is not None
    torch.testing.assert_close(backend.decoded[0], torch.ones((2, 2, 1)))
    assert waveform.shape == (2, 1, 4)


def test_directional_loader_rejects_missing_direction() -> None:
    with pytest.raises(ValueError, match="does not provide.*detokenizer"):
        load_audio_detokenizer("glm4", device="cpu")
