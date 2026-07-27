from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
import torch

import anytrain.codec as codec_api
from anytrain.codec import (
    AcousticLayout,
    CodecCapability,
    SemanticAcousticCodes,
    capabilities,
    load_frame,
    load_semantic_acoustic,
)


def test_codec_enums_have_stable_string_values() -> None:
    assert AcousticLayout.FRAME_ALIGNED.value == "frame_aligned"
    assert AcousticLayout.FIXED_LENGTH.value == "fixed_length"
    assert CodecCapability.FRAME_CODES.value == "frame_codes"
    assert CodecCapability.SEMANTIC_ACOUSTIC.value == "semantic_acoustic"


def test_frame_codec_has_no_legacy_codec_alias() -> None:
    assert not hasattr(codec_api, "Codec")


def test_semantic_acoustic_codes_preserve_tensor_boundaries() -> None:
    semantic = torch.zeros((2, 3, 1), dtype=torch.long)
    acoustic = torch.ones((2, 4, 2), dtype=torch.long)
    codes = SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    assert codes.semantic is semantic
    assert codes.acoustic is acoustic
    with pytest.raises(FrozenInstanceError):
        codes.semantic = acoustic  # type: ignore[misc]


def test_structured_codec_protocol_is_exported() -> None:
    assert codec_api.SemanticAcousticCodec.__name__ == "SemanticAcousticCodec"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dac", {CodecCapability.FRAME_CODES}),
        ("stable_codec", {CodecCapability.FRAME_CODES}),
        ("unicodec", {CodecCapability.FRAME_CODES}),
        (
            "longcat",
            {
                CodecCapability.FRAME_CODES,
                CodecCapability.SEMANTIC_ACOUSTIC,
            },
        ),
        (
            "bicodec",
            {
                CodecCapability.SEMANTIC_ACOUSTIC,
            },
        ),
    ],
)
def test_registry_classifies_codecs_without_loading(
    name: str,
    expected: set[CodecCapability],
) -> None:
    assert capabilities(name) == frozenset(expected)


def test_registry_rejects_unknown_codec() -> None:
    with pytest.raises(ValueError, match="Unknown codec 'missing'"):
        capabilities("missing")


def test_semantic_acoustic_loader_rejects_frame_only_codec_before_loading() -> None:
    with pytest.raises(ValueError, match="does not provide.*semantic_acoustic"):
        load_semantic_acoustic("dac", device="cpu")


def test_semantic_acoustic_loader_uses_lazy_longcat_loader() -> None:
    backend = object()
    with patch("anytrain.codec.longcat.LongCat.from_pretrained", return_value=backend) as load:
        loaded = load_semantic_acoustic("longcat", device="cpu")

    assert loaded is backend
    load.assert_called_once_with(device="cpu")


def test_semantic_acoustic_loader_uses_lazy_bicodec_loader() -> None:
    backend = object()
    with patch("anytrain.codec.bicodec.BiCodec.from_pretrained", return_value=backend) as load:
        loaded = load_semantic_acoustic("bicodec", device="cpu")

    assert loaded is backend
    load.assert_called_once_with(device="cpu")


def test_frame_loader_rejects_structured_only_codec_before_loading() -> None:
    with pytest.raises(ValueError, match="does not provide.*frame_codes"):
        load_frame("bicodec", device="cpu")


def test_frame_loader_uses_lazy_longcat_loader() -> None:
    backend = object()
    with patch("anytrain.codec.longcat.LongCat.from_pretrained", return_value=backend) as load:
        loaded = load_frame("longcat", device="cpu")

    assert loaded is backend
    load.assert_called_once_with(device="cpu")
