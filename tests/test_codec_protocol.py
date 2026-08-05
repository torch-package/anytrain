from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
import torch

import anytrain.codec as codec_api
from anytrain.codec import (
    AcousticLayout,
    CodecCapability,
    SemanticAcousticCodecSpec,
    SemanticAcousticCodes,
    SemanticGlobalCodecSpec,
    SemanticGlobalCodes,
    capabilities,
    load_frame,
    load_semantic_acoustic,
    load_semantic_global,
    masked_acoustic_features,
    semantic_acoustic_spec,
    semantic_global_spec,
)
from anytrain.codec._module import validate_frame_codes


def test_codec_enums_have_stable_string_values() -> None:
    assert AcousticLayout.FRAME_ALIGNED.value == "frame_aligned"
    assert AcousticLayout.FIXED_LENGTH.value == "fixed_length"
    assert CodecCapability.FRAME_CODES.value == "frame_codes"
    assert CodecCapability.SEMANTIC_ACOUSTIC.value == "semantic_acoustic"
    assert CodecCapability.SEMANTIC_GLOBAL.value == "semantic_global"


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


def test_semantic_global_codes_preserve_tensor_boundaries() -> None:
    semantic = torch.zeros((2, 3, 1), dtype=torch.long)
    global_codes = torch.ones((2, 4, 2), dtype=torch.long)
    codes = SemanticGlobalCodes(semantic=semantic, global_codes=global_codes)

    assert codes.semantic is semantic
    assert codes.global_codes is global_codes
    assert not hasattr(codes, "acoustic")
    with pytest.raises(FrozenInstanceError):
        codes.global_codes = semantic  # type: ignore[misc]


class _FeatureCodec:
    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return acoustic_codes.float().sum(dim=-1, keepdim=True)


def test_masked_acoustic_features_zeroes_padded_units() -> None:
    codes = torch.tensor([[[1, 2], [9, 9]]])
    mask = torch.tensor([[True, False]])

    features = masked_acoustic_features(_FeatureCodec(), codes, mask)  # type: ignore[arg-type]

    torch.testing.assert_close(features, torch.tensor([[[3.0], [0.0]]]))


def test_masked_acoustic_features_validation_can_be_skipped() -> None:
    codes = torch.tensor([[[1, 2], [9, 9]]])
    mask = torch.tensor([[True, False]])

    features = masked_acoustic_features(
        _FeatureCodec(),
        codes,
        mask,
        validate=False,
    )  # type: ignore[arg-type]

    torch.testing.assert_close(features, torch.tensor([[[3.0], [0.0]]]))


def test_structured_codec_protocol_is_exported() -> None:
    assert codec_api.SemanticAcousticCodec.__name__ == "SemanticAcousticCodec"
    assert codec_api.SemanticGlobalCodec.__name__ == "SemanticGlobalCodec"


class _StructuredCodec:
    name = "fake"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    semantic_codebook_sizes = (4,)
    semantic_codebook = torch.zeros((4, 3))
    acoustic_codebook_sizes = (5, 7)
    acoustic_feature_dim = 6
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None


class _GlobalCodec:
    name = "fake-global"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    semantic_codebook_sizes = (4,)
    semantic_codebook = torch.zeros((4, 3))
    global_codebook_sizes = (5, 7)
    global_feature_dim = 6
    global_unit_length = 32


def test_semantic_acoustic_spec_inspects_backend_structure() -> None:
    spec = semantic_acoustic_spec(_StructuredCodec())  # type: ignore[arg-type]

    assert spec == SemanticAcousticCodecSpec(
        sample_rate=16_000,
        frame_rate=50.0,
        semantic_frame_rate=50.0,
        semantic_codebook_sizes=(4,),
        semantic_embedding_dim=3,
        acoustic_codebook_sizes=(5, 7),
        acoustic_feature_dim=6,
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        acoustic_unit_length=None,
    )


def test_semantic_acoustic_spec_requires_codebook_vocab_match() -> None:
    codec = _StructuredCodec()
    codec.semantic_codebook = torch.zeros((3, 3))

    with pytest.raises(ValueError, match="vocab axis"):
        semantic_acoustic_spec(codec)  # type: ignore[arg-type]


def test_semantic_acoustic_spec_requires_fixed_unit_length() -> None:
    with pytest.raises(ValueError, match="requires acoustic_unit_length"):
        SemanticAcousticCodecSpec(
            sample_rate=16_000,
            frame_rate=50.0,
            semantic_frame_rate=50.0,
            semantic_codebook_sizes=(4,),
            semantic_embedding_dim=3,
            acoustic_codebook_sizes=(5,),
            acoustic_feature_dim=6,
            acoustic_layout=AcousticLayout.FIXED_LENGTH,
            acoustic_unit_length=None,
        )


def test_semantic_global_spec_inspects_backend_structure() -> None:
    spec = semantic_global_spec(_GlobalCodec())  # type: ignore[arg-type]

    assert spec == SemanticGlobalCodecSpec(
        sample_rate=16_000,
        frame_rate=50.0,
        semantic_frame_rate=50.0,
        semantic_codebook_sizes=(4,),
        semantic_embedding_dim=3,
        global_codebook_sizes=(5, 7),
        global_feature_dim=6,
        global_unit_length=32,
    )


def test_semantic_global_spec_requires_positive_unit_length() -> None:
    with pytest.raises(ValueError, match="global_unit_length must be positive"):
        SemanticGlobalCodecSpec(
            sample_rate=16_000,
            frame_rate=50.0,
            semantic_frame_rate=50.0,
            semantic_codebook_sizes=(4,),
            semantic_embedding_dim=3,
            global_codebook_sizes=(5,),
            global_feature_dim=6,
            global_unit_length=0,
        )


@pytest.mark.parametrize(
    "codes",
    [
        torch.tensor([[[-1, 0]]]),
        torch.tensor([[[0, 3]]]),
    ],
)
def test_frame_code_validation_rejects_ids_outside_each_codebook(codes: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="outside the configured codebook ranges"):
        validate_frame_codes(codes, (2, 3))


def test_frame_code_validation_requires_positive_integer_codebook_sizes() -> None:
    codes = torch.zeros((1, 1, 1), dtype=torch.long)

    with pytest.raises(TypeError, match="must be an integer"):
        validate_frame_codes(codes, (True,))
    with pytest.raises(ValueError, match="must be positive"):
        validate_frame_codes(codes, (0,))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.uint32, torch.uint64])
def test_frame_code_validation_requires_signed_ids(dtype: torch.dtype) -> None:
    codes = torch.zeros((1, 1, 1), dtype=dtype)

    with pytest.raises(TypeError, match="signed integer ids"):
        validate_frame_codes(codes, (2,))


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
                CodecCapability.SEMANTIC_GLOBAL,
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


def test_semantic_acoustic_loader_rejects_semantic_global_codec() -> None:
    with pytest.raises(ValueError, match="does not provide.*semantic_acoustic"):
        load_semantic_acoustic("bicodec", device="cpu")


def test_semantic_global_loader_rejects_semantic_acoustic_codec() -> None:
    with pytest.raises(ValueError, match="does not provide.*semantic_global"):
        load_semantic_global("longcat", device="cpu")


def test_semantic_global_loader_uses_lazy_bicodec_loader() -> None:
    backend = object()
    with patch("anytrain.codec.bicodec.BiCodec.from_pretrained", return_value=backend) as load:
        loaded = load_semantic_global("bicodec", device="cpu")

    assert loaded is backend
    load.assert_called_once_with(
        device="cpu",
        revision="642071559bfc6346c2359d19dcb6be3f9dd8a05d",
    )


def test_frame_loader_rejects_structured_only_codec_before_loading() -> None:
    with pytest.raises(ValueError, match="does not provide.*frame_codes"):
        load_frame("bicodec", device="cpu")


def test_frame_loader_uses_lazy_longcat_loader() -> None:
    backend = object()
    with patch("anytrain.codec.longcat.LongCat.from_pretrained", return_value=backend) as load:
        loaded = load_frame("longcat", device="cpu")

    assert loaded is backend
    load.assert_called_once_with(device="cpu")
