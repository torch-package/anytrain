from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

import anytrain.audio.codec as codec_api
from anytrain.audio.codec import (
    AudioBackendIdentity,
    AudioCapability,
    AudioCodeSpec,
    AudioCodeStreamSpec,
    audio_backend_identity,
    audio_capabilities,
    audio_planning_estimator,
    audio_spec,
    can_load_audio_detokenizer,
    can_load_audio_tokenizer,
    clear_audio_backend_cache,
    codebook_codec,
    codec_frame_rate,
    codec_sample_rate,
    frame_codebook_sizes,
    fsq_level_values,
    fsq_levels,
    fsq_radix_order,
    load_audio_detokenizer,
    load_audio_tokenizer,
    semantic_feature_dim,
)
from anytrain.audio.codec._module import validate_frame_codes
from anytrain.audio.codec._registry import CodecEntry
from anytrain.audio.codec.longcat import DEFAULT_LONGCAT_REVISION
from anytrain.audio.codec.planning import AudioPlanningEstimator
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
        "AudioCodeStreamSpec",
        "AudioPlanningEstimator",
        "CodecBackend",
        "SemanticCodebookCodec",
        "audio_backend_identity",
        "audio_capabilities",
        "audio_planning_estimator",
        "audio_spec",
        "can_load_audio_detokenizer",
        "can_load_audio_tokenizer",
        "clear_audio_backend_cache",
        "codebook_codec",
        "codec_frame_rate",
        "codec_sample_rate",
        "frame_codebook_sizes",
        "fsq_level_values",
        "fsq_levels",
        "fsq_radix_order",
        "load_audio_backend",
        "load_audio_detokenizer",
        "load_audio_tokenizer",
        "semantic_feature_dim",
    ]


def test_audio_planning_protocol_has_a_public_owner() -> None:
    assert codec_api.AudioPlanningEstimator is AudioPlanningEstimator


def test_audio_specs_are_static_named_stream_inventories() -> None:
    assert audio_spec("bicodec") == AudioCodeSpec(
        sample_rate=16_000,
        streams=(
            AudioCodeStreamSpec("semantic", 8192, frame_rate=50.0),
            AudioCodeStreamSpec("global", 4096, fixed_units=32),
        ),
    )
    assert tuple(stream.name for stream in audio_spec("longcat").streams) == (
        "semantic",
        "rvq_0",
        "rvq_1",
        "rvq_2",
    )
    assert tuple(stream.name for stream in audio_spec("dac").streams) == tuple(
        f"rvq_{index}" for index in range(9)
    )


def test_audio_code_spec_selects_and_excludes_streams() -> None:
    spec = audio_spec("longcat")
    assert tuple(
        stream.name for stream in spec.select(("rvq_2", "semantic")).streams
    ) == ("rvq_2", "semantic")
    assert tuple(stream.name for stream in spec.exclude(("semantic",)).streams) == (
        "rvq_0",
        "rvq_1",
        "rvq_2",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "", "vocab_size": 4, "frame_rate": 50.0}, "name"),
        ({"name": "semantic", "vocab_size": 0, "frame_rate": 50.0}, "vocab_size"),
        ({"name": "semantic", "vocab_size": 4}, "frame_rate or fixed_units"),
        (
            {
                "name": "semantic",
                "vocab_size": 4,
                "frame_rate": 50.0,
                "fixed_units": 2,
            },
            "cannot set both",
        ),
    ],
)
def test_audio_code_stream_spec_rejects_invalid_metadata(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        AudioCodeStreamSpec(**kwargs)  # type: ignore[arg-type]


class _CodebookCodec:
    semantic_codebook = torch.ones((8, 3))


class _FsqCodec:
    semantic_feature_dim = 1
    codebook_sizes = (6,)
    fsq_levels = ((2, 3),)
    fsq_level_values = (((-1.0, 1.0), (-1.0, 0.0, 1.0)),)
    fsq_radix_order = "first_fastest"
    sample_rate = 24_000
    frame_rate = 75.0


def test_codec_capability_accessors_validate_public_metadata() -> None:
    codebook = _CodebookCodec()
    codec = _FsqCodec()

    assert codebook_codec(codebook) is codebook
    assert semantic_feature_dim(codebook) == 3
    assert frame_codebook_sizes(codec) == (6,)
    assert fsq_levels(codec) == ((2, 3),)
    assert fsq_level_values(codec) == (((-1.0, 1.0), (-1.0, 0.0, 1.0)),)
    assert fsq_radix_order(codec) == "first_fastest"
    assert codec_sample_rate(codec) == 24_000
    assert codec_frame_rate(codec) == 75.0


def test_codec_capability_accessors_reject_invalid_metadata() -> None:
    codec = _FsqCodec()
    codec.fsq_level_values = (((1.0, -1.0), (-1.0, 0.0, 1.0)),)

    with pytest.raises(ValueError, match="strictly increasing"):
        fsq_level_values(codec)
    with pytest.raises(TypeError, match="sample rate"):
        codec_sample_rate(object())


@pytest.mark.parametrize("levels", [((True, 3),), ((2.0, 3),), ([2, 3],)])
def test_fsq_levels_reject_coercible_non_integer_metadata(levels: object) -> None:
    codec = _FsqCodec()
    codec.fsq_levels = cast(Any, levels)

    with pytest.raises(TypeError, match="fsq_levels"):
        fsq_levels(codec)


@pytest.mark.parametrize("values", [(((False, 1.0), (-1.0, 0.0, 1.0)),), ([(-1.0, 1.0)],)])
def test_fsq_values_reject_coercible_non_real_metadata(values: object) -> None:
    codec = _FsqCodec()
    codec.fsq_level_values = cast(Any, values)

    with pytest.raises(TypeError, match="FSQ|fsq_level_values"):
        fsq_level_values(codec)


class _ExampleDetokenizer(AudioDetokenizer):
    _code_spec = AudioCodeSpec(
        sample_rate=16_000,
        streams=(
            AudioCodeStreamSpec("left", 4, frame_rate=50.0),
            AudioCodeStreamSpec("right", 8, frame_rate=50.0),
        ),
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


@pytest.mark.parametrize("name", [None, 1, True])
def test_registry_queries_require_string_names(name: object) -> None:
    with pytest.raises(TypeError, match="codec name"):
        can_load_audio_tokenizer(cast(Any, name))
    with pytest.raises(TypeError, match="codec name"):
        audio_spec(cast(Any, name))


def test_registry_entry_validates_static_descriptor_ownership() -> None:
    spec = AudioCodeSpec(
        sample_rate=16_000,
        streams=(AudioCodeStreamSpec("semantic", 4, frame_rate=50.0),),
    )
    factory = cast(Any, lambda backend, code_spec, identity: object())

    with pytest.raises(TypeError, match="capabilities"):
        CodecEntry(
            name="example",
            audio_capabilities=cast(Any, {AudioCapability.TOKENIZER}),
            audio_spec=spec,
            tokenizer=factory,
        )
    with pytest.raises(ValueError, match="identity preset"):
        CodecEntry(
            name="example",
            audio_capabilities=frozenset({AudioCapability.TOKENIZER}),
            audio_spec=spec,
            identity=AudioBackendIdentity("different"),
            tokenizer=factory,
        )
    with pytest.raises(TypeError, match="tokenizer factory"):
        CodecEntry(
            name="example",
            audio_capabilities=frozenset({AudioCapability.TOKENIZER}),
            audio_spec=spec,
            tokenizer=cast(Any, object()),
        )


def test_audio_planning_estimators_are_static_versioned_backend_contracts() -> None:
    with patch("anytrain.audio.codec.registry._load_backend") as load_backend:
        bicodec = audio_planning_estimator("bicodec")
        glm4 = audio_planning_estimator("glm4")

    load_backend.assert_not_called()
    assert bicodec.backend_identity == audio_backend_identity("bicodec")
    assert bicodec.version == "bicodec-wav2vec2-convolution-v1"
    assert bicodec.identity == (
        "audio-token-planning-v1:"
        "89ea0721c3d9d0e9034d5f675b29076fb99d92bd63d6a45d95957e8cfd90d796"
    )
    assert glm4.backend_identity == audio_backend_identity("glm4")
    assert glm4.version == "glm4-whisper-vq-pooling-v1"
    assert glm4.identity == (
        "audio-token-planning-v1:"
        "dd934a65ae8ee669d7f74b0a63c9e41c13ed7e7b2d1f59604b1d8107fd505455"
    )


def test_audio_planning_estimators_follow_pinned_length_boundaries() -> None:
    bicodec = audio_planning_estimator("bicodec")
    assert bicodec.estimate(400, 16_000) == 1
    assert bicodec.estimate(719, 16_000) == 1
    assert bicodec.estimate(720, 16_000) == 2
    assert bicodec.estimate(16_000, 16_000) == 49
    assert bicodec.estimate(8_000, 8_000) == 49
    with pytest.raises(ValueError, match="receptive field"):
        bicodec.estimate(399, 16_000)

    glm4 = audio_planning_estimator("glm4")
    assert glm4.estimate(1, 16_000) == 1
    assert glm4.estimate(1_280, 16_000) == 1
    assert glm4.estimate(1_281, 16_000) == 2
    assert glm4.estimate(16_000, 16_000) == 13
    assert glm4.estimate(478_720, 16_000) == 374
    assert glm4.estimate(478_721, 16_000) == 375


@pytest.mark.parametrize(
    ("source_frames", "source_rate"),
    [(0, 16_000), (1, 0), (True, 1)],
)
def test_audio_planning_estimators_reject_invalid_source_metadata(
    source_frames: int,
    source_rate: int,
) -> None:
    with pytest.raises((TypeError, ValueError), match="source_frames|source_rate"):
        audio_planning_estimator("glm4").estimate(source_frames, source_rate)


def test_audio_planning_estimator_requires_a_registered_static_contract() -> None:
    with pytest.raises(ValueError, match="does not provide"):
        audio_planning_estimator("longcat")


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
