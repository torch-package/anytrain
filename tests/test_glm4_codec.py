from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import Tensor, nn

from anytrain.codec import (
    AudioCapability,
    FrameTokenizerBackend,
    audio_capabilities,
    can_load_audio_detokenizer,
    can_load_audio_tokenizer,
    clear_audio_backend_cache,
    load_audio_detokenizer,
    load_audio_tokenizer,
)
from anytrain.codec.glm4 import (
    DEFAULT_GLM4_ARTIFACT,
    DEFAULT_GLM4_REVISION,
    GLM4_SOURCE_ROOT_ENV,
    GLM4Tokenizer,
)
from anytrain.codec.glm4.codec import CODEBOOK_SIZE, _load_runtime, _validate_source_root


@pytest.fixture(autouse=True)
def _isolated_backend_cache():
    clear_audio_backend_cache()
    yield
    clear_audio_backend_cache()


class _FeatureExtractor:
    sampling_rate = 16_000
    hop_length = 160
    n_fft = 400

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, arrays, **kwargs):
        waveform = arrays[0]
        max_length = int(kwargs["max_length"])
        feature_frames = max_length // self.hop_length
        valid_frames = math.ceil(len(waveform) / self.hop_length)
        attention_mask = torch.arange(feature_frames).unsqueeze(0) < valid_frames
        self.calls.append({"arrays": arrays, **kwargs})
        return {
            "input_features": torch.zeros((1, 80, feature_frames)),
            "attention_mask": attention_mask.to(dtype=torch.long),
        }


class _Model(nn.Module):
    def __init__(
        self,
        *,
        first_id: int = 0,
        floating_ids: bool = False,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(pooling_kernel_size=4)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, stride=2, bias=False)
        self.codebook = nn.Embedding(CODEBOOK_SIZE, 2)
        self.first_id = first_id
        self.floating_ids = floating_ids
        self.calls = 0

    def forward(self, input_features: Tensor, attention_mask: Tensor):
        del input_features
        token_frames = attention_mask[:, ::2][:, ::4].shape[1]
        offset = self.first_id + self.calls * 10
        self.calls += 1
        ids = torch.arange(token_frames, device=attention_mask.device) + offset
        if self.floating_ids:
            ids = ids.float()
        return SimpleNamespace(quantized_token_ids=ids.unsqueeze(0))


def _tokenizer(
    *,
    model: _Model | None = None,
    feature_extractor: _FeatureExtractor | None = None,
) -> GLM4Tokenizer:
    return GLM4Tokenizer(
        model if model is not None else _Model(),
        feature_extractor if feature_extractor is not None else _FeatureExtractor(),
        device=torch.device("cpu"),
        strict_contract=False,
    )


def _write_source_tree(root: Path) -> None:
    source = root / "speech_tokenizer"
    source.mkdir(parents=True)
    for name in (
        "configuration_whisper.py",
        "generation_whisper.py",
        "modeling_whisper.py",
    ):
        (source / name).write_text("", encoding="utf-8")


def _assert_runtime_guidance(message: str) -> None:
    assert GLM4_SOURCE_ROOT_ENV in message
    assert "source Git commit is not pinned" in message
    assert "transformers==4.44.1" in message
    assert "BiCodec" in message
    assert "independent GLM producer" in message


def test_glm4_registry_exposes_only_online_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _tokenizer()
    monkeypatch.setenv(GLM4_SOURCE_ROOT_ENV, "/opt/glm-4-voice")

    with patch(
        "anytrain.codec.glm4.codec.GLM4Tokenizer.from_pretrained",
        return_value=backend,
    ) as load:
        first = load_audio_tokenizer("glm4", device="cpu")
        second = load_audio_tokenizer("glm4", device="cpu")

    assert audio_capabilities("glm4") == frozenset({AudioCapability.TOKENIZER})
    assert can_load_audio_tokenizer("glm4")
    assert not can_load_audio_detokenizer("glm4")
    assert first.backend is backend
    assert second.backend is backend
    load.assert_called_once_with(
        source_root="/opt/glm-4-voice",
        revision=DEFAULT_GLM4_REVISION,
        device="cpu",
    )
    with pytest.raises(ValueError, match="does not provide.*detokenizer"):
        load_audio_detokenizer("glm4", device="cpu")


def test_glm4_registry_requires_explicit_source_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(GLM4_SOURCE_ROOT_ENV, raising=False)

    with pytest.raises(RuntimeError, match=GLM4_SOURCE_ROOT_ENV) as error:
        load_audio_tokenizer("glm4", device="cpu")
    _assert_runtime_guidance(str(error.value))


def test_glm4_tokenizer_is_directional_and_exposes_frozen_codebook() -> None:
    tokenizer = _tokenizer()

    assert FrameTokenizerBackend.__name__ == "FrameTokenizerBackend"
    assert tokenizer.name == "glm4"
    assert tokenizer.codebook_sizes == (16_384,)
    assert tokenizer.semantic_codebook.shape == (16_384, 2)
    assert tokenizer.semantic_feature_dim == 2
    assert not tokenizer.semantic_codebook.requires_grad
    assert not hasattr(tokenizer, "decode")


def test_glm4_encode_uses_canonical_single_sample_padding_and_shape() -> None:
    model = _Model()
    feature_extractor = _FeatureExtractor()
    tokenizer = _tokenizer(model=model, feature_extractor=feature_extractor)

    codes = tokenizer.encode(torch.zeros((2, 1, 2_560)), sample_rate=16_000)

    assert codes.dtype is torch.long
    assert codes.shape == (2, 2, 1)
    torch.testing.assert_close(codes[0, :, 0], torch.tensor([0, 1]))
    torch.testing.assert_close(codes[1, :, 0], torch.tensor([10, 11]))
    assert model.calls == 2
    assert len(feature_extractor.calls) == 2
    assert all(call["max_length"] == 3_840 for call in feature_extractor.calls)
    assert all(call["pad_to_multiple_of"] == 1_280 for call in feature_extractor.calls)
    assert all(call["device"] == "cpu" for call in feature_extractor.calls)


def test_glm4_encode_resamples_before_canonical_extraction() -> None:
    tokenizer = _tokenizer()
    audio = torch.zeros((1, 1, 1_280))
    resampled = torch.zeros((1, 1, 2_560))

    with patch("anytrain.codec.glm4.codec.resample", return_value=resampled) as resample:
        codes = tokenizer.encode(audio, sample_rate=8_000)

    assert codes.shape == (1, 2, 1)
    assert resample.call_count == 1
    waveform, sample_rate, target_sample_rate = resample.call_args.args
    torch.testing.assert_close(waveform, audio)
    assert waveform.device.type == "cpu"
    assert waveform.dtype is torch.float32
    assert (sample_rate, target_sample_rate) == (8_000, 16_000)


@pytest.mark.parametrize(
    "audio",
    [
        torch.zeros((1, 2_560)),
        torch.zeros((1, 2, 2_560)),
        torch.zeros((0, 1, 2_560)),
        torch.zeros((1, 1, 0)),
    ],
)
def test_glm4_encode_rejects_noncanonical_waveform_shape(audio: Tensor) -> None:
    tokenizer = _tokenizer()

    with pytest.raises(ValueError, match="mono audio|non-empty"):
        tokenizer.encode(audio, sample_rate=16_000)


def test_glm4_encode_requires_floating_waveform() -> None:
    tokenizer = _tokenizer()

    with pytest.raises(TypeError, match="floating-point"):
        tokenizer.encode(torch.zeros((1, 1, 2_560), dtype=torch.long), 16_000)


def test_glm4_encode_rejects_non_integer_backend_ids() -> None:
    tokenizer = _tokenizer(model=_Model(floating_ids=True))

    with pytest.raises(TypeError, match="integer ids"):
        tokenizer.encode(torch.zeros((1, 1, 2_560)), 16_000)


def test_glm4_encode_rejects_ids_outside_preset_vocab() -> None:
    tokenizer = _tokenizer(model=_Model(first_id=CODEBOOK_SIZE))

    with pytest.raises(ValueError, match="outside the configured codebook ranges"):
        tokenizer.encode(torch.zeros((1, 1, 2_560)), 16_000)


def test_glm4_strict_contract_rejects_model_drift() -> None:
    with pytest.raises(RuntimeError, match="contract drift") as error:
        GLM4Tokenizer(
            _Model(),
            _FeatureExtractor(),
            device=torch.device("cpu"),
        )
    _assert_runtime_guidance(str(error.value))


def test_glm4_from_pretrained_uses_pinned_artifact_without_test_download(
    tmp_path: Path,
) -> None:
    model = _Model()
    feature_extractor = _FeatureExtractor()
    model_from_pretrained = Mock(return_value=model)
    feature_from_pretrained = Mock(return_value=feature_extractor)
    model_loader = SimpleNamespace(from_pretrained=model_from_pretrained)
    feature_loader = SimpleNamespace(from_pretrained=feature_from_pretrained)

    with (
        patch(
            "anytrain.codec.glm4.codec._validate_source_root",
            return_value=tmp_path,
        ),
        patch(
            "anytrain.codec.glm4.codec._load_runtime",
            return_value=(model_loader, feature_loader),
        ),
        patch("anytrain.codec.glm4.codec._require_contract"),
    ):
        tokenizer = GLM4Tokenizer.from_pretrained(
            source_root=tmp_path,
            device="cpu",
            local_files_only=True,
        )

    assert tokenizer.model is model
    assert tokenizer.feature_extractor is feature_extractor
    expected_kwargs = {
        "revision": DEFAULT_GLM4_REVISION,
        "local_files_only": True,
    }
    model_from_pretrained.assert_called_once_with(DEFAULT_GLM4_ARTIFACT, **expected_kwargs)
    feature_from_pretrained.assert_called_once_with(
        DEFAULT_GLM4_ARTIFACT,
        **expected_kwargs,
    )


def test_glm4_source_root_accepts_fork_without_git_metadata(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)

    assert _validate_source_root(tmp_path) == tmp_path.resolve()


def test_glm4_source_root_requires_runtime_files(tmp_path: Path) -> None:
    source = tmp_path / "speech_tokenizer"
    source.mkdir()
    (source / "modeling_whisper.py").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing required files") as error:
        _validate_source_root(tmp_path)
    _assert_runtime_guidance(str(error.value))


def test_glm4_weights_revision_remains_fixed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="weights revision is fixed") as error:
        GLM4Tokenizer.from_pretrained(
            source_root=tmp_path,
            revision="user-revision",
            local_files_only=True,
        )
    assert DEFAULT_GLM4_REVISION in str(error.value)
    _assert_runtime_guidance(str(error.value))


def test_glm4_runtime_rejects_unvalidated_transformers_version(tmp_path: Path) -> None:
    transformers = SimpleNamespace(__version__="4.57.6")

    with (
        patch("anytrain.codec.glm4.codec.import_module", return_value=transformers),
        pytest.raises(
            RuntimeError,
            match="only been validated with Transformers 4.44.1",
        ) as error,
    ):
        _load_runtime(tmp_path)
    _assert_runtime_guidance(str(error.value))


def test_glm4_runtime_rejects_source_imported_outside_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pinned"
    root.mkdir()
    outside = tmp_path / "outside" / "modeling_whisper.py"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    transformers = SimpleNamespace(
        __version__="4.44.1",
        WhisperFeatureExtractor=object(),
    )
    modeling = SimpleNamespace(
        __file__=str(outside),
        WhisperVQEncoder=object(),
    )

    def _import(name: str):
        return transformers if name == "transformers" else modeling

    monkeypatch.setattr(sys, "path", list(sys.path))
    with (
        patch("anytrain.codec.glm4.codec.import_module", side_effect=_import),
        pytest.raises(
            RuntimeError,
            match="outside the configured source_root",
        ) as error,
    ):
        _load_runtime(root)
    _assert_runtime_guidance(str(error.value))


def test_glm4_runtime_rejects_incompatible_source_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_source_tree(tmp_path)
    modeling_path = tmp_path / "speech_tokenizer" / "modeling_whisper.py"
    transformers = SimpleNamespace(
        __version__="4.44.1",
        WhisperFeatureExtractor=SimpleNamespace(from_pretrained=Mock()),
    )
    modeling = SimpleNamespace(
        __file__=str(modeling_path),
        WhisperVQEncoder=object,
    )

    def _import(name: str):
        return transformers if name == "transformers" else modeling

    monkeypatch.setattr(sys, "path", list(sys.path))
    with (
        patch("anytrain.codec.glm4.codec.import_module", side_effect=_import),
        pytest.raises(RuntimeError, match="must expose an nn.Module WhisperVQEncoder") as error,
    ):
        _load_runtime(tmp_path)
    _assert_runtime_guidance(str(error.value))


def test_glm4_constants_pin_model_weights() -> None:
    assert DEFAULT_GLM4_ARTIFACT == "zai-org/glm-4-voice-tokenizer"
    assert DEFAULT_GLM4_REVISION == "a5f2404e63c84e92f5238908e1706316324ebafa"
