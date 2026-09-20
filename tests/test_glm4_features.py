from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anytrain.audio.codec.glm4.features import (
    GLM4PreVQTokenizer,
    nearest_codebook_ids,
    official_codebook_ids,
)


def vector_quantize(inputs: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flattened = inputs.reshape(-1, codebook.shape[1])
    distances = torch.addmm(
        (codebook**2).sum(dim=1) + (flattened**2).sum(dim=1, keepdim=True),
        flattened,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )
    ids = distances.argmin(dim=1)
    return codebook[ids].view_as(inputs), ids, distances


class _FeatureExtractor:
    def __call__(self, rows: list[Any], **options: Any) -> dict[str, Tensor]:
        maximum = int(options["max_length"])
        features = torch.zeros((len(rows), 1, maximum), dtype=torch.float32)
        mask = torch.zeros((len(rows), maximum), dtype=torch.long)
        for index, row in enumerate(rows):
            length = len(row)
            features[index, 0, :length] = torch.as_tensor(row)
            mask[index, :length] = 1
        return {"input_features": features, "attention_mask": mask}


class _Layer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: Tensor, *_: Any, **__: Any) -> tuple[Tensor, None]:
        return hidden + self.value, None


class _Model(nn.Module):
    def __init__(self, *, corrupt_ids: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            quantize_position=2,
            pooling_position=1,
            pooling_kernel_size=2,
            pooling_type="avg",
            quantize_causal_block_size=None,
            quantize_vocab_size=4,
            quantize_encoder_only=True,
            encoder_causal_convolution=False,
            d_model=2,
        )
        self.dropout = 0.0
        self.conv1 = nn.Conv1d(1, 2, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=1, stride=2, bias=False)
        self.embed_positions = nn.Embedding(32, 2)
        self.layers = nn.ModuleList((_Layer(0.25), _Layer(-0.125)))
        self.pooling_layer = nn.AvgPool1d(kernel_size=2)
        self.codebook = nn.Embedding(4, 2)
        self._corrupt_ids = corrupt_ids
        with torch.no_grad():
            self.conv1.weight.copy_(torch.tensor([[[1.0]], [[-0.5]]]))
            self.conv2.weight.copy_(torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]]))
            self.embed_positions.weight.zero_()
            self.codebook.weight.copy_(
                torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, -0.5], [2.0, -1.0]])
            )

    def get_extended_attention_mask(self, mask: Tensor, _: tuple[int, int]) -> Tensor:
        return mask[:, None, None, :].float()

    def forward(self, input_features: Tensor, attention_mask: Tensor) -> Any:
        sequence_length = input_features.shape[-1] // 2
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states)).permute(0, 2, 1)
        hidden_states = hidden_states + self.embed_positions.weight[:sequence_length]
        mask = attention_mask[:, ::2]
        extended = self.get_extended_attention_mask(
            mask, (input_features.shape[0], sequence_length)
        )
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                extended,
                layer_head_mask=None,
                output_attentions=False,
            )[0]
            if index == 0:
                hidden_states = self.pooling_layer(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
                mask = mask[:, ::2]
                extended = self.get_extended_attention_mask(
                    mask, (input_features.shape[0], sequence_length // 2)
                )
        ids = vector_quantize(hidden_states, self.codebook.weight)[1].reshape(
            input_features.shape[0], hidden_states.shape[1]
        )
        if self._corrupt_ids:
            ids = (ids + 1) % self.codebook.num_embeddings
        return SimpleNamespace(quantized_token_ids=ids)


class _Backend:
    sample_rate = 1
    _maximum_chunk_samples = 100
    _stft_guard_samples = 1
    _pad_multiple = 4
    _convolution_stride = 2
    _pooling = 2

    def __init__(self, *, corrupt_ids: bool = False) -> None:
        self.model = _Model(corrupt_ids=corrupt_ids).eval()
        self.feature_extractor = _FeatureExtractor()
        self.device = torch.device("cpu")

    @property
    def codebook(self) -> Tensor:
        return self.model.codebook.weight.detach()

    @staticmethod
    def _validate_audio(audio: Tensor) -> None:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("expected [batch, 1, time]")


def test_pre_vq_extraction_matches_same_tensor_quantization_and_full_forward() -> None:
    tokenizer = GLM4PreVQTokenizer(_Backend())
    audio = torch.tensor(
        [[[0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0]], [[1.0] * 7]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 7], dtype=torch.long)

    targets = tokenizer.extract_padded(audio, 1, lengths)

    assert targets.target_features.shape == (2, 2, 2)
    assert targets.target_lengths.tolist() == [2, 2]
    for index in range(2):
        features, ids = targets.row(index)
        assert torch.equal(ids, nearest_codebook_ids(features, tokenizer.codebook))
        assert torch.equal(
            ids,
            official_codebook_ids(features, tokenizer.codebook, tokenizer.backend.model),
        )
    report = tokenizer.verify_determinism(audio, 1, lengths)
    assert report.passed
    assert report.max_abs_difference == 0.0


def test_pre_vq_extraction_fails_closed_on_official_forward_id_drift() -> None:
    tokenizer = GLM4PreVQTokenizer(_Backend(corrupt_ids=True))
    audio = torch.ones((1, 1, 5), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="differ from the official GLM-4 forward"):
        tokenizer.extract_padded(audio, 1, torch.tensor([5], dtype=torch.long))


def test_extract_prefix_returns_pre_pool_layers_and_matches_padded_targets() -> None:
    tokenizer = GLM4PreVQTokenizer(_Backend())
    audio = torch.tensor(
        [[[0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0]], [[1.0] * 7]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 7], dtype=torch.long)

    padded = tokenizer.extract_padded(audio, 1, lengths)
    prefix = tokenizer.extract_prefix(audio, 1, lengths, layer_ids=(1,))

    assert prefix.layer_ids == (1,)
    assert prefix.layers[0].shape == (2, 4, 2)
    assert prefix.layer_lengths.tolist() == [3, 4]
    assert tokenizer.geometry.pre_pool_layer_ids == (1,)
    assert tokenizer.geometry.instantiated_layers == 2
    assert tokenizer.geometry.pooling_kernel_size == 2
    assert torch.equal(prefix.targets.target_features, padded.target_features)
    assert torch.equal(prefix.targets.target_ids, padded.target_ids)
    assert torch.equal(prefix.targets.target_lengths, padded.target_lengths)


def test_extract_prefix_records_stage_timings() -> None:
    tokenizer = GLM4PreVQTokenizer(_Backend())
    audio = torch.tensor(
        [[[0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0]], [[1.0] * 7]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 7], dtype=torch.long)
    timing: dict[str, float] = {}

    tokenizer.extract_prefix(audio, 1, lengths, layer_ids=(1,), timing=timing)

    assert timing["teacher.preprocess"] >= 0.0
    assert timing["teacher.feature_extractor"] >= 0.0
    assert timing["teacher.encoder"] >= 0.0


def test_async_feature_extract_matches_sync_prefix() -> None:
    tokenizer = GLM4PreVQTokenizer(_Backend())
    audio = torch.tensor(
        [[[0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0]], [[1.0] * 7]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 7], dtype=torch.long)

    sync = tokenizer.extract_prefix(audio, 1, lengths, layer_ids=(1,))
    job = tokenizer.begin_feature_extract(audio, 1, lengths)
    assert job is not None
    async_prefix = tokenizer.complete_extract_prefix(job, layer_ids=(1,))

    assert torch.equal(async_prefix.targets.target_features, sync.targets.target_features)
    assert torch.equal(async_prefix.targets.target_ids, sync.targets.target_ids)
    assert torch.equal(async_prefix.targets.target_lengths, sync.targets.target_lengths)
    assert async_prefix.layer_ids == sync.layer_ids
    assert torch.equal(async_prefix.layers[0], sync.layers[0])
    assert torch.equal(async_prefix.layer_lengths, sync.layer_lengths)


def test_waveform_rows_same_rate_preserves_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    from anytrain.audio.codec.glm4.features import _waveform_rows

    audio = torch.randn(2, 1, 8, device="cuda")
    rows = _waveform_rows(audio, (5, 7), sample_rate=16_000, target_rate=16_000)
    assert all(row.device.type == "cuda" for row in rows)


def test_extract_prefix_rejects_post_pool_layers() -> None:
    tokenizer = GLM4PreVQTokenizer(_Backend())
    audio = torch.ones((1, 1, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="not a pre-pool layer"):
        tokenizer.extract_prefix(
            audio,
            1,
            torch.tensor([5], dtype=torch.long),
            layer_ids=(2,),
        )


def test_nearest_codebook_ids_rejects_non_finite_features() -> None:
    with pytest.raises(ValueError, match="finite"):
        nearest_codebook_ids(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([[0.0, 0.0]]),
        )
