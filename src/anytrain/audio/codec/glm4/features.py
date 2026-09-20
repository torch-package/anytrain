"""Extract the pinned GLM-4 encoder state immediately before vector quantization.

The public adapter consumes an AnyTrain ``GLM4Tokenizer`` backend.  It deliberately stops the encoder at the
validated call boundary
``vector_quantize(hidden_states, model.codebook.weight)`` and returns that
``hidden_states`` tensor together with the IDs obtained from the same tensor.

This module does not use forward hooks.  It mirrors only the structurally
validated encoder prefix needed to expose an upstream state that the official
model does not return.  By default every extraction is checked against the
official complete forward's token IDs, making source/API drift fail closed.

Optional pre-pool hidden states are taken from the same prefix, never stored by
this adapter, and are unavailable after pooling.  ``extract_padded`` remains
the bit-identical pre-VQ-only path.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PRE_VQ_EXTRACTION_POINT = (
    "WhisperVQEncoder hidden_states immediately before "
    "vector_quantize(hidden_states, self.codebook.weight)"
)
PRE_VQ_IMPLEMENTATION_ID = "zhuyin-glm4-pre-vq-structural-prefix-v1"


@dataclass(frozen=True)
class PaddedPreVQTargets:
    """One padded batch of continuous GLM targets and aligned discrete IDs."""

    target_features: Tensor
    target_ids: Tensor
    target_lengths: Tensor

    def __post_init__(self) -> None:
        features = self.target_features
        ids = self.target_ids
        lengths = self.target_lengths
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("target_features must have shape [batch, time, feature].")
        if not features.is_floating_point() or features.is_complex():
            raise TypeError("target_features must have a real floating-point dtype.")
        if not isinstance(ids, Tensor) or ids.ndim != 2:
            raise ValueError("target_ids must have shape [batch, time].")
        if ids.dtype is torch.bool or ids.is_floating_point() or ids.is_complex():
            raise TypeError("target_ids must contain integer IDs.")
        if not isinstance(lengths, Tensor) or lengths.ndim != 1:
            raise ValueError("target_lengths must have shape [batch].")
        if lengths.dtype is torch.bool or lengths.is_floating_point() or lengths.is_complex():
            raise TypeError("target_lengths must contain integers.")
        if features.shape[:2] != ids.shape or lengths.numel() != features.shape[0]:
            raise ValueError("pre-VQ features, IDs, and lengths do not share one batch grid.")
        if features.shape[0] == 0 or features.shape[1] == 0 or features.shape[2] == 0:
            raise ValueError("pre-VQ target batches must be non-empty.")
        if bool((lengths <= 0).any()) or bool((lengths > features.shape[1]).any()):
            raise ValueError("target_lengths exceed the padded pre-VQ extent.")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("target_features contain NaN or infinity.")

    def row(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one unpadded feature/ID pair on the shared valid mask."""

        length = int(self.target_lengths[index].item())
        return (
            self.target_features[index, :length].contiguous(),
            self.target_ids[index, :length].contiguous(),
        )


@dataclass(frozen=True)
class WhisperVQGeometry:
    """Instantiated Whisper-VQ encoder layout used by a pinned GLM backend."""

    instantiated_layers: int
    quantize_position: int
    pooling_position: int
    pooling_kernel_size: int | None
    pooling_type: str | None
    quantize_encoder_only: bool
    d_model: int
    pre_pool_layer_ids: tuple[int, ...]
    encoder_causal_convolution: bool
    quantize_causal_block_size: int | None


@dataclass(frozen=True)
class EncoderPrefix:
    """Live encoder prefix: pre-VQ targets plus requested 50 Hz layer states."""

    targets: PaddedPreVQTargets
    layers: tuple[Tensor, ...]
    layer_ids: tuple[int, ...]
    layer_lengths: Tensor

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.layer_ids):
            raise ValueError("layers and layer_ids must have the same length.")
        if tuple(sorted(set(self.layer_ids))) != tuple(sorted(self.layer_ids)):
            raise ValueError("layer_ids must be unique.")
        if any(layer_id <= 0 for layer_id in self.layer_ids):
            raise ValueError("layer_ids must be positive 1-indexed encoder depths.")
        if not isinstance(self.layer_lengths, Tensor) or self.layer_lengths.ndim != 1:
            raise ValueError("layer_lengths must have shape [batch].")
        if (
            self.layer_lengths.dtype is torch.bool
            or self.layer_lengths.is_floating_point()
            or self.layer_lengths.is_complex()
        ):
            raise TypeError("layer_lengths must contain integers.")
        batch = self.targets.target_features.shape[0]
        if self.layer_lengths.numel() != batch:
            raise ValueError("layer_lengths must align with the pre-VQ batch.")
        if self.layers:
            if bool((self.layer_lengths <= 0).any()) or bool(
                (self.layer_lengths > self.layers[0].shape[1]).any()
            ):
                raise ValueError("layer_lengths exceed the padded 50 Hz extent.")
            feature_dim = self.targets.target_features.shape[-1]
            padded = self.layers[0].shape[1]
            for layer, layer_id in zip(self.layers, self.layer_ids, strict=True):
                if not isinstance(layer, Tensor) or layer.ndim != 3:
                    raise ValueError(f"layer {layer_id} must have shape [batch, time, feature].")
                if layer.shape != (batch, padded, feature_dim):
                    raise ValueError(
                        f"layer {layer_id} must share the pre-pool grid "
                        f"[{batch}, {padded}, {feature_dim}], got {tuple(layer.shape)}."
                    )
                if not layer.is_floating_point() or layer.is_complex():
                    raise TypeError(f"layer {layer_id} must use a real floating-point dtype.")
                if not bool(torch.isfinite(layer).all()):
                    raise ValueError(f"layer {layer_id} contains NaN or infinity.")
        elif bool((self.layer_lengths != 0).any()):
            raise ValueError("layer_lengths must be zero when no layers were requested.")

    def layer_row(self, index: int, layer_index: int = 0) -> Tensor:
        """Return one unpadded 50 Hz hidden-state row."""

        length = int(self.layer_lengths[index].item())
        return self.layers[layer_index][index, :length].contiguous()


@dataclass(frozen=True)
class _FeatureExtractPayload:
    feature_mapping: dict[str, Tensor]
    feature_extractor_seconds: float


@dataclass(frozen=True)
class FeatureExtractJob:
    """In-flight CPU feature extraction for one preprocessed waveform batch."""

    rows: tuple[Tensor, ...]
    preprocess_seconds: float
    _future: Future[_FeatureExtractPayload]


@dataclass(frozen=True)
class DeterminismReport:
    """Numerical repeatability observed for two identical extractions."""

    max_abs_difference: float
    ids_equal: bool
    lengths_equal: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.ids_equal and self.lengths_equal and self.max_abs_difference <= self.tolerance


class GLM4PreVQTokenizer:
    """Stable pre-VQ interface over one workspace-pinned GLM tokenizer backend."""

    def __init__(self, backend: Any, *, verify_forward_tokens: bool = True) -> None:
        self.backend = backend
        self.verify_forward_tokens = _boolean(verify_forward_tokens, "verify_forward_tokens")
        self._feature_executor_pool: ThreadPoolExecutor | None = None
        self._validate_backend()

    @property
    def device(self) -> torch.device:
        value = getattr(self.backend, "device", None)
        if not isinstance(value, torch.device):
            raise TypeError("GLM-4 backend must expose a torch.device.")
        return value

    @property
    def codebook(self) -> Tensor:
        value = getattr(self.backend, "codebook", None)
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise TypeError("GLM-4 backend must expose a rank-2 codebook tensor.")
        return value.detach()

    @property
    def feature_dim(self) -> int:
        return int(self.codebook.shape[1])

    @property
    def geometry(self) -> WhisperVQGeometry:
        """Return the instantiated encoder pooling/VQ layout."""

        return whisper_vq_geometry(self.backend.model)

    @torch.inference_mode()
    def extract_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
        *,
        timing: MutableMapping[str, float] | None = None,
    ) -> PaddedPreVQTargets:
        """Extract a padded waveform batch with the backend's exact chunk policy."""

        return self._extract(audio, sample_rate, lengths, layer_ids=(), timing=timing).targets

    @torch.inference_mode()
    def extract_prefix(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
        *,
        layer_ids: Sequence[int],
        storage_device: torch.device | str | None = None,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        """Run the same prefix as ``extract_padded`` and keep selected pre-pool layers.

        ``layer_ids`` are 1-indexed encoder depths after that layer and before
        pooling.  States stay in memory; this method does not write a store.
        ``storage_device`` defaults to CPU to match ``extract_padded``.
        Optional ``timing`` accumulates wall seconds for preprocess,
        feature_extractor, and encoder stages.
        """

        return self._extract(
            audio,
            sample_rate,
            lengths,
            layer_ids=_pre_pool_layer_ids(layer_ids, self.geometry),
            storage_device=_storage_device(storage_device),
            timing=timing,
        )

    @torch.inference_mode()
    def begin_feature_extract(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> FeatureExtractJob | None:
        """Start CPU feature extraction so it can overlap with GPU student work.

        Returns ``None`` when the backend chunk policy requires the chunked sync
        path in :meth:`extract_prefix`.
        """

        rows, preprocess_seconds = self._prepare_rows(audio, sample_rate, lengths)
        maximum_chunk_samples = _positive_int_attribute(self.backend, "_maximum_chunk_samples")
        if not all(int(row.numel()) <= maximum_chunk_samples for row in rows):
            return None
        future = self._feature_executor().submit(self._run_feature_extractor, rows)
        return FeatureExtractJob(
            rows=rows,
            preprocess_seconds=preprocess_seconds,
            _future=future,
        )

    @torch.inference_mode()
    def complete_extract_prefix(
        self,
        job: FeatureExtractJob,
        *,
        layer_ids: Sequence[int],
        storage_device: torch.device | str | None = None,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        """Finish one async feature-extract job with the GPU encoder prefix."""

        wait_started = perf_counter() if timing is not None else None
        payload = job._future.result()
        if timing is not None:
            timing["teacher.preprocess"] = job.preprocess_seconds
            timing["teacher.feature_extractor"] = payload.feature_extractor_seconds
            if wait_started is not None:
                timing["teacher.feature_extractor_wait"] = perf_counter() - wait_started
        return self._encode_from_features(
            job.rows,
            payload.feature_mapping,
            layer_ids=_pre_pool_layer_ids(layer_ids, self.geometry),
            storage_device=_storage_device(storage_device),
            timing=timing,
        )

    @torch.inference_mode()
    def extract_feature_mapping(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> dict[str, Tensor]:
        """Run the CPU Whisper feature extractor for one padded waveform batch."""

        rows, _ = self._prepare_rows(audio, sample_rate, lengths)
        return dict(self._feature_extract_rows(rows))

    @torch.inference_mode()
    def encode_prefix_from_mapping(
        self,
        feature_mapping: Mapping[str, Tensor],
        *,
        layer_ids: Sequence[int],
        storage_device: torch.device | str | None = None,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        """Run the GPU encoder prefix from cached Whisper features."""

        input_features = feature_mapping.get("input_features")
        attention_mask = feature_mapping.get("attention_mask")
        if not isinstance(input_features, Tensor) or input_features.ndim != 3:
            raise TypeError("feature_mapping requires rank-3 input_features.")
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise TypeError("feature_mapping requires rank-2 attention_mask.")
        batch_size = input_features.shape[0]
        if attention_mask.shape[0] != batch_size:
            raise ValueError("feature_mapping batch sizes do not align.")
        dummy_rows = tuple(torch.empty(0) for _ in range(batch_size))
        return self._encode_from_features(
            dummy_rows,
            feature_mapping,
            layer_ids=_pre_pool_layer_ids(layer_ids, self.geometry),
            storage_device=_storage_device(storage_device),
            timing=timing,
        )

    def _extract(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
        *,
        layer_ids: tuple[int, ...],
        storage_device: torch.device | None = None,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        rows, preprocess_seconds = self._prepare_rows(
            audio,
            sample_rate,
            lengths,
            timing=timing,
        )
        device = _storage_device(storage_device)
        maximum_chunk_samples = _positive_int_attribute(self.backend, "_maximum_chunk_samples")
        if all(int(row.numel()) <= maximum_chunk_samples for row in rows):
            return self._extract_rows(
                rows,
                layer_ids=layer_ids,
                storage_device=device,
                timing=timing,
            )
        return self._extract_chunked_rows(
            rows,
            maximum_chunk_samples,
            layer_ids=layer_ids,
            storage_device=device,
            timing=timing,
        )

    @torch.inference_mode()
    def verify_determinism(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
        *,
        tolerance: float = 0.0,
    ) -> DeterminismReport:
        """Run the exact request twice and report continuous/discrete agreement."""

        if not isinstance(tolerance, int | float) or isinstance(tolerance, bool):
            raise TypeError("determinism tolerance must be a number.")
        if tolerance < 0:
            raise ValueError("determinism tolerance must not be negative.")
        first = self.extract_padded(audio, sample_rate, lengths)
        second = self.extract_padded(audio, sample_rate, lengths)
        difference = float((first.target_features - second.target_features).abs().max().item())
        return DeterminismReport(
            max_abs_difference=difference,
            ids_equal=torch.equal(first.target_ids, second.target_ids),
            lengths_equal=torch.equal(first.target_lengths, second.target_lengths),
            tolerance=float(tolerance),
        )

    def _extract_chunked_rows(
        self,
        rows: tuple[Tensor, ...],
        maximum_chunk_samples: int,
        *,
        layer_ids: tuple[int, ...],
        storage_device: torch.device,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        chunks = tuple(row.split(maximum_chunk_samples) for row in rows)
        feature_rows: list[list[Tensor]] = [[] for _ in rows]
        id_rows: list[list[Tensor]] = [[] for _ in rows]
        layer_rows: list[list[list[Tensor]]] = [[[] for _ in layer_ids] for _ in rows]
        for chunk_index in range(max(len(row_chunks) for row_chunks in chunks)):
            row_indices = tuple(
                index for index, row_chunks in enumerate(chunks) if chunk_index < len(row_chunks)
            )
            chunk_timing: dict[str, float] | None = {} if timing is not None else None
            extracted = self._extract_rows(
                tuple(chunks[index][chunk_index] for index in row_indices),
                layer_ids=layer_ids,
                storage_device=storage_device,
                timing=chunk_timing,
            )
            if timing is not None and chunk_timing is not None:
                for key, value in chunk_timing.items():
                    timing[key] = timing.get(key, 0.0) + value
            for batch_index, row_index in enumerate(row_indices):
                features, ids = extracted.targets.row(batch_index)
                feature_rows[row_index].append(features)
                id_rows[row_index].append(ids)
                for layer_index in range(len(layer_ids)):
                    layer_rows[row_index][layer_index].append(
                        extracted.layer_row(batch_index, layer_index)
                    )
        targets = _pad_rows(
            tuple(torch.cat(values, dim=0) for values in feature_rows),
            tuple(torch.cat(values, dim=0) for values in id_rows),
        )
        if not layer_ids:
            return EncoderPrefix(
                targets=targets,
                layers=(),
                layer_ids=(),
                layer_lengths=targets.target_lengths.new_zeros(len(rows)),
            )
        concatenated = tuple(
            tuple(torch.cat(pieces, dim=0) for pieces in sample_layers)
            for sample_layers in layer_rows
        )
        return EncoderPrefix(
            targets=targets,
            layers=_pad_hidden_rows(
                tuple(
                    tuple(sample_layers[layer_index] for sample_layers in concatenated)
                    for layer_index in range(len(layer_ids))
                )
            ),
            layer_ids=layer_ids,
            layer_lengths=torch.tensor(
                [int(sample_layers[0].shape[0]) for sample_layers in concatenated],
                dtype=torch.long,
            ),
        )

    def _prepare_rows(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
        *,
        timing: MutableMapping[str, float] | None = None,
    ) -> tuple[tuple[Tensor, ...], float]:
        validator = getattr(self.backend, "_validate_audio", None)
        if not callable(validator):
            raise TypeError("GLM-4 backend has no waveform validator.")
        validator(audio)
        valid_lengths = _waveform_lengths(
            lengths,
            batch_size=audio.size(0),
            maximum=audio.size(-1),
        )
        target_rate = int(self.backend.sample_rate)
        preprocess_started = perf_counter()
        rows = _waveform_rows(
            audio.detach().float(),
            valid_lengths,
            sample_rate=sample_rate,
            target_rate=target_rate,
        )
        preprocess_seconds = perf_counter() - preprocess_started
        if timing is not None:
            timing["teacher.preprocess"] = preprocess_seconds
        return rows, preprocess_seconds

    def _feature_executor(self) -> ThreadPoolExecutor:
        if self._feature_executor_pool is None:
            self._feature_executor_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="glm4-feature-extractor",
            )
        return self._feature_executor_pool

    def _run_feature_extractor(
        self,
        rows: tuple[Tensor, ...],
    ) -> _FeatureExtractPayload:
        started = perf_counter()
        feature_mapping = self._feature_extract_rows(rows)
        return _FeatureExtractPayload(
            feature_mapping=feature_mapping,
            feature_extractor_seconds=perf_counter() - started,
        )

    def _feature_extract_rows(self, rows: tuple[Tensor, ...]) -> dict[str, Tensor]:
        maximum = max(int(row.numel()) for row in rows)
        stft_guard = _positive_int_attribute(self.backend, "_stft_guard_samples")
        pad_multiple = _positive_int_attribute(self.backend, "_pad_multiple")
        padded_samples = ((maximum + stft_guard + pad_multiple - 1) // pad_multiple) * pad_multiple
        feature_extractor = getattr(self.backend, "feature_extractor", None)
        if not callable(feature_extractor):
            raise TypeError("GLM-4 backend must expose a callable feature_extractor.")
        features = feature_extractor(
            [_row_numpy(row) for row in rows],
            sampling_rate=int(self.backend.sample_rate),
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            max_length=padded_samples,
            pad_to_multiple_of=pad_multiple,
            device="cpu",
        )
        feature_mapping = _feature_mapping(features)
        attention_mask = feature_mapping.get("attention_mask")
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise RuntimeError("GLM-4 feature extractor must return a 2D attention_mask.")
        if attention_mask.shape[0] != len(rows):
            raise RuntimeError("GLM-4 feature attention_mask does not align with the batch.")
        return dict(feature_mapping)

    def _encode_from_features(
        self,
        rows: tuple[Tensor, ...],
        feature_mapping: Mapping[str, Tensor],
        *,
        layer_ids: tuple[int, ...],
        storage_device: torch.device,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        attention_mask = feature_mapping.get("attention_mask")
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise RuntimeError("GLM-4 feature extractor must return a 2D attention_mask.")
        if attention_mask.shape[0] != len(rows):
            raise RuntimeError("GLM-4 feature attention_mask does not align with the batch.")
        model_inputs = {
            key: value.to(device=self.device)
            for key, value in feature_mapping.items()
            if isinstance(value, Tensor)
        }
        model = getattr(self.backend, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError("GLM-4 backend must expose an nn.Module model.")
        encoder_started = perf_counter() if timing is not None else None
        with _tf32_disabled():
            hidden_states, collected, layer_mask = _run_encoder_to_pre_vq(
                cast("_GlmEncoder", model),
                model_inputs,
                layer_ids=layer_ids,
            )
            ids = _model_vector_quantize(model, hidden_states, self.codebook)[1].reshape(
                len(rows), hidden_states.shape[1]
            )
            if self.verify_forward_tokens:
                official = model(**model_inputs)
                official_ids = getattr(official, "quantized_token_ids", None)
                if not isinstance(official_ids, Tensor) or official_ids.shape != ids.shape:
                    raise RuntimeError(
                        "official GLM-4 forward did not return aligned quantized_token_ids."
                    )
                if not torch.equal(ids, official_ids):
                    mismatches = int((ids != official_ids).sum().item())
                    raise RuntimeError(
                        "pre-VQ IDs differ from the official GLM-4 forward: "
                        f"{mismatches}/{ids.numel()} positions."
                    )
        if timing is not None and encoder_started is not None:
            timing["teacher.encoder"] = perf_counter() - encoder_started

        convolution_stride = _positive_int_attribute(self.backend, "_convolution_stride")
        pooling = _positive_int_attribute(self.backend, "_pooling")
        token_mask = attention_mask[:, ::convolution_stride]
        token_mask = token_mask[:, ::pooling]
        if token_mask.shape != ids.shape or hidden_states.shape[:2] != ids.shape:
            raise RuntimeError(
                "pre-VQ states, token mask, and quantized IDs do not share one grid: "
                f"hidden={tuple(hidden_states.shape)}, mask={tuple(token_mask.shape)}, "
                f"ids={tuple(ids.shape)}."
            )
        if layer_mask.shape[0] != len(rows):
            raise RuntimeError("pre-pool layer mask does not align with the batch.")
        target_lengths = token_mask.to(dtype=torch.long).sum(dim=1).to(device=storage_device)
        targets = PaddedPreVQTargets(
            target_features=_store(hidden_states, storage_device, dtype=torch.float32),
            target_ids=_store(ids, storage_device, dtype=torch.long),
            target_lengths=target_lengths.contiguous(),
        )
        layer_lengths = layer_mask.to(dtype=torch.long).sum(dim=1).to(device=storage_device)
        if not layer_ids:
            layer_lengths = layer_lengths.new_zeros(len(rows))
        return EncoderPrefix(
            targets=targets,
            layers=tuple(
                _store(collected[layer_id], storage_device, dtype=torch.float32)
                for layer_id in layer_ids
            ),
            layer_ids=layer_ids,
            layer_lengths=layer_lengths.contiguous(),
        )

    def _extract_rows(
        self,
        rows: tuple[Tensor, ...],
        *,
        layer_ids: tuple[int, ...],
        storage_device: torch.device,
        timing: MutableMapping[str, float] | None = None,
    ) -> EncoderPrefix:
        feature_started = perf_counter() if timing is not None else None
        feature_mapping = self._feature_extract_rows(rows)
        if timing is not None and feature_started is not None:
            timing["teacher.feature_extractor"] = perf_counter() - feature_started
        return self._encode_from_features(
            rows,
            feature_mapping,
            layer_ids=layer_ids,
            storage_device=storage_device,
            timing=timing,
        )

    def _validate_backend(self) -> None:
        model = getattr(self.backend, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError("GLM-4 pre-VQ extraction requires an nn.Module model.")
        if model.training:
            raise ValueError("GLM-4 pre-VQ extraction requires model.eval().")
        config = getattr(model, "config", None)
        quantize_position = getattr(config, "quantize_position", None)
        layers = getattr(model, "layers", None)
        if (
            isinstance(quantize_position, bool)
            or not isinstance(quantize_position, int)
            or quantize_position <= 0
            or not isinstance(layers, nn.ModuleList)
            or quantize_position > len(layers)
        ):
            raise RuntimeError("GLM encoder has no valid in-layer quantize_position.")
        codebook = getattr(model, "codebook", None)
        if not isinstance(codebook, nn.Embedding):
            raise TypeError("GLM encoder must expose an embedding codebook.")
        if tuple(codebook.weight.shape) != tuple(self.codebook.shape):
            raise RuntimeError("GLM backend and encoder codebooks differ in shape.")
        if not torch.equal(codebook.weight.detach(), self.codebook):
            raise RuntimeError("GLM backend and encoder codebooks differ in value.")
        if self.feature_dim <= 0:
            raise RuntimeError("GLM codebook feature dimension must be positive.")
        _validate_extraction_source(model)


def official_codebook_ids(
    features: Tensor,
    codebook: Tensor,
    model: nn.Module,
) -> Tensor:
    """Re-quantize stored features with the GLM ``vector_quantize`` used at extraction.

    Features are moved onto the codebook device and TF32 is disabled so the
    writer/reader ID gate matches extraction rather than a second local GEMM.
    """

    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError("features must have shape [time, feature].")
    if not isinstance(codebook, Tensor) or codebook.ndim != 2:
        raise ValueError("codebook must have shape [vocabulary, feature].")
    if features.shape[1] != codebook.shape[1]:
        raise ValueError("feature and codebook dimensions differ.")
    if not isinstance(model, nn.Module):
        raise TypeError("official codebook IDs require the GLM encoder module.")
    resolved = codebook.detach().float().contiguous()
    moved = features.detach().float().to(device=resolved.device).contiguous()
    if not bool(torch.isfinite(moved).all()) or not bool(torch.isfinite(resolved).all()):
        raise ValueError("features and codebook must contain only finite values.")
    with _tf32_disabled():
        _, ids, _ = _model_vector_quantize(model, moved, resolved)
    return ids.reshape(-1).detach().to(dtype=torch.long, device="cpu").contiguous()


def nearest_codebook_ids(
    features: Tensor,
    codebook: Tensor,
    *,
    frame_chunk_size: int = 1_024,
) -> Tensor:
    """Return exact nearest-codebook IDs with bounded frame-distance memory."""

    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError("features must have shape [time, feature].")
    if not isinstance(codebook, Tensor) or codebook.ndim != 2:
        raise ValueError("codebook must have shape [vocabulary, feature].")
    if features.shape[1] != codebook.shape[1]:
        raise ValueError("feature and codebook dimensions differ.")
    if not features.is_floating_point() or not codebook.is_floating_point():
        raise TypeError("features and codebook must be floating-point tensors.")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(codebook).all()):
        raise ValueError("features and codebook must contain only finite values.")
    chunk_size = _positive_integer(frame_chunk_size, "frame_chunk_size")
    resolved_codebook = codebook.detach().float()
    outputs = []
    for chunk in features.detach().float().split(chunk_size, dim=0):
        _, ids, _ = _vector_quantize(chunk, resolved_codebook)
        outputs.append(ids.to(dtype=torch.long))
    return torch.cat(outputs, dim=0).contiguous()


def codebook_sha256(codebook: Tensor) -> str:
    """Hash the raw contiguous CPU bytes of a frozen GLM codebook."""

    if not isinstance(codebook, Tensor) or codebook.ndim != 2:
        raise ValueError("codebook must have shape [vocabulary, feature].")
    value = codebook.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


class _GlmEncoderConfig(Protocol):
    """Pinned GLM encoder config fields the pre-VQ extractor reads."""

    quantize_position: int
    pooling_position: int | None
    pooling_kernel_size: int | None
    quantize_causal_block_size: int | None


class _GlmEncoder(Protocol):
    """Instantiated GLM encoder surface the pre-VQ extractor depends on.

    The runtime contract stays ``nn.Module``-based (``_validate_backend`` checks
    the codebook and quantize point); this declares the extra members the
    extractor reads so they are not resolved as opaque ``Tensor`` attributes.
    """

    config: _GlmEncoderConfig
    layers: nn.ModuleList
    dropout: float
    training: bool

    def get_block_causal_attention_mask(
        self,
        attention_mask: Tensor,
        *,
        block_size: int,
    ) -> Tensor: ...

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        shape: tuple[int, int],
    ) -> Tensor: ...


def _run_encoder_to_pre_vq(
    model: _GlmEncoder,
    inputs: Mapping[str, Tensor],
    *,
    layer_ids: tuple[int, ...] = (),
) -> tuple[Tensor, dict[int, Tensor], Tensor]:
    input_features = inputs.get("input_features")
    attention_mask = inputs.get("attention_mask")
    if not isinstance(input_features, Tensor) or input_features.ndim != 3:
        raise RuntimeError("GLM model inputs require rank-3 input_features.")
    if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
        raise RuntimeError("GLM model inputs require rank-2 attention_mask.")
    config = model.config
    conv1 = _module(model, "conv1")
    conv2 = _module(model, "conv2")
    convolution_stride = _stride(conv1) * _stride(conv2)
    batch_size = input_features.shape[0]
    sequence_length = input_features.shape[-1] // convolution_stride
    attention_mask = attention_mask[:, ::convolution_stride]
    layer_mask = attention_mask.detach()
    requested = set(layer_ids)
    block_size = getattr(config, "quantize_causal_block_size", None)
    if block_size is not None:
        extended_attention_mask = model.get_block_causal_attention_mask(
            attention_mask,
            block_size=block_size,
        )
    else:
        extended_attention_mask = model.get_extended_attention_mask(
            attention_mask,
            (batch_size, sequence_length),
        )
    hidden_states = F.gelu(conv1(input_features))
    hidden_states = F.gelu(conv2(hidden_states)).permute(0, 2, 1)
    embed_positions = _module(model, "embed_positions")
    position_weight = getattr(embed_positions, "weight", None)
    if not isinstance(position_weight, Tensor) or position_weight.shape[0] < sequence_length:
        raise RuntimeError("GLM positional embedding is shorter than the encoder input.")
    hidden_states = hidden_states + position_weight[:sequence_length]
    hidden_states = F.dropout(
        hidden_states,
        p=float(model.dropout),
        training=model.training,
    )
    if attention_mask.shape[-1] != hidden_states.shape[1]:
        raise RuntimeError("GLM attention mask does not match encoder hidden states.")

    quantize_position = cast(int, config.quantize_position)
    pooling_position = getattr(config, "pooling_position", None)
    pooling_kernel = getattr(config, "pooling_kernel_size", None)
    layers = cast(nn.ModuleList, model.layers)
    collected: dict[int, Tensor] = {}
    pooled = False
    for index, encoder_layer in enumerate(layers[:quantize_position]):
        layer_outputs = encoder_layer(
            hidden_states,
            extended_attention_mask,
            layer_head_mask=None,
            output_attentions=False,
        )
        if not isinstance(layer_outputs, tuple) or not isinstance(layer_outputs[0], Tensor):
            raise RuntimeError("GLM encoder layer did not return hidden states.")
        hidden_states = layer_outputs[0]
        layer_id = index + 1
        if layer_id in requested:
            if pooled:
                raise RuntimeError(
                    f"encoder layer {layer_id} is after pooling; only pre-pool layers "
                    "are returned as hidden states."
                )
            collected[layer_id] = hidden_states
        if layer_id == pooling_position and pooling_kernel is not None:
            pooling = _module(model, "pooling_layer")
            transposed = hidden_states.permute(0, 2, 1)
            remainder = transposed.shape[-1] % pooling_kernel
            if remainder:
                transposed = F.pad(transposed, (0, pooling_kernel - remainder))
            hidden_states = pooling(transposed).permute(0, 2, 1)
            attention_mask = attention_mask[:, ::pooling_kernel]
            pooled = True
            if block_size is not None:
                extended_attention_mask = model.get_block_causal_attention_mask(
                    attention_mask,
                    block_size=block_size // pooling_kernel,
                )
            else:
                extended_attention_mask = model.get_extended_attention_mask(
                    attention_mask,
                    (batch_size, sequence_length // pooling_kernel),
                )
        if layer_id == quantize_position:
            missing = requested.difference(collected)
            if missing:
                raise RuntimeError(
                    "encoder prefix did not collect requested pre-pool layers: "
                    f"{tuple(sorted(missing))}."
                )
            return hidden_states, collected, layer_mask
    raise RuntimeError("GLM encoder prefix did not reach quantize_position.")


def _validate_extraction_source(model: nn.Module) -> None:
    """Fail if the pinned source no longer quantizes the declared state/table."""

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(type(model).forward)))
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        raise RuntimeError("cannot inspect the GLM encoder forward extraction point.") from exc
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "vector_quantize"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "hidden_states"
        and _is_self_codebook_weight(node.args[1])
    ]
    if len(calls) != 1:
        raise RuntimeError(
            "GLM encoder forward must contain exactly one "
            "vector_quantize(hidden_states, self.codebook.weight) call."
        )


def _is_self_codebook_weight(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "weight"
        and isinstance(value.value, ast.Attribute)
        and value.value.attr == "codebook"
        and isinstance(value.value.value, ast.Name)
        and value.value.value.id == "self"
    )


def _model_vector_quantize(
    model: nn.Module,
    hidden_states: Tensor,
    codebook: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    module = import_module(type(model).__module__)
    function = getattr(module, "vector_quantize", None)
    if not callable(function):
        raise RuntimeError("GLM source module does not expose vector_quantize().")
    output = function(hidden_states, codebook)
    if (
        not isinstance(output, tuple)
        or len(output) != 3
        or not all(isinstance(value, Tensor) for value in output)
    ):
        raise RuntimeError("GLM vector_quantize() returned an unsupported value.")
    return cast(tuple[Tensor, Tensor, Tensor], output)


def _vector_quantize(inputs: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    embedding_size = codebook.size(1)
    flattened = inputs.reshape(-1, embedding_size)
    codebook_squared = torch.sum(codebook**2, dim=1)
    input_squared = torch.sum(flattened**2, dim=1, keepdim=True)
    distances = torch.addmm(
        codebook_squared + input_squared,
        flattened,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )
    indices = torch.min(distances, dim=1).indices
    codes = torch.index_select(codebook, dim=0, index=indices).view_as(inputs)
    return codes, indices, distances


def _pad_rows(
    feature_rows: tuple[Tensor, ...],
    id_rows: tuple[Tensor, ...],
) -> PaddedPreVQTargets:
    lengths = torch.tensor([row.shape[0] for row in feature_rows], dtype=torch.long)
    maximum = int(lengths.max().item())
    feature_dim = feature_rows[0].shape[1]
    features = feature_rows[0].new_zeros((len(feature_rows), maximum, feature_dim))
    ids = id_rows[0].new_zeros((len(id_rows), maximum), dtype=torch.long)
    for index, (feature_row, id_row) in enumerate(zip(feature_rows, id_rows, strict=True)):
        if feature_row.shape[0] != id_row.shape[0] or feature_row.shape[1] != feature_dim:
            raise RuntimeError("chunked pre-VQ rows do not share one aligned feature grid.")
        features[index, : feature_row.shape[0]] = feature_row
        ids[index, : id_row.shape[0]] = id_row
    return PaddedPreVQTargets(features, ids, lengths)


def _feature_mapping(features: object) -> Mapping[str, Tensor]:
    if isinstance(features, Mapping):
        return cast(Mapping[str, Tensor], features)
    data = getattr(features, "data", None)
    if isinstance(data, Mapping):
        return cast(Mapping[str, Tensor], data)
    raise TypeError("GLM feature extractor output must be a mapping of tensors.")


def _waveform_lengths(value: Tensor, *, batch_size: int, maximum: int) -> tuple[int, ...]:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError("padded waveform lengths must have shape [batch].")
    if value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("padded waveform lengths must contain signed integers.")
    if value.numel() != batch_size:
        raise ValueError("padded waveform lengths do not align with the batch.")
    lengths = tuple(int(length) for length in value.detach().cpu().tolist())
    if any(length <= 0 or length > maximum for length in lengths):
        raise ValueError("padded waveform lengths exceed the waveform extent.")
    return lengths


def _waveform_rows(
    waveform: Tensor,
    valid_lengths: tuple[int, ...],
    *,
    sample_rate: int,
    target_rate: int,
) -> tuple[Tensor, ...]:
    if sample_rate == target_rate:
        return tuple(
            waveform[index, 0, :length].contiguous() for index, length in enumerate(valid_lengths)
        )
    return tuple(
        _resample(
            waveform[index : index + 1, :, :length],
            sample_rate,
            target_rate,
        )[0, 0].contiguous()
        for index, length in enumerate(valid_lengths)
    )


def _row_numpy(row: Tensor) -> Any:
    return row.detach().cpu().numpy()


def _resample(audio: Tensor, source_rate: int, target_rate: int) -> Tensor:
    from anytrain.audio.codec._audio import resample

    return resample(audio, source_rate, target_rate)


def _module(value: object, name: str) -> nn.Module:
    resolved = getattr(value, name, None)
    if not isinstance(resolved, nn.Module):
        raise TypeError(f"GLM encoder must expose module {name!r}.")
    return resolved


def _stride(value: nn.Module) -> int:
    stride = getattr(value, "stride", None)
    if not isinstance(stride, tuple) or len(stride) != 1 or int(stride[0]) <= 0:
        raise TypeError("GLM convolution must expose one positive stride.")
    return int(stride[0])


def _positive_int_attribute(value: object, name: str) -> int:
    return _positive_integer(getattr(value, name, None), name)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer.")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def whisper_vq_geometry(model: _GlmEncoder) -> WhisperVQGeometry:
    """Read the instantiated encoder layout; pooling happens after these layers."""

    if not isinstance(model, nn.Module):
        raise TypeError("whisper_vq_geometry requires an nn.Module encoder.")
    config = getattr(model, "config", None)
    layers = getattr(model, "layers", None)
    quantize_position = getattr(config, "quantize_position", None)
    pooling_position = getattr(config, "pooling_position", None)
    pooling_kernel = getattr(config, "pooling_kernel_size", None)
    if (
        isinstance(quantize_position, bool)
        or not isinstance(quantize_position, int)
        or quantize_position <= 0
        or not isinstance(layers, nn.ModuleList)
        or quantize_position > len(layers)
    ):
        raise RuntimeError("GLM encoder has no valid in-layer quantize_position.")
    if isinstance(pooling_position, bool) or not isinstance(pooling_position, int):
        raise RuntimeError("GLM encoder pooling_position must be an integer.")
    if pooling_position <= 0:
        pre_pool = tuple(range(1, quantize_position + 1))
    else:
        pre_pool = tuple(range(1, min(pooling_position, quantize_position) + 1))
    pooling_type = getattr(config, "pooling_type", None)
    if pooling_type is not None and not isinstance(pooling_type, str):
        raise RuntimeError("GLM encoder pooling_type must be a string.")
    d_model = getattr(config, "d_model", None)
    if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
        raise RuntimeError("GLM encoder d_model must be a positive integer.")
    block_size = getattr(config, "quantize_causal_block_size", None)
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0
    ):
        raise RuntimeError("quantize_causal_block_size must be a positive integer.")
    kernel: int | None
    if pooling_kernel is None:
        kernel = None
    else:
        kernel = _positive_integer(pooling_kernel, "pooling_kernel_size")
    return WhisperVQGeometry(
        instantiated_layers=len(layers),
        quantize_position=quantize_position,
        pooling_position=pooling_position,
        pooling_kernel_size=kernel,
        pooling_type=pooling_type,
        quantize_encoder_only=bool(getattr(config, "quantize_encoder_only", False)),
        d_model=d_model,
        pre_pool_layer_ids=pre_pool,
        encoder_causal_convolution=bool(getattr(config, "encoder_causal_convolution", False)),
        quantize_causal_block_size=block_size,
    )


def _pre_pool_layer_ids(
    layer_ids: Sequence[int],
    geometry: WhisperVQGeometry,
) -> tuple[int, ...]:
    if isinstance(layer_ids, (str, bytes)) or not isinstance(layer_ids, Sequence):
        raise TypeError("layer_ids must be a sequence of integers.")
    ids = tuple(int(layer_id) for layer_id in layer_ids)
    if not ids:
        raise ValueError("layer_ids must not be empty.")
    if tuple(sorted(set(ids))) != tuple(sorted(ids)):
        raise ValueError("layer_ids must be unique.")
    allowed = set(geometry.pre_pool_layer_ids)
    invalid = tuple(layer_id for layer_id in ids if layer_id not in allowed)
    if invalid:
        raise ValueError(
            f"layer_ids {invalid} are not a pre-pool layer; allowed {geometry.pre_pool_layer_ids}."
        )
    return ids


def _storage_device(value: torch.device | str | None) -> torch.device:
    if value is None:
        return torch.device("cpu")
    return torch.device(value)


def _store(value: Tensor, device: torch.device, *, dtype: torch.dtype) -> Tensor:
    return value.detach().to(device=device, dtype=dtype).contiguous()


def _pad_hidden_rows(layer_batches: tuple[tuple[Tensor, ...], ...]) -> tuple[Tensor, ...]:
    return tuple(_pad_one_hidden(rows) for rows in layer_batches)


def _pad_one_hidden(rows: tuple[Tensor, ...]) -> Tensor:
    if not rows:
        raise ValueError("cannot pad an empty hidden-state batch.")
    maximum = max(int(row.shape[0]) for row in rows)
    feature_dim = int(rows[0].shape[1])
    padded = rows[0].new_zeros((len(rows), maximum, feature_dim))
    for index, row in enumerate(rows):
        padded[index, : row.shape[0]] = row
    return padded.contiguous()


@contextmanager
def _tf32_disabled() -> Generator[None, None, None]:
    cudnn = torch.backends.cudnn
    matmul = torch.backends.cuda.matmul
    previous_cudnn = cudnn.allow_tf32
    previous_matmul = matmul.allow_tf32
    cudnn.allow_tf32 = False
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        cudnn.allow_tf32 = previous_cudnn
        matmul.allow_tf32 = previous_matmul


__all__ = [
    "DeterminismReport",
    "EncoderPrefix",
    "FeatureExtractJob",
    "GLM4PreVQTokenizer",
    "PRE_VQ_EXTRACTION_POINT",
    "PRE_VQ_IMPLEMENTATION_ID",
    "PaddedPreVQTargets",
    "WhisperVQGeometry",
    "codebook_sha256",
    "nearest_codebook_ids",
    "official_codebook_ids",
    "whisper_vq_geometry",
]
