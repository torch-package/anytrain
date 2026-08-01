from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import _checks
from ._residual_runner import run_residual_quantization, sample_active_mask
from .embedding import EmbeddingVectorQuantizer, VQConfig
from .output import QuantizeOutput


@dataclass(frozen=True)
class RVQConfig:
    vq_configs: list[VQConfig] | tuple[VQConfig, ...]
    dropout: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vq_configs", tuple(self.vq_configs))
        if not self.vq_configs:
            raise ValueError("vq_configs must contain at least one VQConfig.")
        _checks.dropout(self.dropout)

        first = self.vq_configs[0]
        for config in self.vq_configs:
            if config.input_dim != first.input_dim:
                raise ValueError("RVQ first version requires uniform input_dim.")
            if config.codebook_dim != first.codebook_dim:
                raise ValueError("RVQ first version requires uniform codebook_dim.")
            if config.codebook_size != first.codebook_size:
                raise ValueError("RVQ first version requires uniform codebook_size.")

    @property
    def num_codebooks(self) -> int:
        return len(self.vq_configs)


class ResidualVectorQuantizer(nn.Module):
    config: RVQConfig
    quantizers: nn.ModuleList

    def __init__(self, config: RVQConfig) -> None:
        super().__init__()
        codebook_dim = config.vq_configs[0].codebook_dim
        if codebook_dim is None:
            raise RuntimeError("VQConfig.codebook_dim should be resolved in __post_init__.")

        self.config = config
        self.num_codebooks = config.num_codebooks
        self.input_dim = config.vq_configs[0].input_dim
        self.codebook_dim = codebook_dim
        self.codebook_size = config.vq_configs[0].codebook_size
        self.dropout = config.dropout
        self.quantizers = nn.ModuleList(
            [EmbeddingVectorQuantizer(vq_config) for vq_config in config.vq_configs]
        )

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.num_codebooks

    def forward(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> QuantizeOutput:
        return self.quantize(latents, num_active_codebooks=num_active_codebooks)

    def quantize(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> QuantizeOutput:
        _checks.input_latents(latents, self.input_dim)
        active_count = self._active_codebooks(num_active_codebooks)
        leading_shape = latents.shape[:-1]
        flat_latents = latents.reshape(-1, self.input_dim)
        return run_residual_quantization(
            flat_latents=flat_latents,
            leading_shape=leading_shape,
            active_count=active_count,
            quantizers=self.quantizers,
            input_dim=self.input_dim,
            codebook_dim=self.codebook_dim,
            training=self.training,
            dropout=self.dropout,
            sample_mask=self._sample_active_mask,
            on_inactive=self._on_inactive_quantizer,
        )

    def latents_to_codebook_vectors(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> Tensor:
        _checks.input_latents(latents, self.input_dim)
        active_count = self._active_codebooks(num_active_codebooks)
        leading_shape = latents.shape[:-1]
        residual = latents.reshape(-1, self.input_dim)

        vector_list = []
        for quantizer in self.quantizers[:active_count]:
            codebook_vectors = quantizer.latents_to_codebook_vectors(residual)
            vector_list.append(codebook_vectors)
            residual = residual - quantizer.project_codebook_vectors(codebook_vectors)

        return torch.stack(vector_list, dim=1).reshape(
            *leading_shape,
            active_count,
            self.codebook_dim,
        )

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.active_codebook_vectors(
            codebook_vectors,
            codebook_dim=self.codebook_dim,
            num_codebooks=self.num_codebooks,
        )
        indices = []
        for index, quantizer in enumerate(self.quantizers[: codebook_vectors.shape[-2]]):
            indices.append(quantizer.codebook_vectors_to_indices(codebook_vectors[..., index, :]))
        return torch.stack(indices, dim=-1)

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        _checks.active_indices(
            indices,
            codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            allow_inactive=True,
        )
        vectors = []
        for index, quantizer in enumerate(self.quantizers[: indices.shape[-1]]):
            indices_i = indices[..., index]
            mask = indices_i >= 0
            vectors_i = torch.zeros(
                *indices_i.shape,
                self.codebook_dim,
                dtype=quantizer.codebook.weight.dtype,
                device=indices.device,
            )
            if bool(mask.any().item()):
                vectors_i[mask] = quantizer.indices_to_codebook_vectors(indices_i[mask])
            vectors.append(vectors_i)
        return torch.stack(vectors, dim=-2)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.active_codebook_vectors(
            codebook_vectors,
            codebook_dim=self.codebook_dim,
            num_codebooks=self.num_codebooks,
        )
        projected = []
        for index, quantizer in enumerate(self.quantizers[: codebook_vectors.shape[-2]]):
            projected.append(quantizer.project_codebook_vectors(codebook_vectors[..., index, :]))
        return torch.stack(projected, dim=-2).sum(dim=-2)

    def _sample_active_mask(
        self,
        flat_size: int,
        active_count: int,
        device: torch.device,
    ) -> Tensor:
        return sample_active_mask(
            flat_size,
            active_count,
            device,
            training=self.training,
            dropout=self.dropout,
        )

    def set_dropout(self, dropout: float | None) -> ResidualVectorQuantizer:
        _checks.dropout(dropout)
        self.dropout = dropout
        return self

    @staticmethod
    def _on_inactive_quantizer(quantizer: nn.Module) -> None:
        if not isinstance(quantizer, EmbeddingVectorQuantizer):
            raise TypeError("RVQ quantizers must be EmbeddingVectorQuantizer instances.")
        if quantizer.training and quantizer.use_ema:
            # Match peer EMA collectives when this rank has no assignments.
            quantizer._update_ema_without_assignments()

    def _active_codebooks(self, num_active_codebooks: int | None) -> int:
        if num_active_codebooks is None:
            return self.num_codebooks
        if num_active_codebooks <= 0 or num_active_codebooks > self.num_codebooks:
            raise ValueError(
                "num_active_codebooks must be in [1, num_codebooks]: "
                f"got {num_active_codebooks}, num_codebooks={self.num_codebooks}."
            )
        return num_active_codebooks
