from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import _checks
from ._residual_runner import run_residual_quantization, sample_active_mask
from .lookup import nearest_codebook_indices
from .output import QuantizationLoss, QuantizeOutput
from .projection import make_projection


@dataclass
class AGRVQConfig:
    input_dim: int
    num_codebooks: int
    codebook_size: int
    codebook_dim: int = 8
    normalize_latents: bool = True
    projection_bias: bool = True
    weight_norm: bool = False
    dropout: float | None = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.input_dim % 2 != 0:
            raise ValueError("input_dim must be even for two-group AGRVQ projection.")
        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {self.num_codebooks}.")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}.")
        if self.codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {self.codebook_dim}.")
        if self.dropout is not None and not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}.")

    @classmethod
    def from_kwargs(
        cls,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int = 8,
        normalize_latents: bool = True,
        projection_bias: bool = True,
        weight_norm: bool = False,
        dropout: float | None = None,
    ) -> AGRVQConfig:
        return cls(
            input_dim=input_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            normalize_latents=normalize_latents,
            projection_bias=projection_bias,
            weight_norm=weight_norm,
            dropout=dropout,
        )


class AutoGroupResidualVectorQuantizer(nn.Module):
    config: AGRVQConfig
    quantizers: nn.ModuleList

    def __init__(self, config: AGRVQConfig) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.input_dim = config.input_dim
        self.group_size = config.codebook_size
        self.codebook_size = config.codebook_size * config.codebook_size
        self.group_codebook_dim = config.codebook_dim
        self.codebook_dim = 2 * config.codebook_dim
        self.quantizers = nn.ModuleList(
            [
                _AutoGroupVectorQuantizer(
                    input_dim=config.input_dim,
                    codebook_size=config.codebook_size,
                    codebook_dim=config.codebook_dim,
                    normalize_latents=config.normalize_latents,
                    projection_bias=config.projection_bias,
                    weight_norm=config.weight_norm,
                )
                for _ in range(config.num_codebooks)
            ]
        )

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
            dropout=self.config.dropout,
            sample_mask=self._sample_active_mask,
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

        vectors = []
        for quantizer in self.quantizers[:active_count]:
            vectors_i = quantizer.latents_to_codebook_vectors(residual)
            vectors.append(vectors_i)
            residual = residual - quantizer.project_codebook_vectors(vectors_i)
        return torch.stack(vectors, dim=1).reshape(
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
                dtype=quantizer.codebook_a.weight.dtype,
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

    def indices_to_group_indices(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size)
        return torch.stack(
            [indices // self.group_size, indices % self.group_size],
            dim=-1,
        )

    def group_indices_to_indices(self, group_indices: Tensor) -> Tensor:
        self._validate_group_indices(group_indices)
        return group_indices[..., 0] * self.group_size + group_indices[..., 1]

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
            dropout=self.config.dropout,
        )

    def _active_codebooks(self, num_active_codebooks: int | None) -> int:
        if num_active_codebooks is None:
            return self.num_codebooks
        if num_active_codebooks <= 0 or num_active_codebooks > self.num_codebooks:
            raise ValueError(
                "num_active_codebooks must be in [1, num_codebooks]: "
                f"got {num_active_codebooks}, num_codebooks={self.num_codebooks}."
            )
        return num_active_codebooks

    def _validate_group_indices(self, group_indices: Tensor) -> None:
        if group_indices.ndim == 0 or group_indices.shape[-1] != 2:
            raise ValueError("group_indices must end with two group ids.")
        if torch.is_floating_point(group_indices) or torch.is_complex(group_indices):
            raise TypeError("group_indices must be an integer tensor.")
        if group_indices.numel() == 0:
            raise ValueError("group_indices must contain at least one value.")
        min_index = int(group_indices.min().item())
        max_index = int(group_indices.max().item())
        if min_index < 0 or max_index >= self.group_size:
            raise ValueError(
                f"group_indices must be in [0, {self.group_size - 1}]: "
                f"got min={min_index}, max={max_index}."
            )


class _AutoGroupVectorQuantizer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        normalize_latents: bool,
        projection_bias: bool,
        weight_norm: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.output_dim = input_dim // 2
        self.normalize_latents = normalize_latents

        self.in_proj_a = make_projection(
            input_dim,
            codebook_dim,
            bias=projection_bias,
            weight_norm=weight_norm,
        )
        self.out_proj_a = make_projection(
            codebook_dim,
            self.output_dim,
            bias=projection_bias,
            weight_norm=weight_norm,
        )
        self.in_proj_b = make_projection(
            input_dim,
            codebook_dim,
            bias=projection_bias,
            weight_norm=weight_norm,
        )
        self.out_proj_b = make_projection(
            codebook_dim,
            self.output_dim,
            bias=projection_bias,
            weight_norm=weight_norm,
        )
        self.codebook_a = nn.Embedding(codebook_size, codebook_dim)
        self.codebook_b = nn.Embedding(codebook_size, codebook_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.codebook_a.weight, mean=0.0, std=1.0)
            nn.init.normal_(self.codebook_b.weight, mean=0.0, std=1.0)

    def forward(self, latents: Tensor) -> QuantizeOutput:
        projected_a = self.in_proj_a(latents)
        projected_b = self.in_proj_b(latents)
        vectors_a, indices_a = self._nearest_codebook_vectors(projected_a, self.codebook_a)
        vectors_b, indices_b = self._nearest_codebook_vectors(projected_b, self.codebook_b)

        loss = None
        if self.training:
            projected = torch.cat([projected_a, projected_b], dim=-1)
            vectors = torch.cat([vectors_a, vectors_b], dim=-1)
            loss = QuantizationLoss(
                commitment=F.mse_loss(projected, vectors.detach()),
                codebook=F.mse_loss(vectors, projected.detach()),
            )

        straight_a = projected_a + (vectors_a - projected_a).detach()
        straight_b = projected_b + (vectors_b - projected_b).detach()
        quantized_latents = torch.cat(
            [self.out_proj_a(straight_a), self.out_proj_b(straight_b)],
            dim=-1,
        )
        return QuantizeOutput(
            quantized_latents=quantized_latents,
            indices=indices_a * self.codebook_size + indices_b,
            codebook_vectors=torch.cat([vectors_a, vectors_b], dim=-1),
            latents=torch.cat([projected_a, projected_b], dim=-1),
            loss=loss,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        projected_a = self.in_proj_a(latents)
        projected_b = self.in_proj_b(latents)
        vectors_a, _ = self._nearest_codebook_vectors(projected_a, self.codebook_a)
        vectors_b, _ = self._nearest_codebook_vectors(projected_b, self.codebook_b)
        return torch.cat([vectors_a, vectors_b], dim=-1)

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        vectors_a, vectors_b = codebook_vectors.split(self.codebook_dim, dim=-1)
        _, indices_a = self._nearest_codebook_vectors(vectors_a, self.codebook_a)
        _, indices_b = self._nearest_codebook_vectors(vectors_b, self.codebook_b)
        return indices_a * self.codebook_size + indices_b

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        indices_a = indices // self.codebook_size
        indices_b = indices % self.codebook_size
        return torch.cat([self.codebook_a(indices_a), self.codebook_b(indices_b)], dim=-1)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        vectors_a, vectors_b = codebook_vectors.split(self.codebook_dim, dim=-1)
        return torch.cat([self.out_proj_a(vectors_a), self.out_proj_b(vectors_b)], dim=-1)

    def _nearest_codebook_vectors(
        self,
        projected_latents: Tensor,
        codebook: nn.Embedding,
    ) -> tuple[Tensor, Tensor]:
        leading_shape = projected_latents.shape[:-1]
        indices = nearest_codebook_indices(
            projected_latents,
            codebook.weight,
            normalize=self.normalize_latents,
        )
        return codebook(indices).reshape(*leading_shape, self.codebook_dim), indices
