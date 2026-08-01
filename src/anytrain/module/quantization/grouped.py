from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anytrain._buffer import register_buffer
from anytrain._compat import strict_zip

from . import _checks
from .lookup import nearest_codebook_indices
from .output import QuantizationLoss, QuantizeOutput
from .projection import make_projection


@dataclass(frozen=True)
class GVQConfig:
    input_dim: int
    group_sizes: tuple[int, ...]
    codebook_dim: int | None = None
    group_dims: tuple[int, ...] | None = None
    normalize_latents: bool = True
    weight_norm: bool = False
    projection_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if not self.group_sizes:
            raise ValueError("group_sizes must contain at least one value.")
        for group_size in self.group_sizes:
            if group_size <= 0:
                raise ValueError(f"each group size must be positive, got {group_size}.")

        codebook_dim = self.input_dim if self.codebook_dim is None else self.codebook_dim
        if codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {codebook_dim}.")
        object.__setattr__(self, "codebook_dim", codebook_dim)

        if self.group_dims is None:
            group_dims = _balanced_group_dims(codebook_dim, len(self.group_sizes))
        else:
            group_dims = tuple(self.group_dims)
        object.__setattr__(self, "group_dims", group_dims)

        if len(group_dims) != len(self.group_sizes):
            raise ValueError(
                "group_dims must have the same length as group_sizes: "
                f"got {len(group_dims)} and {len(self.group_sizes)}."
            )
        if sum(group_dims) != codebook_dim:
            raise ValueError(
                f"sum(group_dims) must equal codebook_dim={codebook_dim}, "
                f"got {sum(group_dims)}."
            )
        for group_dim in group_dims:
            if group_dim <= 0:
                raise ValueError(f"each group dim must be positive, got {group_dim}.")

        object.__setattr__(self, "group_sizes", tuple(int(size) for size in self.group_sizes))
        object.__setattr__(self, "group_dims", tuple(int(dim) for dim in group_dims))


class GroupedVectorQuantizer(nn.Module):
    config: GVQConfig
    project_in: nn.Module
    project_out: nn.Module
    codebooks: nn.ParameterList
    _basis: Tensor
    _group_sizes: Tensor

    def __init__(self, config: GVQConfig) -> None:
        super().__init__()
        codebook_dim = config.codebook_dim
        group_dims = config.group_dims
        if codebook_dim is None:
            raise RuntimeError("GVQConfig.codebook_dim should be resolved in __post_init__.")
        if group_dims is None:
            raise RuntimeError("GVQConfig.group_dims should be resolved in __post_init__.")

        self.config = config
        self.input_dim = config.input_dim
        self.codebook_dim = codebook_dim
        self.group_sizes = config.group_sizes
        self.group_dims = group_dims
        self.num_groups = len(config.group_sizes)
        self.num_codebooks = 1
        self.codebook_size = reduce(mul, config.group_sizes, 1)

        self.project_in = make_projection(
            config.input_dim,
            codebook_dim,
            bias=config.projection_bias,
            weight_norm=config.weight_norm,
        )
        self.project_out = make_projection(
            codebook_dim,
            config.input_dim,
            bias=config.projection_bias,
            weight_norm=config.weight_norm,
        )
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.empty(size, dim)) for size, dim in strict_zip(config.group_sizes, group_dims)]
        )
        register_buffer(
            self,
            "_basis",
            torch.cumprod(torch.tensor([1, *config.group_sizes[:-1]]), dim=0),
        )
        register_buffer(
            self,
            "_group_sizes",
            torch.tensor(config.group_sizes),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for codebook in self.codebooks:
                nn.init.normal_(codebook, mean=0.0, std=1.0)

    def forward(self, latents: Tensor) -> QuantizeOutput:
        return self.quantize(latents)

    def quantize(self, latents: Tensor) -> QuantizeOutput:
        _checks.input_latents(latents, self.input_dim)
        projected_latents = self.project_in(latents)
        codebook_vectors, group_indices = self._nearest_group_vectors(projected_latents)
        indices = self.group_indices_to_indices(group_indices)
        loss = None
        if self.training:
            loss = QuantizationLoss(
                commitment=F.mse_loss(projected_latents, codebook_vectors.detach()),
                codebook=F.mse_loss(codebook_vectors, projected_latents.detach()),
            )

        straight_through = projected_latents + (codebook_vectors - projected_latents).detach()
        quantized_latents = self.project_out(straight_through)
        return QuantizeOutput(
            quantized_latents=quantized_latents,
            indices=indices,
            codebook_vectors=codebook_vectors,
            latents=projected_latents,
            loss=loss,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        _checks.input_latents(latents, self.input_dim)
        codebook_vectors, _ = self._nearest_group_vectors(self.project_in(latents))
        return codebook_vectors

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(codebook_vectors, self.codebook_dim)
        _, group_indices = self._nearest_group_vectors(codebook_vectors)
        return self.group_indices_to_indices(group_indices)

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        group_indices = self.indices_to_group_indices(indices)
        return self.group_indices_to_codebook_vectors(group_indices)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(codebook_vectors, self.codebook_dim)
        return self.project_out(codebook_vectors)

    def indices_to_group_indices(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size)
        basis = self._basis.to(device=indices.device)
        group_sizes = self._group_sizes.to(device=indices.device, dtype=indices.dtype)
        return (indices.unsqueeze(-1) // basis) % group_sizes

    def group_indices_to_indices(self, group_indices: Tensor) -> Tensor:
        self._validate_group_indices(group_indices)
        return (group_indices * self._basis.to(device=group_indices.device)).sum(dim=-1)

    def group_indices_to_codebook_vectors(self, group_indices: Tensor) -> Tensor:
        self._validate_group_indices(group_indices)
        vectors = []
        for group_index, codebook in enumerate(self.codebooks):
            vectors.append(codebook[group_indices[..., group_index]])
        return torch.cat(vectors, dim=-1)

    @classmethod
    def from_kwargs(
        cls,
        input_dim: int,
        group_sizes: tuple[int, ...] | list[int],
        codebook_dim: int | None = None,
        group_dims: tuple[int, ...] | list[int] | None = None,
        normalize_latents: bool = True,
        weight_norm: bool = False,
        projection_bias: bool = True,
    ) -> GroupedVectorQuantizer:
        return cls(
            GVQConfig(
                input_dim=input_dim,
                group_sizes=tuple(group_sizes),
                codebook_dim=codebook_dim,
                group_dims=None if group_dims is None else tuple(group_dims),
                normalize_latents=normalize_latents,
                weight_norm=weight_norm,
                projection_bias=projection_bias,
            )
        )

    def _nearest_group_vectors(self, projected_latents: Tensor) -> tuple[Tensor, Tensor]:
        _checks.codebook_vectors(
            projected_latents,
            self.codebook_dim,
            name="projected_latents",
        )
        parts = projected_latents.split(self.group_dims, dim=-1)
        vector_parts = []
        index_parts = []
        for part, codebook in strict_zip(parts, self.codebooks):
            indices = self._nearest_indices(part, codebook)
            vector_parts.append(codebook[indices])
            index_parts.append(indices)
        return torch.cat(vector_parts, dim=-1), torch.stack(index_parts, dim=-1)

    def _nearest_indices(self, latents: Tensor, codebook: Tensor) -> Tensor:
        return nearest_codebook_indices(
            latents,
            codebook,
            normalize=self.config.normalize_latents,
        )

    def _validate_group_indices(self, group_indices: Tensor) -> None:
        if group_indices.ndim == 0:
            raise ValueError(f"group_indices must end with num_groups={self.num_groups}.")
        if group_indices.shape[-1] != self.num_groups:
            raise ValueError(
                f"group_indices must end with num_groups={self.num_groups}, "
                f"got {tuple(group_indices.shape)}."
            )
        if torch.is_floating_point(group_indices) or torch.is_complex(group_indices):
            raise TypeError("group_indices must be an integer tensor.")
        if group_indices.numel() == 0:
            raise ValueError("group_indices must contain at least one value.")
        for group_index, group_size in enumerate(self.group_sizes):
            values = group_indices[..., group_index]
            min_value = int(values.min().item())
            max_value = int(values.max().item())
            if min_value < 0 or max_value >= group_size:
                raise ValueError(
                    f"group_indices[..., {group_index}] must be in [0, {group_size - 1}], "
                    f"got min={min_value}, max={max_value}."
                )


def _balanced_group_dims(codebook_dim: int, num_groups: int) -> tuple[int, ...]:
    base = codebook_dim // num_groups
    remainder = codebook_dim % num_groups
    return tuple(base + (1 if index < remainder else 0) for index in range(num_groups))
