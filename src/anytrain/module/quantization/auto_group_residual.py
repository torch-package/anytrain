from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anytrain._buffer import register_buffer
from anytrain._compat import strict_zip

from . import _checks
from ._product_codebook import (
    group_indices_to_indices as _group_indices_to_indices,
)
from ._product_codebook import (
    indices_to_group_indices as _indices_to_group_indices,
)
from ._product_codebook import (
    mixed_radix_basis,
    product_codebook_size,
    uniform_group_sizes,
)
from ._residual_runner import run_residual_quantization, sample_active_mask
from .lookup import nearest_codebook_indices
from .output import QuantizationLoss, QuantizeOutput
from .projection import make_projection

ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class AGRVQConfig:
    """Configuration for auto-group residual vector quantization.

    group_size is the per-group size. The public flat codebook size is
    group_size ** num_groups.
    """

    input_dim: int
    num_codebooks: int
    group_size: int
    codebook_dim: int = 8
    num_groups: int = 2
    normalize_latents: bool = True
    projection_bias: bool = True
    weight_norm: bool = False
    dropout: float | None = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {self.num_codebooks}.")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")
        if self.codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {self.codebook_dim}.")
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {self.num_groups}.")
        if self.input_dim % self.num_groups != 0:
            raise ValueError(
                "input_dim must be divisible by num_groups for AGRVQ projection: "
                f"got input_dim={self.input_dim}, num_groups={self.num_groups}."
            )
        public_codebook_size = product_codebook_size(
            uniform_group_sizes(self.group_size, self.num_groups)
        )
        if public_codebook_size > torch.iinfo(torch.long).max:
            raise ValueError(
                "flat AGRVQ codebook_size must fit in torch.long: "
                f"got {public_codebook_size}."
            )
        _checks.dropout(self.dropout)


class AutoGroupResidualVectorQuantizer(nn.Module):
    config: AGRVQConfig
    quantizers: nn.ModuleList
    _index_basis: Tensor

    def __init__(self, config: AGRVQConfig) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.input_dim = config.input_dim
        self.num_groups = config.num_groups
        self.group_size = config.group_size
        group_sizes = uniform_group_sizes(config.group_size, config.num_groups)
        self.codebook_size = product_codebook_size(group_sizes)
        self.group_codebook_dim = config.codebook_dim
        self.codebook_dim = config.num_groups * config.codebook_dim
        self.dropout = config.dropout
        register_buffer(
            self,
            "_index_basis",
            mixed_radix_basis(group_sizes),
            persistent=False,
        )
        self.quantizers = nn.ModuleList(
            [
                _AutoGroupVectorQuantizer(
                    input_dim=config.input_dim,
                    group_size=config.group_size,
                    codebook_dim=config.codebook_dim,
                    num_groups=config.num_groups,
                    normalize_latents=config.normalize_latents,
                    projection_bias=config.projection_bias,
                    weight_norm=config.weight_norm,
                )
                for _ in range(config.num_codebooks)
            ]
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
        for quantizer in _typed_module_list(self.quantizers, _AutoGroupVectorQuantizer)[
            :active_count
        ]:
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
        for index, quantizer in enumerate(
            _typed_module_list(self.quantizers, _AutoGroupVectorQuantizer)[
                : codebook_vectors.shape[-2]
            ]
        ):
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
        for index, quantizer in enumerate(
            _typed_module_list(self.quantizers, _AutoGroupVectorQuantizer)[
                : indices.shape[-1]
            ]
        ):
            indices_i = indices[..., index]
            mask = indices_i >= 0
            vectors_i = torch.zeros(
                *indices_i.shape,
                self.codebook_dim,
                dtype=_typed_module_list(quantizer.codebooks, nn.Embedding)[0].weight.dtype,
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
        for index, quantizer in enumerate(
            _typed_module_list(self.quantizers, _AutoGroupVectorQuantizer)[
                : codebook_vectors.shape[-2]
            ]
        ):
            projected.append(quantizer.project_codebook_vectors(codebook_vectors[..., index, :]))
        return torch.stack(projected, dim=-2).sum(dim=-2)

    def indices_to_group_indices(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size)
        return _indices_to_group_indices(
            indices,
            index_basis=self._index_basis,
            group_sizes=self.group_size,
        )

    def group_indices_to_indices(self, group_indices: Tensor) -> Tensor:
        self._validate_group_indices(group_indices)
        return _group_indices_to_indices(group_indices, index_basis=self._index_basis)

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

    def set_dropout(self, dropout: float | None) -> AutoGroupResidualVectorQuantizer:
        _checks.dropout(dropout)
        self.dropout = dropout
        return self

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
        min_index = int(group_indices.min().item())
        max_index = int(group_indices.max().item())
        if min_index < 0 or max_index >= self.group_size:
            raise ValueError(
                f"group_indices must be in [0, {self.group_size - 1}]: "
                f"got min={min_index}, max={max_index}."
            )


class _AutoGroupVectorQuantizer(nn.Module):
    _index_basis: Tensor

    def __init__(
        self,
        *,
        input_dim: int,
        group_size: int,
        codebook_dim: int,
        num_groups: int,
        normalize_latents: bool,
        projection_bias: bool,
        weight_norm: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.group_size = group_size
        group_sizes = uniform_group_sizes(group_size, num_groups)
        self.codebook_size = product_codebook_size(group_sizes)
        self.group_codebook_dim = codebook_dim
        self.num_groups = num_groups
        self.codebook_dim = num_groups * codebook_dim
        self.output_dim = input_dim // num_groups
        self.normalize_latents = normalize_latents
        register_buffer(
            self,
            "_index_basis",
            mixed_radix_basis(group_sizes),
            persistent=False,
        )

        self.in_projs = nn.ModuleList(
            [
                make_projection(
                    input_dim,
                    codebook_dim,
                    bias=projection_bias,
                    weight_norm=weight_norm,
                )
                for _ in range(num_groups)
            ]
        )
        self.out_projs = nn.ModuleList(
            [
                make_projection(
                    codebook_dim,
                    self.output_dim,
                    bias=projection_bias,
                    weight_norm=weight_norm,
                )
                for _ in range(num_groups)
            ]
        )
        self.codebooks = nn.ModuleList(
            [nn.Embedding(group_size, codebook_dim) for _ in range(num_groups)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for codebook in _typed_module_list(self.codebooks, nn.Embedding):
                nn.init.normal_(codebook.weight, mean=0.0, std=1.0)

    def forward(self, latents: Tensor) -> QuantizeOutput:
        projected = [project(latents) for project in self.in_projs]
        nearest = [
            self._nearest_codebook_vectors(projected_i, codebook)
            for projected_i, codebook in strict_zip(
                projected,
                _typed_module_list(self.codebooks, nn.Embedding),
            )
        ]
        vectors = [vectors_i for vectors_i, _ in nearest]
        group_indices = torch.stack([indices_i for _, indices_i in nearest], dim=-1)

        loss = None
        if self.training:
            projected_for_loss = torch.cat(projected, dim=-1)
            vectors_for_loss = torch.cat(vectors, dim=-1)
            loss = QuantizationLoss(
                commitment=F.mse_loss(projected_for_loss, vectors_for_loss.detach()),
                codebook=F.mse_loss(vectors_for_loss, projected_for_loss.detach()),
            )

        straight = [
            projected_i + (vectors_i - projected_i).detach()
            for projected_i, vectors_i in strict_zip(projected, vectors)
        ]
        quantized_latents = torch.cat(
            [
                project(straight_i)
                for project, straight_i in strict_zip(self.out_projs, straight)
            ],
            dim=-1,
        )
        return QuantizeOutput(
            quantized_latents=quantized_latents,
            indices=self._group_indices_to_indices(group_indices),
            codebook_vectors=torch.cat(vectors, dim=-1),
            latents=torch.cat(projected, dim=-1),
            loss=loss,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        vectors = []
        for project, codebook in strict_zip(
            self.in_projs,
            _typed_module_list(self.codebooks, nn.Embedding),
        ):
            vectors_i, _ = self._nearest_codebook_vectors(project(latents), codebook)
            vectors.append(vectors_i)
        return torch.cat(vectors, dim=-1)

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(codebook_vectors, self.codebook_dim)
        vectors = codebook_vectors.split(self.group_codebook_dim, dim=-1)
        group_indices = []
        for vectors_i, codebook in strict_zip(
            vectors,
            _typed_module_list(self.codebooks, nn.Embedding),
        ):
            _, indices_i = self._nearest_codebook_vectors(vectors_i, codebook)
            group_indices.append(indices_i)
        return self._group_indices_to_indices(torch.stack(group_indices, dim=-1))

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size)
        group_indices = self._indices_to_group_indices(indices)
        return torch.cat(
            [
                codebook(group_indices[..., index])
                for index, codebook in enumerate(
                    _typed_module_list(self.codebooks, nn.Embedding)
                )
            ],
            dim=-1,
        )

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(codebook_vectors, self.codebook_dim)
        vectors = codebook_vectors.split(self.group_codebook_dim, dim=-1)
        return torch.cat(
            [
                project(vectors_i)
                for project, vectors_i in strict_zip(self.out_projs, vectors)
            ],
            dim=-1,
        )

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
        return codebook(indices).reshape(*leading_shape, self.group_codebook_dim), indices

    def _indices_to_group_indices(self, indices: Tensor) -> Tensor:
        return _indices_to_group_indices(
            indices,
            index_basis=self._index_basis,
            group_sizes=self.group_size,
        )

    def _group_indices_to_indices(self, group_indices: Tensor) -> Tensor:
        return _group_indices_to_indices(group_indices, index_basis=self._index_basis)


def _typed_module_list(
    modules: nn.ModuleList,
    module_type: type[ModuleT],
) -> tuple[ModuleT, ...]:
    typed: list[ModuleT] = []
    for module in modules:
        if not isinstance(module, module_type):
            raise TypeError(
                f"expected {module_type.__name__} in ModuleList, got {type(module)!r}."
            )
        typed.append(module)
    return tuple(typed)
