from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import isfinite
from numbers import Real
from operator import mul

import torch
from torch import Tensor, nn

from anytrain._buffer import register_buffer

from . import _checks
from .finite_scalar import DEFAULT_FSQ_LEVELS, FSQConfig
from .output import QuantizeOutput
from .projection import make_projection


@dataclass(frozen=True)
class RFSQConfig:
    input_dim: int
    levels: tuple[int, ...] = DEFAULT_FSQ_LEVELS
    scales: tuple[float, ...] = (1.0,)
    projection_bias: bool = True

    def __post_init__(self) -> None:
        # Reuse the public FSQ config validation for the scalar grid. RFSQ uses
        # its own scale-space quantization but shares the same level/index
        # contract as ordinary FSQ.
        fsq = FSQConfig(
            input_dim=self.input_dim,
            levels=tuple(self.levels),
            projection_bias=self.projection_bias,
        )
        object.__setattr__(self, "levels", fsq.levels)

        raw_scales = tuple(self.scales)
        if not raw_scales:
            raise ValueError("scales must contain at least one value.")
        scales: list[float] = []
        for scale in raw_scales:
            if isinstance(scale, bool) or not isinstance(scale, Real):
                raise TypeError(f"each residual FSQ scale must be real, got {scale!r}.")
            value = float(scale)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"each residual FSQ scale must be positive, got {scale!r}.")
            scales.append(value)
        object.__setattr__(self, "scales", tuple(scales))


class ResidualFiniteScalarQuantizer(nn.Module):
    """Residual finite scalar quantizer with a shared scalar grid.

    Each residual stage quantizes into the same mixed-radix FSQ codebook while
    allowing stage-specific scalar ranges via the configured scales. This keeps
    codec-specific wrapper code thin while exposing the core residual FSQ logic
    as an ordinary anytrain module.
    """

    config: RFSQConfig
    project_in: nn.Module
    project_out: nn.Module
    _levels: Tensor
    _basis: Tensor
    _dtype: Tensor

    def __init__(self, config: RFSQConfig) -> None:
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.input_dim = config.input_dim
        self.quantize_dim = len(config.levels)
        self.codebook_dim = self.quantize_dim
        self.codebook_size = reduce(mul, config.levels, 1)
        self.num_codebooks = len(config.scales)
        self.project_in = make_projection(
            self.input_dim,
            self.quantize_dim,
            bias=config.projection_bias,
        )
        self.project_out = make_projection(
            self.quantize_dim,
            self.input_dim,
            bias=config.projection_bias,
        )

        levels = torch.tensor(config.levels, dtype=torch.long)
        register_buffer(self, "_levels", levels, persistent=False)
        register_buffer(
            self,
            "_basis",
            torch.cumprod(torch.tensor((1, *config.levels[:-1]), dtype=torch.long), dim=0),
            persistent=False,
        )
        register_buffer(self, "_dtype", torch.empty(0), persistent=False)

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.num_codebooks

    def forward(self, latents: Tensor) -> QuantizeOutput:
        return self.quantize(latents)

    def quantize(self, latents: Tensor) -> QuantizeOutput:
        _checks.input_latents(latents, self.input_dim)
        dtype = latents.dtype
        projected = self.project_in(latents)
        residual = torch.tanh(projected.to(torch.float64))
        vectors: list[Tensor] = []
        indices: list[Tensor] = []

        for scale in self.scales:
            vector, index = self._quantize_residual_stage(residual, scale)
            vectors.append(vector)
            indices.append(index)
            residual = residual - vector

        stacked = torch.stack(vectors, dim=-2).to(dtype=dtype)
        return QuantizeOutput(
            quantized_latents=self.project_codebook_vectors(stacked),
            indices=torch.stack(indices, dim=-1),
            codebook_vectors=stacked,
            latents=projected,
            loss=None,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        codebook_vectors = self.quantize(latents).codebook_vectors
        if codebook_vectors is None:
            raise RuntimeError("Residual FSQ output unexpectedly omitted codebook_vectors.")
        return codebook_vectors

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.active_codebook_vectors(
            codebook_vectors,
            codebook_dim=self.codebook_dim,
            num_codebooks=self.num_codebooks,
        )
        indices = []
        for stage, scale in enumerate(self.scales[: codebook_vectors.shape[-2]]):
            levels = self.codebook_vectors_to_levels(codebook_vectors[..., stage, :], scale)
            indices.append(self.levels_to_indices(levels))
        return torch.stack(indices, dim=-1)

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        self._validate_indices(indices)
        vectors: list[Tensor] = []
        for stage, scale in enumerate(self.scales[: indices.shape[-1]]):
            levels = self.indices_to_levels(indices[..., stage].long())
            vectors.append(self.levels_to_codebook_vectors(levels, scale))
        return torch.stack(vectors, dim=-2)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.active_codebook_vectors(
            codebook_vectors,
            codebook_dim=self.codebook_dim,
            num_codebooks=self.num_codebooks,
        )
        return self.project_out(
            codebook_vectors.sum(dim=-2).to(dtype=self._out_dtype(codebook_vectors))
        )

    def indices_to_levels(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size, require_ndim=True)
        return (indices.long().unsqueeze(-1) // self._basis) % self._levels

    def levels_to_indices(self, levels: Tensor) -> Tensor:
        self._validate_levels(levels)
        return (levels.long() * self._basis).sum(dim=-1)

    def levels_to_codebook_vectors(self, levels: Tensor, scale: float = 1.0) -> Tensor:
        self._validate_levels(levels)
        step = self._step(scale, self._levels)
        return levels.to(dtype=step.dtype) * step - scale

    def codebook_vectors_to_levels(self, codebook_vectors: Tensor, scale: float = 1.0) -> Tensor:
        _checks.codebook_vectors(
            codebook_vectors,
            self.codebook_dim,
            require_non_empty=False,
        )
        step = self._step(scale, codebook_vectors)
        levels = ((codebook_vectors + scale) / step).round().long()
        return torch.minimum(levels.clamp_min(0), self._levels - 1)

    def _quantize_residual_stage(self, residual: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        step = self._step(scale, residual)
        levels = (residual + scale) / step
        rounded = levels + (levels.round() - levels).detach()
        vector = rounded * step - scale
        index = (rounded.long() * self._basis).sum(dim=-1)
        return vector, index

    def _step(self, scale: float, like: Tensor) -> Tensor:
        dtype = like.dtype if torch.is_floating_point(like) else self._dtype.dtype
        return (2 * scale / (self._levels - 1)).to(device=like.device, dtype=dtype)

    def _out_dtype(self, like: Tensor) -> torch.dtype:
        if isinstance(self.project_out, nn.Identity):
            return like.dtype if torch.is_floating_point(like) else torch.float32
        weight = self.project_out.weight
        return weight.dtype

    def _validate_indices(self, indices: Tensor) -> None:
        if indices.ndim == 0:
            raise ValueError(
                f"indices must end with num_codebooks={self.num_codebooks}, got scalar."
            )
        if indices.shape[-1] <= 0 or indices.shape[-1] > self.num_codebooks:
            raise ValueError(
                f"indices active dimension must be in [1, {self.num_codebooks}], "
                f"got {indices.shape[-1]}."
            )
        if indices.dtype is torch.bool or torch.is_floating_point(indices) or torch.is_complex(indices):
            raise TypeError("indices must be an integer tensor.")
        if indices.numel() and bool(((indices < 0) | (indices >= self.codebook_size)).any().item()):
            raise ValueError(
                "indices must be in [0, codebook_size - 1]: "
                f"codebook_size={self.codebook_size}."
            )

    def _validate_levels(self, levels: Tensor) -> None:
        if levels.ndim == 0:
            raise ValueError(f"levels must end with codebook_dim={self.codebook_dim}.")
        if levels.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"levels must end with codebook_dim={self.codebook_dim}, "
                f"got {tuple(levels.shape)}."
            )
        if levels.dtype is torch.bool or torch.is_floating_point(levels) or torch.is_complex(levels):
            raise TypeError("levels must be an integer tensor.")
        if levels.numel() and bool(((levels < 0) | (levels >= self._levels)).any().item()):
            raise ValueError("levels contain values outside the configured FSQ level ranges.")
