from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import reduce
from numbers import Integral
from operator import mul
from typing import Literal

import torch
from torch import Tensor, nn

from anytrain._buffer import register_buffer

from . import _checks
from .output import QuantizeOutput
from .projection import make_projection

DEFAULT_FSQ_LEVELS: tuple[int, ...] = (9, 7, 7, 7, 7, 3)


@dataclass
class FSQConfig:
    input_dim: int
    levels: tuple[int, ...] = DEFAULT_FSQ_LEVELS
    bound_scale: float = 1.0
    eps: float = 1e-3
    projection_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.bound_scale <= 0:
            raise ValueError(f"bound_scale must be positive, got {self.bound_scale}.")
        if self.eps <= 0 or self.eps >= 1:
            raise ValueError(f"eps must be in (0, 1), got {self.eps}.")

        raw_levels = tuple(self.levels)
        if not raw_levels:
            raise ValueError("levels must contain at least one value.")
        levels = []
        for level in raw_levels:
            if isinstance(level, bool) or not isinstance(level, Integral):
                raise TypeError(f"each FSQ level must be an integer, got {level!r}.")
            if level < 2:
                raise ValueError(f"each FSQ level must be >= 2, got {level}.")
            levels.append(int(level))
        codebook_size = reduce(mul, levels, 1)
        if codebook_size > torch.iinfo(torch.long).max:
            raise ValueError("FSQ codebook_size must fit in torch.long.")
        self.levels = tuple(levels)
        even_levels = tuple(level for level in self.levels if level % 2 == 0)
        if even_levels:
            warnings.warn(
                "FSQ levels should preferably be odd so the scalar grid stays symmetric "
                "around zero. Even levels use a zero-friendly offset grid instead.",
                stacklevel=2,
            )


class FiniteScalarQuantizer(nn.Module):
    config: FSQConfig
    project_in: nn.Module
    project_out: nn.Module
    _basis: Tensor
    _levels: Tensor
    _half_width: Tensor
    _half_levels: Tensor
    _offsets: Tensor
    _shifts: Tensor
    _level_mask: Tensor

    def __init__(self, config: FSQConfig) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = 1
        self.input_dim = config.input_dim
        self.levels = config.levels
        self.codebook_dim = len(config.levels)
        self.codebook_size = reduce(mul, config.levels, 1)

        self.project_in = make_projection(
            self.input_dim,
            self.codebook_dim,
            bias=config.projection_bias,
        )
        self.project_out = make_projection(
            self.codebook_dim,
            self.input_dim,
            bias=config.projection_bias,
        )

        levels = torch.tensor(config.levels, dtype=torch.long)
        register_buffer(
            self,
            "_basis",
            torch.cumprod(torch.tensor([1, *config.levels[:-1]], dtype=torch.long), dim=0),
        )
        register_buffer(self, "_levels", levels)
        register_buffer(self, "_half_width", levels.div(2, rounding_mode="floor").float())
        register_buffer(self, "_half_levels", (levels.float() - 1) * (1 - config.eps) / 2)
        offsets = torch.tensor([0.5 if level % 2 == 0 else 0.0 for level in config.levels])
        register_buffer(self, "_offsets", offsets)
        register_buffer(self, "_shifts", (self._offsets / self._half_levels).tan())
        register_buffer(
            self,
            "_level_mask",
            self._build_level_mask(config.levels),
            persistent=False,
        )

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,)

    def forward(self, latents: Tensor) -> QuantizeOutput:
        return self.quantize(latents)

    def quantize(self, latents: Tensor) -> QuantizeOutput:
        _checks.input_latents(latents, self.input_dim)
        projected_latents = self._project_input(latents)
        codebook_vectors = self._quantize_projected_latents(projected_latents)
        indices = self.codebook_vectors_to_indices(codebook_vectors)
        quantized_latents = self.project_codebook_vectors(codebook_vectors)
        return QuantizeOutput(
            quantized_latents=quantized_latents,
            indices=indices,
            codebook_vectors=codebook_vectors,
            latents=projected_latents,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        _checks.input_latents(latents, self.input_dim)
        return self._quantize_projected_latents(self._project_input(latents))

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(
            codebook_vectors,
            self.codebook_dim,
            require_non_empty=False,
        )
        levels = self.codebook_vectors_to_levels(codebook_vectors)
        return self.levels_to_indices(levels)

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        levels = self.indices_to_levels(indices)
        return self.levels_to_codebook_vectors(levels)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(
            codebook_vectors,
            self.codebook_dim,
            require_non_empty=False,
        )
        return self.project_out(codebook_vectors)

    def round_to_codebook_vectors(self, continuous_codebook_vectors: Tensor) -> Tensor:
        levels = self.codebook_vectors_to_levels(continuous_codebook_vectors)
        return self.levels_to_codebook_vectors(levels)

    def codebook_vectors_to_levels(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(
            codebook_vectors,
            self.codebook_dim,
            require_non_empty=False,
        )
        levels = codebook_vectors * self._half_width + self._half_width
        levels = levels.round().long()
        return torch.minimum(levels.clamp_min(0), self._levels - 1)

    def levels_to_codebook_vectors(self, levels: Tensor) -> Tensor:
        self._validate_levels(levels)
        return (levels.to(dtype=self._half_width.dtype) - self._half_width) / self._half_width

    def indices_to_levels(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size)
        expanded = indices.unsqueeze(-1)
        return (expanded // self._basis) % self._levels

    def levels_to_indices(self, levels: Tensor) -> Tensor:
        self._validate_levels(levels)
        return (levels * self._basis).sum(dim=-1)

    def indices_to_level_probs(self, indices: Tensor, tau: float = 1.0) -> Tensor:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}.")
        level_indices = self.indices_to_levels(indices)
        max_level = max(self.levels)
        level_values = torch.arange(max_level, device=level_indices.device)
        dist = (level_indices.unsqueeze(-1) - level_values) ** 2
        logits = -dist.to(dtype=torch.float32) / tau
        mask = self._level_mask.to(device=indices.device)
        logits = logits.masked_fill(~mask.view((1,) * (logits.ndim - 2) + mask.shape), -torch.inf)
        return logits.softmax(dim=-1)

    def level_logits_mask(self, logits: Tensor) -> Tensor:
        mask = self._level_mask.to(device=logits.device)
        codebook_dim, max_level = mask.shape

        if logits.shape[-2:] == (codebook_dim, max_level):
            return mask.view((1,) * (logits.ndim - 2) + mask.shape)

        flattened_size = codebook_dim * max_level
        if logits.shape[-1] == flattened_size:
            flat = mask.reshape(-1)
            return flat.view((1,) * (logits.ndim - 1) + (flattened_size,))

        raise ValueError(
            f"Unsupported logits shape {tuple(logits.shape)} for FSQ level mask."
        )

    def _project_input(self, latents: Tensor) -> Tensor:
        return self.project_in(latents)

    def _quantize_projected_latents(self, projected_latents: Tensor) -> Tensor:
        _checks.codebook_vectors(
            projected_latents,
            self.codebook_dim,
            name="projected_latents",
            require_non_empty=False,
        )
        scaled_latents = projected_latents / self.config.bound_scale
        bounded = (scaled_latents + self._shifts).tanh() * self._half_levels - self._offsets
        rounded = bounded + (bounded.round() - bounded).detach()
        return rounded / self._half_width

    def _validate_levels(self, levels: Tensor) -> None:
        if levels.ndim == 0:
            raise ValueError(f"levels must end with codebook_dim={self.codebook_dim}.")
        if levels.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"levels must end with codebook_dim={self.codebook_dim}, "
                f"got {tuple(levels.shape)}."
            )
        if torch.is_floating_point(levels) or torch.is_complex(levels):
            raise TypeError("levels must be an integer tensor.")
        below = levels < 0
        above = levels >= self._levels
        if bool((below | above).any().item()):
            raise ValueError("levels contain values outside the configured FSQ level ranges.")

    @staticmethod
    def _build_level_mask(levels: tuple[int, ...]) -> Tensor:
        max_level = max(levels)
        mask = torch.zeros((len(levels), max_level), dtype=torch.bool)
        for index, level in enumerate(levels):
            mask[index, :level] = True
        return mask


def default_fsq_levels(power: Literal[8, 9, 10, 12, 14, 16]) -> tuple[int, ...]:
    if power == 8:
        return (7, 7, 5)
    if power == 9:
        return (5, 5, 5, 5)
    if power == 10:
        return (7, 7, 7, 3)
    if power == 12:
        return (7, 5, 5, 5, 5)
    if power == 14:
        return (11, 11, 9, 5, 3)
    if power == 16:
        return DEFAULT_FSQ_LEVELS
    raise ValueError(f"power={power} is not supported.")
