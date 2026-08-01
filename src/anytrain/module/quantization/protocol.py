from __future__ import annotations

from typing import Protocol

from torch import Tensor

from .output import QuantizeOutput


class QuantizerProtocol(Protocol):
    num_codebooks: int
    codebook_size: int
    codebook_dim: int
    input_dim: int

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...
    def forward(self, latents: Tensor) -> QuantizeOutput: ...
    def quantize(self, latents: Tensor) -> QuantizeOutput: ...
    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor: ...
    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor: ...
    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor: ...
    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor: ...


class ResidualQuantizerProtocol(QuantizerProtocol, Protocol):
    def forward(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> QuantizeOutput: ...
    def quantize(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> QuantizeOutput: ...
    def latents_to_codebook_vectors(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> Tensor: ...
