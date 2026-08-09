from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

_MAX_LOOKUP_ELEMENTS = 16 * 1024 * 1024


@torch.no_grad()
def nearest_codebook_indices(
    latents: Tensor,
    codebook: Tensor,
    *,
    normalize: bool,
    max_lookup_elements: int = _MAX_LOOKUP_ELEMENTS,
    lookup_in_float32: bool = True,
) -> Tensor:
    if latents.ndim == 0:
        raise ValueError("latents must have at least one dimension.")
    if codebook.ndim != 2:
        raise ValueError(f"codebook must have shape (size, dim), got {tuple(codebook.shape)}.")
    if codebook.shape[0] == 0:
        raise ValueError("codebook must contain at least one vector.")
    if latents.shape[-1] != codebook.shape[-1]:
        raise ValueError(
            f"latents must end with codebook dim={codebook.shape[-1]}, got {tuple(latents.shape)}."
        )
    if max_lookup_elements <= 0:
        raise ValueError("max_lookup_elements must be positive.")

    leading_shape = latents.shape[:-1]
    flat_latents = latents.reshape(-1, codebook.shape[-1])
    if flat_latents.shape[0] == 0:
        return torch.empty(leading_shape, dtype=torch.long, device=latents.device)

    if lookup_in_float32:
        flat_latents = flat_latents.float()
        codebook = codebook.float()

    codebook_chunk_size = min(codebook.shape[0], max_lookup_elements)
    latent_chunk_size = max(1, max_lookup_elements // codebook_chunk_size)
    codebook_squared_norm: Tensor | None = None
    if normalize:
        codebook_lookup = F.normalize(codebook, dim=-1)
    else:
        codebook_lookup = codebook
        codebook_squared_norm = codebook.square().sum(dim=-1)

    indices = []
    for latent_chunk in flat_latents.split(latent_chunk_size):
        lookup_chunk = F.normalize(latent_chunk, dim=-1) if normalize else latent_chunk
        latent_squared_norm = None if normalize else latent_chunk.square().sum(dim=-1, keepdim=True)
        best_values: Tensor | None = None
        best_indices: Tensor | None = None
        for start in range(0, codebook.shape[0], codebook_chunk_size):
            end = min(start + codebook_chunk_size, codebook.shape[0])
            if normalize:
                comparison = F.linear(lookup_chunk, codebook_lookup[start:end])
                chunk_values, chunk_indices = comparison.max(dim=-1)
            else:
                assert latent_squared_norm is not None
                assert codebook_squared_norm is not None
                distances = (
                    latent_squared_norm
                    - 2 * lookup_chunk @ codebook_lookup[start:end].t()
                    + codebook_squared_norm[start:end]
                )
                chunk_values, chunk_indices = distances.min(dim=-1)
            chunk_indices = chunk_indices + start

            if best_values is None or best_indices is None:
                best_values = chunk_values
                best_indices = chunk_indices
                continue
            better = chunk_values > best_values if normalize else chunk_values < best_values
            better |= torch.isnan(best_values) & ~torch.isnan(chunk_values)
            best_values = torch.where(better, chunk_values, best_values)
            best_indices = torch.where(better, chunk_indices, best_indices)

        if best_indices is None:
            raise RuntimeError("codebook lookup produced no indices.")
        indices.append(best_indices)
    return torch.cat(indices).reshape(*leading_shape)
