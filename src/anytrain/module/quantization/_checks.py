from __future__ import annotations

import torch
from torch import Tensor


def input_latents(
    latents: Tensor,
    input_dim: int,
    *,
    require_non_empty: bool = True,
) -> None:
    if not isinstance(latents, Tensor):
        raise TypeError("latents must be a tensor.")
    if latents.ndim == 0:
        raise ValueError("latents must have at least one dimension.")
    if latents.shape[-1] != input_dim:
        raise ValueError(
            f"expected latents last dimension to be input_dim={input_dim}, "
            f"got {latents.shape[-1]}."
        )
    if require_non_empty and latents.numel() == 0:
        raise ValueError("latents must contain at least one vector.")


def codebook_vectors(
    values: Tensor,
    codebook_dim: int,
    *,
    name: str = "codebook_vectors",
    require_non_empty: bool = True,
) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if values.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if values.shape[-1] != codebook_dim:
        raise ValueError(
            f"{name} must end with codebook_dim={codebook_dim}, "
            f"got {tuple(values.shape)}."
        )
    if require_non_empty and values.numel() == 0:
        raise ValueError(f"{name} must contain at least one vector.")


def active_codebook_vectors(
    values: Tensor,
    *,
    codebook_dim: int,
    num_codebooks: int,
) -> None:
    if not isinstance(values, Tensor):
        raise TypeError("codebook_vectors must be a tensor.")
    if values.ndim < 2:
        raise ValueError(f"codebook_vectors must have shape (..., n, {codebook_dim}).")
    if values.shape[-1] != codebook_dim:
        raise ValueError(
            f"codebook_vectors must end with codebook_dim={codebook_dim}, "
            f"got {tuple(values.shape)}."
        )
    active_count = values.shape[-2]
    if active_count <= 0 or active_count > num_codebooks:
        raise ValueError(
            f"codebook_vectors active dimension must be in [1, {num_codebooks}], "
            f"got {active_count}."
        )


def indices(
    values: Tensor,
    codebook_size: int,
    *,
    allow_inactive: bool = False,
    require_ndim: bool = False,
) -> None:
    if not isinstance(values, Tensor):
        raise TypeError("indices must be a tensor.")
    if require_ndim and values.ndim == 0:
        raise ValueError("indices must have at least one dimension.")
    _validate_index_bounds(values, codebook_size, allow_inactive=allow_inactive)


def index_dtype(values: Tensor) -> None:
    if values.dtype == torch.bool:
        raise TypeError("indices must be an integer tensor; bool tensors are not valid indices.")
    if torch.is_floating_point(values) or torch.is_complex(values):
        raise TypeError("indices must be an integer tensor.")


def active_indices(
    values: Tensor,
    *,
    codebook_size: int,
    num_codebooks: int,
    allow_inactive: bool = False,
) -> None:
    _validate_active_indices(
        values,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        allow_inactive=allow_inactive,
    )


def active_indices_has_inactive(
    values: Tensor,
    *,
    codebook_size: int,
    num_codebooks: int,
    allow_inactive: bool = False,
) -> bool:
    """Validate active indices and report whether any entry is ``-1``."""

    return _validate_active_indices(
        values,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        allow_inactive=allow_inactive,
    )


def _validate_active_indices(
    values: Tensor,
    *,
    codebook_size: int,
    num_codebooks: int,
    allow_inactive: bool,
) -> bool:
    if not isinstance(values, Tensor):
        raise TypeError("indices must be a tensor.")
    if values.ndim == 0:
        raise ValueError("indices must have at least one dimension.")
    active_count = values.shape[-1]
    if active_count <= 0 or active_count > num_codebooks:
        raise ValueError(
            f"indices active dimension must be in [1, {num_codebooks}], "
            f"got {active_count}."
    )
    min_index, _ = _validate_index_bounds(
        values,
        codebook_size,
        allow_inactive=allow_inactive,
    )
    return allow_inactive and min_index < 0


def _validate_index_bounds(
    values: Tensor,
    codebook_size: int,
    *,
    allow_inactive: bool,
) -> tuple[int, int]:
    index_dtype(values)
    if values.numel() == 0:
        raise ValueError("indices must contain at least one value.")
    min_allowed = -1 if allow_inactive else 0
    min_index = int(values.min().item())
    max_index = int(values.max().item())
    if min_index < min_allowed or max_index >= codebook_size:
        if allow_inactive:
            raise ValueError(
                f"indices must be in [{min_allowed}, codebook_size - 1]: "
                f"got min={min_index}, max={max_index}, codebook_size={codebook_size}."
            )
        raise ValueError(
            "indices must be in [0, codebook_size - 1]: "
            f"got min={min_index}, max={max_index}, codebook_size={codebook_size}."
        )
    return min_index, max_index


def dropout(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise TypeError("dropout must be a real number or None.")
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"dropout must be in [0, 1], got {value}.")
