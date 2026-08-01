from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul

import torch
from torch import Tensor


def product_codebook_size(group_sizes: Sequence[int]) -> int:
    return reduce(mul, group_sizes, 1)


def uniform_group_sizes(group_size: int, num_groups: int) -> tuple[int, ...]:
    return (group_size,) * num_groups


def mixed_radix_basis(group_sizes: Sequence[int]) -> Tensor:
    return torch.cumprod(torch.tensor((1, *tuple(group_sizes)[:-1]), dtype=torch.long), dim=0)


def indices_to_group_indices(
    indices: Tensor,
    *,
    index_basis: Tensor,
    group_sizes: Tensor | int,
) -> Tensor:
    basis = index_basis.to(device=indices.device)
    if isinstance(group_sizes, Tensor):
        sizes: Tensor | int = group_sizes.to(device=indices.device, dtype=indices.dtype)
    else:
        sizes = group_sizes
    return (indices.to(dtype=torch.long).unsqueeze(-1) // basis) % sizes


def group_indices_to_indices(
    group_indices: Tensor,
    *,
    index_basis: Tensor,
) -> Tensor:
    basis = index_basis.to(device=group_indices.device)
    return (group_indices.to(dtype=torch.long) * basis).sum(dim=-1)
