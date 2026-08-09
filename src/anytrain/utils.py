from __future__ import annotations

import torch
from torch import Tensor

_SIGNED_INTEGER_DTYPES: frozenset[torch.dtype] = frozenset(
    (torch.int8, torch.int16, torch.int32, torch.int64)
)


def is_signed_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _SIGNED_INTEGER_DTYPES


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = sorted_logits.softmax(dim=-1)
    remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
    remove[..., 0] = False
    filtered = logits.new_full(logits.shape, float("-inf"))
    filtered.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(remove, float("-inf")),
    )
    return filtered


__all__ = ["is_signed_integer_dtype", "top_p_filter"]
