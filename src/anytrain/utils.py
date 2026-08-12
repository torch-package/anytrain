from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor

_SIGNED_INTEGER_DTYPES: frozenset[torch.dtype] = frozenset(
    (torch.int8, torch.int16, torch.int32, torch.int64)
)


def is_signed_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _SIGNED_INTEGER_DTYPES


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a tensor.")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must contain a non-empty vocabulary dimension.")
    if logits.is_complex() or not logits.is_floating_point():
        raise TypeError("logits must be a real floating-point tensor.")
    if isinstance(top_p, bool) or not isinstance(top_p, Real):
        raise TypeError("top_p must be a host real number, not bool or Tensor.")
    top_p = float(top_p)
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1].")

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
