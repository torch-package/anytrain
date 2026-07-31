from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import SequenceReduction


def gather_token_logps(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Gather log-probabilities for selected token ids."""

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a tensor.")
    if not isinstance(token_ids, Tensor):
        raise TypeError("token_ids must be a tensor.")
    if logits.ndim < 1:
        raise ValueError("logits must include a vocabulary dimension.")
    if token_ids.shape != logits.shape[:-1]:
        raise ValueError("token_ids shape must match logits without the vocabulary dimension.")
    if token_ids.dtype == torch.bool or token_ids.dtype.is_floating_point:
        raise TypeError("token_ids must contain integer token ids.")
    vocab = logits.shape[-1]
    if bool((token_ids < 0).any()) or bool((token_ids >= vocab).any()):
        raise ValueError("token_ids must be within the logits vocabulary.")
    logps = F.log_softmax(logits, dim=-1)
    return logps.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def sequence_logps(
    token_logps: Tensor,
    mask: Tensor,
    *,
    reduction: SequenceReduction = "sum",
) -> Tensor:
    """Reduce token log-probs to one value per sequence using a boolean mask."""

    _validate_aligned_mask(token_logps, mask)
    weights = mask.to(device=token_logps.device, dtype=token_logps.dtype)
    total = (token_logps * weights).sum(dim=-1)
    resolved = cast(SequenceReduction, reduction)
    if resolved == "sum":
        return total
    if resolved == "mean":
        counts = weights.sum(dim=-1)
        if bool((counts <= 0).any()):
            raise ValueError("mask must select at least one token for every sequence.")
        return total / counts
    raise ValueError("reduction must be 'sum' or 'mean'.")


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Return the mean over mask-selected entries."""

    _validate_aligned_mask(value, mask)
    weights = mask.to(device=value.device, dtype=value.dtype)
    count = weights.sum()
    if bool(count <= 0):
        raise ValueError("mask must select at least one value.")
    return (value * weights).sum() / count


def _validate_aligned_mask(value: Tensor, mask: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("value must be a tensor.")
    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a tensor.")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean.")
    if mask.shape != value.shape:
        raise ValueError("mask shape must match value shape.")
    if mask.device != value.device:
        raise ValueError("mask must be on the same device as value.")


__all__ = ["gather_token_logps", "masked_mean", "sequence_logps"]
