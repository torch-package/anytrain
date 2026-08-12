from __future__ import annotations

import torch
from torch import Tensor

from .._validation import strict_bool


def group_relative_advantages(
    rewards: Tensor,
    group_mask: Tensor | None = None,
    *,
    eps: float = 1e-8,
    validate: bool = True,
) -> Tensor:
    """Normalize rewards within each prompt group.

    Set ``validate=False`` only when the caller owns the reward and mask
    contracts for a trusted training batch.
    """

    validate = strict_bool(validate, "validate")
    if validate:
        if not isinstance(rewards, Tensor):
            raise TypeError("rewards must be a tensor.")
        if rewards.ndim != 2:
            raise ValueError("rewards must have shape [batch, group].")
        if rewards.is_complex() or rewards.dtype == torch.bool:
            raise TypeError("rewards must be real-valued.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
    mask = _candidate_mask(rewards, group_mask, validate=validate)
    if validate and not bool(torch.isfinite(rewards.masked_select(mask)).all()):
        raise ValueError("selected rewards must be finite.")

    counts = mask.to(dtype=rewards.dtype).sum(dim=1)
    if validate and bool((counts < 2).any()):
        raise ValueError("each reward group must contain at least two valid candidates.")

    safe_rewards = rewards.masked_fill(~mask, 0)
    mean = safe_rewards.sum(dim=1, keepdim=True) / counts.unsqueeze(1)
    centered = (rewards - mean).masked_fill(~mask, 0)
    variance = centered.square().sum(dim=1, keepdim=True) / counts.unsqueeze(1)
    std = variance.sqrt()
    advantages = centered / (std + rewards.new_tensor(float(eps)))
    return advantages.masked_fill(~mask, 0)


def _candidate_mask(rewards: Tensor, group_mask: Tensor | None, *, validate: bool) -> Tensor:
    if group_mask is None:
        return torch.ones_like(rewards, dtype=torch.bool)
    if validate:
        if not isinstance(group_mask, Tensor):
            raise TypeError("group_mask must be a tensor.")
        if group_mask.dtype != torch.bool:
            raise TypeError("group_mask must be boolean.")
        if group_mask.shape != rewards.shape:
            raise ValueError("group_mask must have the same shape as rewards.")
        if group_mask.device != rewards.device:
            raise ValueError("group_mask must be on the same device as rewards.")
    return group_mask


__all__ = ["group_relative_advantages"]
