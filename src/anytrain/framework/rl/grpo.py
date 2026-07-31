from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from anytrain.loss.abc import LossDetails

from ._detail import detach_details
from .advantage import group_relative_advantages
from .types import TokenReduction


class GRPOLoss(nn.Module):
    """Group Relative Policy Optimization over offline rollout groups."""

    def __init__(
        self,
        *,
        clip_range: float = 0.2,
        kl_beta: float = 0.0,
        eps: float = 1e-8,
        token_reduction: TokenReduction = "sequence_mean",
    ) -> None:
        super().__init__()
        if clip_range < 0:
            raise ValueError("clip_range must be non-negative.")
        if kl_beta < 0:
            raise ValueError("kl_beta must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if token_reduction not in {"token_mean", "sequence_mean"}:
            raise ValueError("token_reduction must be 'token_mean' or 'sequence_mean'.")
        self.clip_range = float(clip_range)
        self.kl_beta = float(kl_beta)
        self.eps = float(eps)
        self.token_reduction = token_reduction

    def forward(
        self,
        *,
        policy_token_logps: Tensor,
        old_token_logps: Tensor,
        rewards: Tensor,
        response_mask: Tensor,
        ref_token_logps: Tensor | None = None,
        group_mask: Tensor | None = None,
    ) -> tuple[Tensor, LossDetails]:
        _validate_rollout_tensors(policy_token_logps, old_token_logps, response_mask)
        if rewards.shape != policy_token_logps.shape[:2]:
            raise ValueError("rewards must have shape [batch, group].")
        old_token_logps = old_token_logps.detach()
        rewards = rewards.detach()
        candidate_mask = _candidate_mask(response_mask, rewards, group_mask)
        advantages = group_relative_advantages(rewards, candidate_mask, eps=self.eps).detach()

        log_ratio = policy_token_logps - old_token_logps
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        token_advantages = advantages.unsqueeze(-1)
        policy_loss = -torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
        pg_loss = _reduce_token_values(policy_loss, response_mask, candidate_mask, self.token_reduction)

        if ref_token_logps is None:
            if self.kl_beta > 0:
                raise ValueError("ref_token_logps is required when kl_beta is positive.")
            kl = policy_token_logps.new_zeros(())
        else:
            _validate_reference(policy_token_logps, ref_token_logps, response_mask)
            ref_token_logps = ref_token_logps.detach()
            kl_values = _approx_kl(policy_token_logps, ref_token_logps)
            kl = _reduce_token_values(kl_values, response_mask, candidate_mask, self.token_reduction)

        total = pg_loss + policy_token_logps.new_tensor(self.kl_beta) * kl
        selected_rewards = rewards.masked_select(candidate_mask)
        selected_advantages = advantages.masked_select(candidate_mask)
        details = {
            "pg_loss": pg_loss,
            "kl_loss": kl,
            "reward_mean": selected_rewards.mean(),
            "reward_std": selected_rewards.std(unbiased=False),
            "advantage_mean": selected_advantages.mean(),
            "clip_fraction": (
                (torch.abs(ratio - 1.0) > self.clip_range)
                .masked_select(response_mask)
                .to(dtype=total.dtype)
                .mean()
            ),
            "token_count": response_mask.to(dtype=total.dtype).sum(),
            "candidate_count": candidate_mask.to(dtype=total.dtype).sum(),
        }
        return total, detach_details(details)


def _validate_rollout_tensors(policy: Tensor, old: Tensor, response_mask: Tensor) -> None:
    if not isinstance(policy, Tensor) or not isinstance(old, Tensor):
        raise TypeError("policy_token_logps and old_token_logps must be tensors.")
    if policy.shape != old.shape:
        raise ValueError("policy_token_logps and old_token_logps must have the same shape.")
    if policy.ndim != 3:
        raise ValueError("token log-probs must have shape [batch, group, tokens].")
    if policy.is_complex() or old.is_complex():
        raise TypeError("token log-probs must be real-valued.")
    if not isinstance(response_mask, Tensor):
        raise TypeError("response_mask must be a tensor.")
    if response_mask.dtype != torch.bool:
        raise TypeError("response_mask must be boolean.")
    if response_mask.shape != policy.shape:
        raise ValueError("response_mask must have the same shape as token log-probs.")
    if response_mask.device != policy.device:
        raise ValueError("response_mask must be on the same device as token log-probs.")
    if old.device != policy.device:
        raise ValueError("old_token_logps must be on the same device as policy_token_logps.")
    if not bool(torch.isfinite(policy.masked_select(response_mask)).all()):
        raise ValueError("selected policy_token_logps must be finite.")
    if not bool(torch.isfinite(old.masked_select(response_mask)).all()):
        raise ValueError("selected old_token_logps must be finite.")


def _candidate_mask(
    response_mask: Tensor,
    rewards: Tensor,
    group_mask: Tensor | None,
) -> Tensor:
    if rewards.device != response_mask.device:
        raise ValueError("rewards must be on the same device as response_mask.")
    if rewards.is_complex() or rewards.dtype == torch.bool:
        raise TypeError("rewards must be real-valued.")
    derived = response_mask.any(dim=-1)
    if group_mask is None:
        return derived
    if not isinstance(group_mask, Tensor):
        raise TypeError("group_mask must be a tensor.")
    if group_mask.dtype != torch.bool:
        raise TypeError("group_mask must be boolean.")
    if group_mask.shape != rewards.shape:
        raise ValueError("group_mask must have the same shape as rewards.")
    if group_mask.device != rewards.device:
        raise ValueError("group_mask must be on the same device as rewards.")
    if bool((group_mask & ~derived).any()):
        raise ValueError("group_mask cannot select candidates with no response tokens.")
    return group_mask


def _validate_reference(policy: Tensor, reference: Tensor, response_mask: Tensor) -> None:
    if not isinstance(reference, Tensor):
        raise TypeError("ref_token_logps must be a tensor.")
    if reference.shape != policy.shape:
        raise ValueError("ref_token_logps must have the same shape as policy_token_logps.")
    if reference.device != policy.device:
        raise ValueError("ref_token_logps must be on the same device as policy_token_logps.")
    if reference.is_complex():
        raise TypeError("ref_token_logps must be real-valued.")
    if not bool(torch.isfinite(reference.masked_select(response_mask)).all()):
        raise ValueError("selected ref_token_logps must be finite.")


def _approx_kl(policy: Tensor, reference: Tensor) -> Tensor:
    log_ratio = reference - policy
    return torch.exp(log_ratio) - log_ratio - 1.0


def _reduce_token_values(
    value: Tensor,
    response_mask: Tensor,
    candidate_mask: Tensor,
    reduction: TokenReduction,
) -> Tensor:
    resolved = cast(TokenReduction, reduction)
    weights = response_mask.to(device=value.device, dtype=value.dtype)
    if resolved == "token_mean":
        token_count = weights.sum()
        if bool(token_count <= 0):
            raise ValueError("response_mask must select at least one token.")
        return (value * weights).sum() / token_count
    if resolved == "sequence_mean":
        token_counts = weights.sum(dim=-1)
        if bool((token_counts.masked_select(candidate_mask) <= 0).any()):
            raise ValueError("selected candidates must contain at least one token.")
        sequence_loss = (value * weights).sum(dim=-1) / token_counts.clamp_min(1)
        selected = sequence_loss.masked_select(candidate_mask)
        if not bool(selected.numel()):
            raise ValueError("group_mask must select at least one candidate.")
        return selected.mean()
    raise ValueError("token_reduction must be 'token_mean' or 'sequence_mean'.")


__all__ = ["GRPOLoss"]
