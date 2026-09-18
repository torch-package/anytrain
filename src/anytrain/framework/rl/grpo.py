from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .._validation import non_negative_real, positive_real, strict_bool
from ._observation import emit, recommend, register
from .advantage import _reward_values, group_relative_advantages
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
        self.clip_range = non_negative_real(clip_range, "clip_range")
        self.kl_beta = non_negative_real(kl_beta, "kl_beta")
        self.eps = positive_real(eps, "eps")
        if token_reduction not in {"token_mean", "sequence_mean"}:
            raise ValueError("token_reduction must be 'token_mean' or 'sequence_mean'.")
        self.token_reduction: TokenReduction = token_reduction
        recommend(self)

    def forward(
        self,
        *,
        policy_token_logps: Tensor,
        old_token_logps: Tensor,
        rewards: Tensor,
        response_mask: Tensor,
        ref_token_logps: Tensor | None = None,
        group_mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        """Compute GRPO loss.

        Set ``validate=False`` only for batches whose tensor contracts have
        already been checked by their owner. This avoids host synchronization
        from per-step finite and mask validation.
        """

        validate = strict_bool(validate, "validate")
        if validate:
            _validate_rollout_tensors(policy_token_logps, old_token_logps, response_mask)
            if rewards.shape != policy_token_logps.shape[:2]:
                raise ValueError("rewards must have shape [batch, group].")
        policy_token_logps = policy_token_logps.to(
            dtype=torch.promote_types(policy_token_logps.dtype, torch.float32)
        )
        old_token_logps = old_token_logps.detach().to(
            dtype=torch.promote_types(old_token_logps.dtype, torch.float32)
        )
        rewards = _reward_values(rewards.detach())
        candidate_mask = _candidate_mask(response_mask, rewards, group_mask, validate=validate)
        selected_token_mask = response_mask & candidate_mask.unsqueeze(-1)
        if validate:
            _validate_selected_logps(
                policy_token_logps,
                old_token_logps,
                selected_token_mask,
            )
        policy_token_logps = policy_token_logps.masked_fill(~selected_token_mask, 0)
        old_token_logps = old_token_logps.masked_fill(~selected_token_mask, 0)
        advantages = group_relative_advantages(
            rewards,
            candidate_mask,
            eps=self.eps,
            validate=validate,
        ).detach()

        log_ratio = policy_token_logps - old_token_logps
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        token_advantages = advantages.unsqueeze(-1)
        policy_loss = -torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
        pg_loss = _reduce_token_values(
            policy_loss,
            selected_token_mask,
            candidate_mask,
            self.token_reduction,
            validate=validate,
        )

        if ref_token_logps is None:
            if self.kl_beta > 0:
                raise ValueError("ref_token_logps is required when kl_beta is positive.")
            kl = policy_token_logps.new_zeros(())
        else:
            if validate:
                _validate_reference(policy_token_logps, ref_token_logps, selected_token_mask)
            ref_token_logps = ref_token_logps.detach().masked_fill(~selected_token_mask, 0)
            kl_values = _approx_kl(policy_token_logps, ref_token_logps)
            kl = _reduce_token_values(
                kl_values,
                selected_token_mask,
                candidate_mask,
                self.token_reduction,
                validate=validate,
            )

        total = pg_loss + policy_token_logps.new_tensor(self.kl_beta) * kl
        selected_rewards = rewards.masked_select(candidate_mask)
        selected_advantages = advantages.masked_select(candidate_mask)
        emit(
            self,
            {
                "pg_loss": pg_loss,
                "kl_loss": kl,
                "reward_mean": selected_rewards.mean(),
                "reward_std": selected_rewards.std(unbiased=False),
                "advantage_mean": selected_advantages.mean(),
                "clip_fraction": (
                    (torch.abs(ratio - 1.0) > self.clip_range)
                    .masked_select(selected_token_mask)
                    .to(dtype=total.dtype)
                    .mean()
                ),
                "token_count": selected_token_mask.to(dtype=total.dtype).sum(),
                "candidate_count": candidate_mask.to(dtype=total.dtype).sum(),
            },
        )
        return total


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


def _validate_selected_logps(policy: Tensor, old: Tensor, selected_token_mask: Tensor) -> None:
    if not bool(torch.isfinite(policy.masked_select(selected_token_mask)).all()):
        raise ValueError("selected policy_token_logps must be finite.")
    if not bool(torch.isfinite(old.masked_select(selected_token_mask)).all()):
        raise ValueError("selected old_token_logps must be finite.")


def _candidate_mask(
    response_mask: Tensor,
    rewards: Tensor,
    group_mask: Tensor | None,
    *,
    validate: bool,
) -> Tensor:
    if validate:
        if rewards.device != response_mask.device:
            raise ValueError("rewards must be on the same device as response_mask.")
        if rewards.is_complex() or rewards.dtype == torch.bool:
            raise TypeError("rewards must be real-valued.")
    derived = response_mask.any(dim=-1)
    if group_mask is None:
        return derived
    if validate:
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


def _validate_reference(policy: Tensor, reference: Tensor, selected_token_mask: Tensor) -> None:
    if not isinstance(reference, Tensor):
        raise TypeError("ref_token_logps must be a tensor.")
    if reference.shape != policy.shape:
        raise ValueError("ref_token_logps must have the same shape as policy_token_logps.")
    if reference.device != policy.device:
        raise ValueError("ref_token_logps must be on the same device as policy_token_logps.")
    if reference.is_complex():
        raise TypeError("ref_token_logps must be real-valued.")
    if not bool(torch.isfinite(reference.masked_select(selected_token_mask)).all()):
        raise ValueError("selected ref_token_logps must be finite.")


def _approx_kl(policy: Tensor, reference: Tensor) -> Tensor:
    log_ratio = reference - policy
    return torch.exp(log_ratio) - log_ratio - 1.0


def _reduce_token_values(
    value: Tensor,
    selected_token_mask: Tensor,
    candidate_mask: Tensor,
    reduction: TokenReduction,
    *,
    validate: bool,
) -> Tensor:
    resolved = cast(TokenReduction, reduction)
    value = value.to(dtype=torch.promote_types(value.dtype, torch.float32))
    value = value.masked_fill(~selected_token_mask, 0)
    if resolved == "token_mean":
        token_count = selected_token_mask.sum()
        if validate and bool(token_count <= 0):
            raise ValueError("response_mask must select at least one token.")
        return value.sum() / token_count
    if resolved == "sequence_mean":
        token_counts = selected_token_mask.sum(dim=-1)
        if validate and bool((token_counts.masked_select(candidate_mask) <= 0).any()):
            raise ValueError("selected candidates must contain at least one token.")
        sequence_loss = value.sum(dim=-1) / token_counts.clamp_min(1)
        candidate_weights = candidate_mask.to(dtype=torch.int64)
        if validate and bool(candidate_weights.sum() <= 0):
            raise ValueError("group_mask must select at least one candidate.")
        return (sequence_loss * candidate_weights).sum() / candidate_weights.sum().clamp_min(1)
    raise ValueError("token_reduction must be 'token_mean' or 'sequence_mean'.")


register(GRPOLoss)


__all__ = ["GRPOLoss"]
