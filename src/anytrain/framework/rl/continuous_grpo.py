from __future__ import annotations

import torch
from torch import Tensor, nn

from .._validation import non_negative_real, positive_real, strict_bool
from ._observation import emit, recommend, register
from .advantage import group_relative_advantages
from .types import StepReduction


class ContinuousGRPOLoss(nn.Module):
    """GRPO loss for Flow-GRPO style continuous transition log-probabilities.

    Use this when the downstream project has already collected grouped flow or
    diffusion rollouts and can provide per-step Gaussian log-probs, rewards, and
    masks. It does not run sampling, scoring, or model forwarding.

    Reference: Liu et al., "Flow-GRPO: Training Flow Matching Models via Online
    RL", arXiv:2505.05470.
    """

    def __init__(
        self,
        *,
        clip_range: float = 1e-4,
        kl_beta: float = 0.0,
        eps: float = 1e-8,
        adv_clip_max: float | None = None,
        step_reduction: StepReduction = "trajectory_mean",
    ) -> None:
        super().__init__()
        self.clip_range = non_negative_real(clip_range, "clip_range")
        self.kl_beta = non_negative_real(kl_beta, "kl_beta")
        self.eps = positive_real(eps, "eps")
        self.adv_clip_max = (
            None if adv_clip_max is None else positive_real(adv_clip_max, "adv_clip_max")
        )
        if step_reduction not in {"step_mean", "trajectory_mean"}:
            raise ValueError("step_reduction must be 'step_mean' or 'trajectory_mean'.")

        self.step_reduction: StepReduction = step_reduction
        recommend(self)

    def forward(
        self,
        *,
        policy_step_logps: Tensor,
        old_step_logps: Tensor,
        rewards: Tensor,
        step_mask: Tensor,
        kl_values: Tensor | None = None,
        group_mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        """Compute continuous GRPO loss.

        Set ``validate=False`` only for batches whose tensor contracts have
        already been checked by their owner.
        """

        validate = strict_bool(validate, "validate")
        if validate:
            _validate_step_tensors(policy_step_logps, old_step_logps, step_mask)
            if rewards.shape != policy_step_logps.shape[:2]:
                raise ValueError("rewards must have shape [batch, group].")

        old_step_logps = old_step_logps.detach()
        rewards = rewards.detach()
        candidate_mask = _candidate_mask(step_mask, rewards, group_mask, validate=validate)
        selected_step_mask = step_mask & candidate_mask.unsqueeze(-1)
        if validate:
            _validate_selected_logps(
                policy_step_logps,
                old_step_logps,
                selected_step_mask,
            )
        policy_step_logps = policy_step_logps.masked_fill(~selected_step_mask, 0)
        old_step_logps = old_step_logps.masked_fill(~selected_step_mask, 0)
        advantages = group_relative_advantages(
            rewards,
            candidate_mask,
            eps=self.eps,
            validate=validate,
        ).detach()
        if self.adv_clip_max is not None:
            advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)

        log_ratio = policy_step_logps - old_step_logps
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        step_advantages = advantages.unsqueeze(-1)
        policy_loss = -torch.minimum(ratio * step_advantages, clipped_ratio * step_advantages)
        pg_loss = _reduce_step_values(
            policy_loss,
            selected_step_mask,
            candidate_mask,
            self.step_reduction,
            validate=validate,
        )

        if kl_values is None:
            if self.kl_beta > 0:
                raise ValueError("kl_values is required when kl_beta is positive.")
            kl_loss = policy_step_logps.new_zeros(())
        else:
            if validate:
                _validate_kl_values(policy_step_logps, kl_values, selected_step_mask)
            kl_loss = _reduce_step_values(
                kl_values.detach().masked_fill(~selected_step_mask, 0),
                selected_step_mask,
                candidate_mask,
                self.step_reduction,
                validate=validate,
            )

        total = pg_loss + policy_step_logps.new_tensor(self.kl_beta) * kl_loss
        selected_rewards = rewards.masked_select(candidate_mask)
        selected_advantages = advantages.masked_select(candidate_mask)
        emit(
            self,
            {
                "pg_loss": pg_loss,
                "kl_loss": kl_loss,
                "reward_mean": selected_rewards.mean(),
                "reward_std": selected_rewards.std(unbiased=False),
                "advantage_mean": selected_advantages.mean(),
                "approx_kl": 0.5
                * _reduce_step_values(
                    log_ratio.square(),
                    selected_step_mask,
                    candidate_mask,
                    self.step_reduction,
                    validate=validate,
                ),
                "clip_fraction": (
                    (torch.abs(ratio - 1.0) > self.clip_range)
                    .masked_select(selected_step_mask)
                    .to(dtype=total.dtype)
                    .mean()
                ),
                "step_count": selected_step_mask.to(dtype=total.dtype).sum(),
                "candidate_count": candidate_mask.to(dtype=total.dtype).sum(),
            },
        )
        return total


def _validate_step_tensors(policy: Tensor, old: Tensor, step_mask: Tensor) -> None:
    if not isinstance(policy, Tensor) or not isinstance(old, Tensor):
        raise TypeError("policy_step_logps and old_step_logps must be tensors.")
    if policy.shape != old.shape:
        raise ValueError("policy_step_logps and old_step_logps must have the same shape.")
    if policy.ndim != 3:
        raise ValueError("step log-probs must have shape [batch, group, steps].")
    if policy.is_complex() or old.is_complex():
        raise TypeError("step log-probs must be real-valued.")
    if not isinstance(step_mask, Tensor):
        raise TypeError("step_mask must be a tensor.")
    if step_mask.dtype != torch.bool:
        raise TypeError("step_mask must be boolean.")
    if step_mask.shape != policy.shape:
        raise ValueError("step_mask must have the same shape as step log-probs.")
    if step_mask.device != policy.device:
        raise ValueError("step_mask must be on the same device as step log-probs.")
    if old.device != policy.device:
        raise ValueError("old_step_logps must be on the same device as policy_step_logps.")


def _validate_selected_logps(policy: Tensor, old: Tensor, selected_step_mask: Tensor) -> None:
    if not bool(torch.isfinite(policy.masked_select(selected_step_mask)).all()):
        raise ValueError("selected policy_step_logps must be finite.")
    if not bool(torch.isfinite(old.masked_select(selected_step_mask)).all()):
        raise ValueError("selected old_step_logps must be finite.")


def _candidate_mask(
    step_mask: Tensor,
    rewards: Tensor,
    group_mask: Tensor | None,
    *,
    validate: bool,
) -> Tensor:
    if validate:
        if rewards.device != step_mask.device:
            raise ValueError("rewards must be on the same device as step_mask.")
        if rewards.is_complex() or rewards.dtype == torch.bool:
            raise TypeError("rewards must be real-valued.")
    derived = step_mask.any(dim=-1)
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
            raise ValueError("group_mask cannot select candidates with no rollout steps.")
    return group_mask


def _validate_kl_values(policy: Tensor, kl_values: Tensor, selected_step_mask: Tensor) -> None:
    if not isinstance(kl_values, Tensor):
        raise TypeError("kl_values must be a tensor.")
    if kl_values.shape != policy.shape:
        raise ValueError("kl_values must have the same shape as policy_step_logps.")
    if kl_values.device != policy.device:
        raise ValueError("kl_values must be on the same device as policy_step_logps.")
    if kl_values.is_complex():
        raise TypeError("kl_values must be real-valued.")
    if bool((kl_values.masked_select(selected_step_mask) < 0).any()):
        raise ValueError("selected kl_values must be non-negative.")
    if not bool(torch.isfinite(kl_values.masked_select(selected_step_mask)).all()):
        raise ValueError("selected kl_values must be finite.")


def _reduce_step_values(
    value: Tensor,
    selected_step_mask: Tensor,
    candidate_mask: Tensor,
    reduction: StepReduction,
    *,
    validate: bool,
) -> Tensor:
    weights = selected_step_mask.to(device=value.device, dtype=value.dtype)
    if reduction == "step_mean":
        step_count = weights.sum()
        if validate and bool(step_count <= 0):
            raise ValueError("step_mask must select at least one step.")
        return (value * weights).sum() / step_count
    if reduction == "trajectory_mean":
        step_counts = weights.sum(dim=-1)
        if validate and bool((step_counts.masked_select(candidate_mask) <= 0).any()):
            raise ValueError("selected candidates must contain at least one rollout step.")
        trajectory_value = (value * weights).sum(dim=-1) / step_counts.clamp_min(1)
        candidate_weights = candidate_mask.to(dtype=trajectory_value.dtype)
        if validate and bool(candidate_weights.sum() <= 0):
            raise ValueError("group_mask must select at least one candidate.")
        return (trajectory_value * candidate_weights).sum() / candidate_weights.sum().clamp_min(1)
    raise ValueError("step_reduction must be 'step_mean' or 'trajectory_mean'.")


register(ContinuousGRPOLoss)


__all__ = ["ContinuousGRPOLoss"]
