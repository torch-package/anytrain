from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._validation import non_negative_real, positive_real, strict_bool
from ._observation import emit, recommend, register
from .advantage import _reward_values, group_relative_advantages
from .types import NeighborReduction


class NeighborGRPOLoss(nn.Module):
    """GRPO loss over distance-softmax neighbor surrogate policies.

    Use this when a downstream project has already built grouped flow or
    diffusion candidate trajectories and can provide surrogate log-probs for
    choosing candidate neighbors from sampled anchor states. The loss does not
    run sampling, reward scoring, trajectory construction, or model forwarding.
    """

    def __init__(
        self,
        *,
        clip_range: float = 0.2,
        kl_beta: float = 0.0,
        eps: float = 1e-8,
        adv_clip_max: float | None = None,
        advantage_norm_p: float = 2.0,
        neighbor_reduction: NeighborReduction = "anchor_step_mean",
    ) -> None:
        super().__init__()
        self.clip_range = non_negative_real(clip_range, "clip_range")
        self.kl_beta = non_negative_real(kl_beta, "kl_beta")
        self.eps = positive_real(eps, "eps")
        self.adv_clip_max = (
            None if adv_clip_max is None else positive_real(adv_clip_max, "adv_clip_max")
        )
        self.advantage_norm_p = positive_real(advantage_norm_p, "advantage_norm_p")
        if neighbor_reduction not in {"neighbor_mean", "anchor_step_mean"}:
            raise ValueError("neighbor_reduction must be 'neighbor_mean' or 'anchor_step_mean'.")

        self.neighbor_reduction: NeighborReduction = neighbor_reduction
        recommend(self)

    def forward(
        self,
        *,
        policy_neighbor_logps: Tensor,
        old_neighbor_logps: Tensor,
        rewards: Tensor,
        neighbor_mask: Tensor,
        kl_values: Tensor | None = None,
        group_mask: Tensor | None = None,
        anchor_mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        """Compute neighbor GRPO loss.

        Set ``validate=False`` only for batches whose tensor contracts have
        already been checked by their owner.
        """

        validate = strict_bool(validate, "validate")
        if validate:
            _validate_neighbor_tensors(policy_neighbor_logps, old_neighbor_logps, neighbor_mask)
            if rewards.shape != _candidate_shape(policy_neighbor_logps):
                raise ValueError("rewards must have shape [batch, group].")

        policy_neighbor_logps = policy_neighbor_logps.to(
            dtype=torch.promote_types(policy_neighbor_logps.dtype, torch.float32)
        )
        old_neighbor_logps = old_neighbor_logps.detach().to(
            dtype=torch.promote_types(old_neighbor_logps.dtype, torch.float32)
        )
        rewards = _reward_values(rewards.detach())
        candidate_mask = _candidate_mask(neighbor_mask, rewards, group_mask, validate=validate)
        selected_anchor_mask = _anchor_mask(neighbor_mask, anchor_mask, validate=validate)
        selected_neighbor_mask = _selected_neighbor_mask(
            neighbor_mask,
            candidate_mask,
            selected_anchor_mask,
            validate=validate,
        )
        if validate:
            _validate_selected_logps(
                policy_neighbor_logps,
                old_neighbor_logps,
                selected_neighbor_mask,
            )
        # Padding entries may be non-finite. Keep them out of ratio arithmetic
        # and preserve zero gradients for candidates excluded by the masks.
        policy_neighbor_logps = policy_neighbor_logps.masked_fill(
            ~selected_neighbor_mask,
            0,
        )
        old_neighbor_logps = old_neighbor_logps.masked_fill(
            ~selected_neighbor_mask,
            0,
        )
        advantages = _advantages(
            rewards,
            candidate_mask,
            eps=self.eps,
            norm_p=self.advantage_norm_p,
            validate=validate,
        ).detach()
        if self.adv_clip_max is not None:
            advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)

        log_ratio = policy_neighbor_logps - old_neighbor_logps
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        neighbor_advantages = advantages[:, None, None, :]
        policy_loss = -torch.minimum(
            ratio * neighbor_advantages,
            clipped_ratio * neighbor_advantages,
        )
        pg_loss = _reduce_neighbor_values(
            policy_loss,
            selected_neighbor_mask,
            self.neighbor_reduction,
            validate=validate,
        )

        if kl_values is None:
            if self.kl_beta > 0:
                raise ValueError("kl_values is required when kl_beta is positive.")
            kl_loss = policy_neighbor_logps.new_zeros(())
        else:
            if validate:
                _validate_kl_values(policy_neighbor_logps, kl_values, selected_neighbor_mask)
            kl_loss = _reduce_neighbor_values(
                kl_values.detach().masked_fill(~selected_neighbor_mask, 0),
                selected_neighbor_mask,
                self.neighbor_reduction,
                validate=validate,
            )

        total = pg_loss + policy_neighbor_logps.new_tensor(self.kl_beta) * kl_loss
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
                * _reduce_neighbor_values(
                    log_ratio.square(),
                    selected_neighbor_mask,
                    self.neighbor_reduction,
                    validate=validate,
                ),
                "clip_fraction": (
                    (torch.abs(ratio - 1.0) > self.clip_range)
                    .masked_select(selected_neighbor_mask)
                    .to(dtype=total.dtype)
                    .mean()
                ),
                "neighbor_count": selected_neighbor_mask.to(dtype=total.dtype).sum(),
                "anchor_step_count": selected_neighbor_mask.any(dim=-1).to(dtype=total.dtype).sum(),
                "anchor_count": selected_neighbor_mask.any(dim=(2, 3)).to(dtype=total.dtype).sum(),
                "candidate_count": candidate_mask.to(dtype=total.dtype).sum(),
            },
        )
        return total


def neighbor_logps_from_distances(
    distances: Tensor,
    mask: Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Convert non-negative neighbor distances to log-probs with softmax(-d / tau)."""

    if not isinstance(distances, Tensor):
        raise TypeError("distances must be a tensor.")
    if distances.ndim < 1:
        raise ValueError("distances must include a candidate dimension.")
    if distances.is_complex() or not distances.dtype.is_floating_point:
        raise TypeError("distances must be floating-point real values.")
    temperature = positive_real(temperature, "temperature")

    if mask is None:
        selected = torch.ones_like(distances, dtype=torch.bool)
    else:
        if not isinstance(mask, Tensor):
            raise TypeError("mask must be a tensor.")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be boolean.")
        if mask.shape != distances.shape:
            raise ValueError("mask must have the same shape as distances.")
        if mask.device != distances.device:
            raise ValueError("mask must be on the same device as distances.")
        selected = mask
    if bool((selected.sum(dim=-1) <= 0).any()):
        raise ValueError("each neighbor distribution must contain at least one candidate.")
    selected_distances = distances.masked_select(selected)
    if not bool(torch.isfinite(selected_distances).all()):
        raise ValueError("selected distances must be finite.")
    if bool((selected_distances < 0).any()):
        raise ValueError("selected distances must be non-negative.")

    logits = -distances / distances.new_tensor(temperature)
    logits = logits.masked_fill(~selected, float("-inf"))
    return F.log_softmax(logits, dim=-1)


def _validate_neighbor_tensors(policy: Tensor, old: Tensor, neighbor_mask: Tensor) -> None:
    if not isinstance(policy, Tensor) or not isinstance(old, Tensor):
        raise TypeError("policy_neighbor_logps and old_neighbor_logps must be tensors.")
    if policy.shape != old.shape:
        raise ValueError("policy_neighbor_logps and old_neighbor_logps must have the same shape.")
    if policy.ndim != 4:
        raise ValueError("neighbor log-probs must have shape [batch, anchors, steps, group].")
    if policy.is_complex() or old.is_complex():
        raise TypeError("neighbor log-probs must be real-valued.")
    if not isinstance(neighbor_mask, Tensor):
        raise TypeError("neighbor_mask must be a tensor.")
    if neighbor_mask.dtype != torch.bool:
        raise TypeError("neighbor_mask must be boolean.")
    if neighbor_mask.shape != policy.shape:
        raise ValueError("neighbor_mask must have the same shape as neighbor log-probs.")
    if neighbor_mask.device != policy.device:
        raise ValueError("neighbor_mask must be on the same device as neighbor log-probs.")
    if old.device != policy.device:
        raise ValueError("old_neighbor_logps must be on the same device as policy_neighbor_logps.")


def _candidate_shape(neighbor_logps: Tensor) -> tuple[int, int]:
    return (neighbor_logps.shape[0], neighbor_logps.shape[-1])


def _candidate_mask(
    neighbor_mask: Tensor,
    rewards: Tensor,
    group_mask: Tensor | None,
    *,
    validate: bool,
) -> Tensor:
    if validate:
        if rewards.device != neighbor_mask.device:
            raise ValueError("rewards must be on the same device as neighbor_mask.")
        if rewards.is_complex() or rewards.dtype == torch.bool:
            raise TypeError("rewards must be real-valued.")
    derived = neighbor_mask.any(dim=(1, 2))
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
            raise ValueError("group_mask cannot select candidates with no neighbor comparisons.")
    return group_mask


def _anchor_mask(
    neighbor_mask: Tensor,
    anchor_mask: Tensor | None,
    *,
    validate: bool,
) -> Tensor:
    derived = neighbor_mask.any(dim=(2, 3))
    if anchor_mask is None:
        return derived
    if validate:
        if not isinstance(anchor_mask, Tensor):
            raise TypeError("anchor_mask must be a tensor.")
        if anchor_mask.dtype != torch.bool:
            raise TypeError("anchor_mask must be boolean.")
        if anchor_mask.shape != derived.shape:
            raise ValueError("anchor_mask must have shape [batch, anchors].")
        if anchor_mask.device != neighbor_mask.device:
            raise ValueError("anchor_mask must be on the same device as neighbor_mask.")
        if bool((anchor_mask & ~derived).any()):
            raise ValueError("anchor_mask cannot select anchors with no neighbor comparisons.")
    return anchor_mask


def _selected_neighbor_mask(
    neighbor_mask: Tensor,
    candidate_mask: Tensor,
    anchor_mask: Tensor,
    *,
    validate: bool,
) -> Tensor:
    if validate:
        if bool((neighbor_mask & ~candidate_mask[:, None, None, :]).any()):
            raise ValueError("neighbor_mask cannot select candidates outside group_mask.")
        if bool((neighbor_mask & ~anchor_mask[:, :, None, None]).any()):
            raise ValueError("neighbor_mask cannot select anchors outside anchor_mask.")
    return neighbor_mask & candidate_mask[:, None, None, :] & anchor_mask[:, :, None, None]


def _validate_selected_logps(policy: Tensor, old: Tensor, selected_mask: Tensor) -> None:
    selected_policy = policy.masked_select(selected_mask)
    selected_old = old.masked_select(selected_mask)
    if not bool(selected_policy.numel()):
        raise ValueError("neighbor_mask must select at least one neighbor comparison.")
    if not bool(torch.isfinite(selected_policy).all()):
        raise ValueError("selected policy_neighbor_logps must be finite.")
    if not bool(torch.isfinite(selected_old).all()):
        raise ValueError("selected old_neighbor_logps must be finite.")


def _advantages(
    rewards: Tensor,
    group_mask: Tensor,
    *,
    eps: float,
    norm_p: float,
    validate: bool,
) -> Tensor:
    if norm_p == 2.0:
        return group_relative_advantages(rewards, group_mask, eps=eps, validate=validate)

    if validate and not bool(torch.isfinite(rewards.masked_select(group_mask)).all()):
        raise ValueError("selected rewards must be finite.")
    rewards = _reward_values(rewards)
    counts = group_mask.sum(dim=1)
    if validate and bool((counts < 2).any()):
        raise ValueError("each reward group must contain at least two valid candidates.")
    safe_rewards = rewards.masked_fill(~group_mask, 0)
    mean = safe_rewards.sum(dim=1, keepdim=True) / counts.unsqueeze(1)
    centered = (rewards - mean).masked_fill(~group_mask, 0)
    scale = (
        centered.abs().pow(rewards.new_tensor(float(norm_p))).sum(dim=1, keepdim=True)
        / counts.unsqueeze(1)
    ).pow(1.0 / norm_p)
    advantages = centered / (scale + eps)
    return advantages.masked_fill(~group_mask, 0)


def _validate_kl_values(policy: Tensor, kl_values: Tensor, selected_mask: Tensor) -> None:
    if not isinstance(kl_values, Tensor):
        raise TypeError("kl_values must be a tensor.")
    if kl_values.shape != policy.shape:
        raise ValueError("kl_values must have the same shape as policy_neighbor_logps.")
    if kl_values.device != policy.device:
        raise ValueError("kl_values must be on the same device as policy_neighbor_logps.")
    if kl_values.is_complex():
        raise TypeError("kl_values must be real-valued.")
    selected = kl_values.masked_select(selected_mask)
    if bool((selected < 0).any()):
        raise ValueError("selected kl_values must be non-negative.")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("selected kl_values must be finite.")


def _reduce_neighbor_values(
    value: Tensor,
    selected_mask: Tensor,
    reduction: NeighborReduction,
    *,
    validate: bool,
) -> Tensor:
    resolved = cast(NeighborReduction, reduction)
    value = value.to(dtype=torch.promote_types(value.dtype, torch.float32))
    weights = selected_mask.to(dtype=torch.int64)
    safe_value = value.masked_fill(~selected_mask, 0)
    if resolved == "neighbor_mean":
        neighbor_count = weights.sum()
        if validate and bool(neighbor_count <= 0):
            raise ValueError("neighbor_mask must select at least one neighbor comparison.")
        return (safe_value * weights).sum() / neighbor_count
    if resolved == "anchor_step_mean":
        neighbor_counts = weights.sum(dim=-1)
        valid_anchor_steps = neighbor_counts > 0
        anchor_step_value = (safe_value * weights).sum(dim=-1) / neighbor_counts.clamp_min(1)
        selected = anchor_step_value.masked_select(valid_anchor_steps)
        if validate and not bool(selected.numel()):
            raise ValueError("neighbor_mask must select at least one anchor-step comparison.")
        # ``valid_anchor_steps`` is a device mask; use a weighted reduction so
        # trusted calls do not materialize a dynamic host-side predicate.
        weights = valid_anchor_steps.to(dtype=torch.int64)
        return (anchor_step_value * weights).sum() / weights.sum().clamp_min(1)
    raise ValueError("neighbor_reduction must be 'neighbor_mean' or 'anchor_step_mean'.")


register(NeighborGRPOLoss)


__all__ = ["NeighborGRPOLoss", "neighbor_logps_from_distances"]
