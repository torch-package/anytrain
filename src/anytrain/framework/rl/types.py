from __future__ import annotations

from typing import Literal

from torch import Tensor
from typing_extensions import TypedDict

TokenReduction = Literal["token_mean", "sequence_mean"]
SequenceReduction = Literal["sum", "mean"]
StepReduction = Literal["step_mean", "trajectory_mean"]
NeighborReduction = Literal["neighbor_mean", "anchor_step_mean"]


class _RolloutBatchOptional(TypedDict, total=False):
    ref_token_logps: Tensor | None
    group_mask: Tensor | None


class RolloutBatch(_RolloutBatchOptional):
    """Tensor contract for token/logit group rollouts used by GRPO objectives.

    This stays task-agnostic by carrying only tensors that the objective consumes.
    Downstream projects own rollout records, reward components, prompt schema, and
    conversion from their domain-specific candidates into this batch.
    """

    policy_token_logps: Tensor
    old_token_logps: Tensor
    rewards: Tensor
    response_mask: Tensor


class _ContinuousRolloutBatchOptional(TypedDict, total=False):
    kl_values: Tensor | None
    group_mask: Tensor | None


class ContinuousRolloutBatch(_ContinuousRolloutBatchOptional):
    """Tensor contract for continuous latent transition GRPO objectives.

    ``policy_step_logps`` and ``old_step_logps`` describe per-step trajectory
    log-probabilities, usually from a downstream flow/diffusion sampler.
    """

    policy_step_logps: Tensor
    old_step_logps: Tensor
    rewards: Tensor
    step_mask: Tensor


class _NeighborRolloutBatchOptional(TypedDict, total=False):
    kl_values: Tensor | None
    group_mask: Tensor | None
    anchor_mask: Tensor | None


class NeighborRolloutBatch(_NeighborRolloutBatchOptional):
    """Tensor contract for neighbor/symmetric flow GRPO objectives.

    ``policy_neighbor_logps`` and ``old_neighbor_logps`` describe the surrogate
    distribution over grouped candidate neighbors for sampled anchor states.
    """

    policy_neighbor_logps: Tensor
    old_neighbor_logps: Tensor
    rewards: Tensor
    neighbor_mask: Tensor


__all__ = [
    "ContinuousRolloutBatch",
    "NeighborRolloutBatch",
    "NeighborReduction",
    "RolloutBatch",
    "SequenceReduction",
    "StepReduction",
    "TokenReduction",
]
