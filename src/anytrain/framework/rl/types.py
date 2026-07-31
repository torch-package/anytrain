from __future__ import annotations

from typing import Literal, TypedDict

from torch import Tensor
from typing_extensions import NotRequired

TokenReduction = Literal["token_mean", "sequence_mean"]
SequenceReduction = Literal["sum", "mean"]
StepReduction = Literal["step_mean", "trajectory_mean"]
NeighborReduction = Literal["neighbor_mean", "anchor_step_mean"]


class RolloutBatch(TypedDict):
    """Tensor contract for token/logit group rollouts used by GRPO objectives.

    This stays task-agnostic by carrying only tensors that the objective consumes.
    Downstream projects own rollout records, reward components, prompt schema, and
    conversion from their domain-specific candidates into this batch.
    """

    policy_token_logps: Tensor
    old_token_logps: Tensor
    rewards: Tensor
    response_mask: Tensor
    ref_token_logps: NotRequired[Tensor | None]
    group_mask: NotRequired[Tensor | None]


class ContinuousRolloutBatch(TypedDict):
    """Tensor contract for continuous latent transition GRPO objectives.

    ``policy_step_logps`` and ``old_step_logps`` describe per-step trajectory
    log-probabilities, usually from a downstream flow/diffusion sampler.
    """

    policy_step_logps: Tensor
    old_step_logps: Tensor
    rewards: Tensor
    step_mask: Tensor
    kl_values: NotRequired[Tensor | None]
    group_mask: NotRequired[Tensor | None]


class NeighborRolloutBatch(TypedDict):
    """Tensor contract for neighbor/symmetric flow GRPO objectives.

    ``policy_neighbor_logps`` and ``old_neighbor_logps`` describe the surrogate
    distribution over grouped candidate neighbors for sampled anchor states.
    """

    policy_neighbor_logps: Tensor
    old_neighbor_logps: Tensor
    rewards: Tensor
    neighbor_mask: Tensor
    kl_values: NotRequired[Tensor | None]
    group_mask: NotRequired[Tensor | None]
    anchor_mask: NotRequired[Tensor | None]


__all__ = [
    "ContinuousRolloutBatch",
    "NeighborRolloutBatch",
    "NeighborReduction",
    "RolloutBatch",
    "SequenceReduction",
    "StepReduction",
    "TokenReduction",
]
