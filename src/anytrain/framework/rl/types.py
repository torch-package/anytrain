from __future__ import annotations

from typing import Literal, TypedDict

from torch import Tensor
from typing_extensions import NotRequired

TokenReduction = Literal["token_mean", "sequence_mean"]
SequenceReduction = Literal["sum", "mean"]
StepReduction = Literal["step_mean", "trajectory_mean"]


class RolloutBatch(TypedDict):
    """Tensor contract for offline group rollouts used by GRPO-style objectives."""

    policy_token_logps: Tensor
    old_token_logps: Tensor
    rewards: Tensor
    response_mask: Tensor
    ref_token_logps: NotRequired[Tensor | None]
    group_mask: NotRequired[Tensor | None]


__all__ = ["RolloutBatch", "SequenceReduction", "StepReduction", "TokenReduction"]
