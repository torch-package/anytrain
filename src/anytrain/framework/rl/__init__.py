"""Tensor-level RL objectives for preference and rollout training."""

from .advantage import group_relative_advantages
from .continuous_grpo import ContinuousGRPOLoss
from .dpo import DPOLoss
from .grpo import GRPOLoss
from .logprob import gather_token_logps, masked_mean, sequence_logps
from .types import RolloutBatch, SequenceReduction, StepReduction, TokenReduction

__all__ = [
    "ContinuousGRPOLoss",
    "DPOLoss",
    "GRPOLoss",
    "RolloutBatch",
    "SequenceReduction",
    "StepReduction",
    "TokenReduction",
    "gather_token_logps",
    "group_relative_advantages",
    "masked_mean",
    "sequence_logps",
]
