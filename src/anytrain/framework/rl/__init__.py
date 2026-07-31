"""Tensor-level RL objectives for preference and rollout training."""

from .advantage import group_relative_advantages
from .dpo import DPOLoss
from .grpo import GRPOLoss
from .logprob import gather_token_logps, masked_mean, sequence_logps
from .types import RolloutBatch, SequenceReduction, TokenReduction

__all__ = [
    "DPOLoss",
    "GRPOLoss",
    "RolloutBatch",
    "SequenceReduction",
    "TokenReduction",
    "gather_token_logps",
    "group_relative_advantages",
    "masked_mean",
    "sequence_logps",
]
