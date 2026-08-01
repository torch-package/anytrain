from __future__ import annotations

import torch
from torch import Tensor

from ..routing import SparseExpertRouting


def expert_usage(routing: SparseExpertRouting) -> Tensor:
    """Fraction of active route assignments sent to each expert."""

    active_mask = routing.active_mask()
    flat_indices = routing.indices.reshape(-1, routing.top_k)
    flat_weights = routing.weights.reshape(-1, routing.top_k)
    selected = active_mask.unsqueeze(-1) & (flat_weights != 0)
    if not bool(selected.any().item()):
        return routing.probabilities.new_zeros(routing.num_experts)

    counts = torch.bincount(
        flat_indices[selected],
        minlength=routing.num_experts,
    ).to(dtype=routing.probabilities.dtype)
    return counts / counts.sum().clamp_min(torch.finfo(counts.dtype).tiny)


def effective_experts(routing: SparseExpertRouting) -> Tensor:
    """Entropy perplexity of the observed expert assignment distribution."""

    usage = expert_usage(routing)
    if not bool((usage > 0).any().item()):
        return usage.new_zeros(())
    log_usage = usage.clamp_min(torch.finfo(usage.dtype).tiny).log()
    return (-(usage * log_usage).sum()).exp()


def load_balance_loss(routing: SparseExpertRouting) -> Tensor:
    """Switch-style auxiliary load-balance loss.

    The returned scalar is num_experts * sum(mean_probability_i * usage_i).
    It is intentionally not applied inside MoELayer.forward so downstream
    LightningModules can choose the coefficient and logging semantics.
    """

    probability_mean = _probability_mean(routing)
    usage = expert_usage(routing).to(dtype=probability_mean.dtype)
    return routing.num_experts * (probability_mean * usage).sum()


def router_z_loss(routing: SparseExpertRouting) -> Tensor:
    """Mean squared log-normalizer over active routed positions."""

    active_mask = routing.active_mask()
    flat_logits = routing.logits.reshape(-1, routing.num_experts)
    if not bool(active_mask.any().item()):
        return flat_logits.new_zeros(())
    log_z = flat_logits[active_mask].logsumexp(dim=-1)
    return log_z.pow(2).mean()


def _probability_mean(routing: SparseExpertRouting) -> Tensor:
    active_mask = routing.active_mask()
    flat_probabilities = routing.probabilities.reshape(-1, routing.num_experts)
    if not bool(active_mask.any().item()):
        return flat_probabilities.new_zeros(routing.num_experts)
    return flat_probabilities[active_mask].mean(dim=0)


__all__ = [
    "effective_experts",
    "expert_usage",
    "load_balance_loss",
    "router_z_loss",
]
