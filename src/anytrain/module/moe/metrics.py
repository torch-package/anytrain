from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .routing import SparseExpertRouting


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


def expert_probability_mean(routing: SparseExpertRouting) -> Tensor:
    """Mean router probability per expert over active routed positions."""

    return _probability_mean(routing)


def tokens_per_expert(routing: SparseExpertRouting) -> Tensor:
    """Active sparse assignment counts per expert."""

    active_mask = routing.active_mask()
    flat_indices = routing.indices.reshape(-1, routing.top_k)
    flat_weights = routing.weights.reshape(-1, routing.top_k)
    selected = active_mask.unsqueeze(-1) & (flat_weights != 0)
    counts = torch.zeros(
        routing.num_experts,
        dtype=torch.long,
        device=routing.probabilities.device,
    )
    if not bool(selected.any().item()):
        return counts
    return torch.bincount(
        flat_indices[selected],
        minlength=routing.num_experts,
    ).to(dtype=torch.long, device=routing.probabilities.device)


def dropped_token_ratio(
    routing: SparseExpertRouting,
    dropped_mask: Tensor | None = None,
) -> Tensor:
    active_mask = routing.active_mask()
    if not bool(active_mask.any().item()):
        return routing.probabilities.new_zeros(())
    if dropped_mask is None:
        dropped = routing.weights.sum(dim=-1) <= 0
    else:
        if dropped_mask.dtype != torch.bool:
            raise TypeError(f"dropped_mask must be bool, got dtype={dropped_mask.dtype}.")
        dropped = dropped_mask
    try:
        dropped = torch.broadcast_to(dropped, routing.leading_shape)
    except RuntimeError as exc:
        raise ValueError(
            "dropped_mask must be broadcastable to routing leading shape: "
            f"got {tuple(dropped_mask.shape) if dropped_mask is not None else tuple(dropped.shape)}, "
            f"leading shape {tuple(routing.leading_shape)}."
        ) from exc
    dropped = dropped.reshape(-1).to(device=active_mask.device)
    return dropped[active_mask].to(dtype=routing.probabilities.dtype).mean()


def sparse_moe_routing_metrics(
    routing: SparseExpertRouting,
    *,
    tokens: Tensor | None = None,
    dropped_mask: Tensor | None = None,
    metrics: Iterable[str] = ("usage", "probability", "tokens", "dropped"),
) -> dict[str, Tensor]:
    metric_set = _metrics(metrics)
    values: dict[str, Tensor] = {}
    if "usage" in metric_set:
        usage = expert_usage(routing)
        values["effective_experts"] = effective_experts(routing)
        values.update(_per_expert_metrics(usage, "usage"))
    if "probability" in metric_set:
        values.update(_per_expert_metrics(expert_probability_mean(routing), "probability"))
    if "tokens" in metric_set:
        assignment_counts = tokens if tokens is not None else tokens_per_expert(routing)
        values.update(
            _per_expert_metrics(
                assignment_counts.to(dtype=routing.probabilities.dtype),
                "tokens",
            )
        )
        values["tokens/total"] = assignment_counts.sum().to(dtype=routing.probabilities.dtype)
    if "dropped" in metric_set:
        values["dropped/ratio"] = dropped_token_ratio(routing, dropped_mask)
    return values


def _probability_mean(routing: SparseExpertRouting) -> Tensor:
    active_mask = routing.active_mask()
    flat_probabilities = routing.probabilities.reshape(-1, routing.num_experts)
    if not bool(active_mask.any().item()):
        return flat_probabilities.new_zeros(routing.num_experts)
    return flat_probabilities[active_mask].mean(dim=0)


def _per_expert_metrics(values: Tensor, name: str) -> dict[str, Tensor]:
    flat = values.reshape(-1)
    return {f"expert_{index}/{name}": value for index, value in enumerate(flat)}


def _metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"usage", "probability", "tokens", "dropped"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown sparse MoE metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("metrics must contain at least one value.")
    return result


__all__ = [
    "effective_experts",
    "expert_usage",
    "expert_probability_mean",
    "load_balance_loss",
    "dropped_token_ratio",
    "router_z_loss",
    "sparse_moe_routing_metrics",
    "tokens_per_expert",
]
