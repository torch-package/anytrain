from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .routing import ExpertRouting


def expert_usage(routing: ExpertRouting) -> Tensor:
    """Fraction of non-dropped actual route weight mass per expert.

    Dense routing uses the full gate probabilities. Top-k routing scatters the
    selected combine weights back to the expert axis, so this reflects the
    actual expert mixture used by MoE execution rather than the pre-top-k gate
    distribution.
    """

    sums = _route_weight_sums(routing)
    if not bool((sums > 0).any().item()):
        return sums
    return sums / sums.sum().clamp_min(torch.finfo(sums.dtype).tiny)


def effective_experts(routing: ExpertRouting) -> Tensor:
    """Entropy perplexity of the actual route-weight usage distribution."""

    usage = expert_usage(routing)
    if not bool((usage > 0).any().item()):
        return usage.new_zeros(())
    log_usage = usage.clamp_min(torch.finfo(usage.dtype).tiny).log()
    return (-(usage * log_usage).sum()).exp()


def load_balance_loss(routing: ExpertRouting) -> Tensor:
    """Auxiliary load-balance loss from gate probability and actual usage.

    The returned scalar is num_experts * sum(mean_gate_probability_i *
    actual_usage_i). It is intentionally not applied inside MoE.forward so
    downstream LightningModules can choose the coefficient and logging
    semantics.
    """

    probability_mean = expert_probability_mean(routing)
    usage = expert_usage(routing).to(dtype=probability_mean.dtype)
    return routing.num_experts * (probability_mean * usage).sum()


def router_z_loss(routing: ExpertRouting) -> Tensor:
    """Mean squared log-normalizer over active routed positions."""

    active_mask = routing.active_mask()
    flat_logits = routing.logits.reshape(-1, routing.num_experts)
    if not bool(active_mask.any().item()):
        return flat_logits.new_zeros(())
    log_z = flat_logits[active_mask].logsumexp(dim=-1)
    return log_z.pow(2).mean()


def expert_probability_mean(routing: ExpertRouting) -> Tensor:
    """Mean full gate probability per expert over active routed positions."""

    return _probability_mean(routing)


def expert_route_weight_mean(routing: ExpertRouting) -> Tensor:
    """Mean actual combine weight per expert over active routed positions."""

    active_count = _active_count(routing)
    if active_count == 0:
        return routing.probabilities.new_zeros(routing.num_experts)
    return _route_weight_sums(routing) / active_count


def tokens_per_expert(routing: ExpertRouting) -> Tensor:
    """Active routed-position counts per expert.

    Dense routing counts active positions with non-zero full gate probability.
    Top-k routing counts selected non-zero combine weights.
    """

    active_mask = routing.active_mask()
    counts = torch.zeros(
        routing.num_experts,
        dtype=torch.long,
        device=routing.probabilities.device,
    )
    if routing.is_dense:
        flat_probabilities = routing.probabilities.reshape(-1, routing.num_experts)
        selected = active_mask.unsqueeze(-1) & (flat_probabilities != 0)
        return selected.sum(dim=0).to(dtype=torch.long, device=routing.probabilities.device)

    indices, weights, top_k = routing.require_topk()
    flat_indices = indices.reshape(-1, top_k)
    flat_weights = weights.reshape(-1, top_k)
    selected = active_mask.unsqueeze(-1) & (flat_weights != 0)
    safe_indices = torch.where(selected, flat_indices, torch.zeros_like(flat_indices))
    return counts.scatter_add(
        0,
        safe_indices.reshape(-1).to(dtype=torch.long),
        selected.reshape(-1).to(dtype=counts.dtype),
    )


def dropped_token_ratio(
    routing: ExpertRouting,
    dropped_mask: Tensor | None = None,
) -> Tensor:
    if routing.is_dense:
        return routing.probabilities.new_zeros(())

    indices, weights, _top_k = routing.require_topk()
    del indices
    active_mask = routing.active_mask()
    if not bool(active_mask.any().item()):
        return routing.probabilities.new_zeros(())
    if dropped_mask is None:
        dropped = weights.sum(dim=-1) <= 0
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


def moe_routing_metrics(
    routing: ExpertRouting,
    *,
    tokens: Tensor | None = None,
    dropped_mask: Tensor | None = None,
    metrics: Iterable[str] = (
        "usage",
        "gate_probability",
        "route_weight",
        "tokens",
        "dropped",
    ),
) -> dict[str, Tensor]:
    metric_set = _metrics(metrics)
    values: dict[str, Tensor] = {}
    if "usage" in metric_set:
        usage = expert_usage(routing)
        values["effective_experts"] = effective_experts(routing)
        values.update(_summary_metrics(usage, "usage"))
    if "gate_probability" in metric_set:
        values.update(
            _summary_metrics(expert_probability_mean(routing), "gate_probability")
        )
    if "route_weight" in metric_set:
        values.update(
            _summary_metrics(expert_route_weight_mean(routing), "route_weight")
        )
    if "tokens" in metric_set:
        assignment_counts = tokens if tokens is not None else tokens_per_expert(routing)
        values.update(
            _summary_metrics(
                assignment_counts.to(dtype=routing.probabilities.dtype),
                "tokens",
            )
        )
        values["tokens/total"] = assignment_counts.sum().to(dtype=routing.probabilities.dtype)
    if "dropped" in metric_set:
        values["dropped/ratio"] = dropped_token_ratio(routing, dropped_mask)
    return values


def _route_weight_sums(routing: ExpertRouting) -> Tensor:
    active_mask = routing.active_mask()
    if routing.is_dense:
        flat_probabilities = routing.probabilities.reshape(-1, routing.num_experts)
        if not bool(active_mask.any().item()):
            return flat_probabilities.new_zeros(routing.num_experts)
        return flat_probabilities[active_mask].sum(dim=0)

    indices, weights, top_k = routing.require_topk()
    flat_indices = indices.reshape(-1, top_k)
    flat_weights = weights.reshape(-1, top_k)
    selected = active_mask.unsqueeze(-1) & (flat_weights != 0)
    sums = routing.probabilities.new_zeros(routing.num_experts)
    if not bool(selected.any().item()):
        return sums
    return sums.scatter_add(
        0,
        flat_indices[selected],
        flat_weights[selected].to(dtype=sums.dtype),
    )


def _active_count(routing: ExpertRouting) -> int:
    return int(routing.active_mask().sum().item())


def _probability_mean(routing: ExpertRouting) -> Tensor:
    active_mask = routing.active_mask()
    flat_probabilities = routing.probabilities.reshape(-1, routing.num_experts)
    if not bool(active_mask.any().item()):
        return flat_probabilities.new_zeros(routing.num_experts)
    return flat_probabilities[active_mask].mean(dim=0)


def _summary_metrics(values: Tensor, name: str) -> dict[str, Tensor]:
    flat = values.reshape(-1)
    if flat.ndim != 1:
        raise ValueError(f"MoE metric {name} must flatten to 1D, got shape {tuple(values.shape)}.")
    return {
        f"{name}_mean": flat.mean(),
        f"{name}_std": flat.std(unbiased=False),
        f"{name}_min": flat.min(),
        f"{name}_max": flat.max(),
    }


def _metrics(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"usage", "gate_probability", "route_weight", "tokens", "dropped"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown MoE routing metrics: {sorted(unknown)!r}.")
    if not result:
        raise ValueError("metrics must contain at least one value.")
    return result


__all__ = [
    "effective_experts",
    "expert_usage",
    "expert_route_weight_mean",
    "expert_probability_mean",
    "load_balance_loss",
    "dropped_token_ratio",
    "router_z_loss",
    "moe_routing_metrics",
    "tokens_per_expert",
]
