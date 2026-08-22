from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor

from . import _checks
from .output import QuantizeOutput


@dataclass(eq=False)
class QuantizationUsage:
    counts: Tensor
    active_count: Tensor
    total_count: Tensor
    codebook_sizes: tuple[int, ...]

    @property
    def num_codebooks(self) -> int:
        return len(self.codebook_sizes)

    @property
    def codebook_denominator(self) -> Tensor:
        return self.counts.new_tensor(self.codebook_sizes, dtype=torch.float32)

    @property
    def observations(self) -> Tensor:
        return self.counts.sum(dim=-1)

    @property
    def active_ratio(self) -> Tensor:
        return _safe_divide(self.active_count, self.total_count)

    @property
    def unique_count(self) -> Tensor:
        return (self.counts > 0).sum(dim=-1).to(dtype=self.counts.dtype)

    @property
    def unique_ratio(self) -> Tensor:
        return _safe_divide(self.unique_count, self.codebook_denominator)

    @property
    def dead_count(self) -> Tensor:
        return self.codebook_denominator - self.unique_count

    @property
    def dead_ratio(self) -> Tensor:
        return _safe_divide(self.dead_count, self.codebook_denominator)

    @property
    def entropy(self) -> Tensor:
        probabilities = _safe_divide(self.counts, self.observations.unsqueeze(-1))
        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        terms = torch.where(
            probabilities > 0,
            probabilities * log_probabilities,
            torch.zeros_like(probabilities),
        )
        return -terms.sum(dim=-1)

    @property
    def perplexity(self) -> Tensor:
        return torch.where(
            self.observations > 0,
            self.entropy.exp(),
            torch.zeros_like(self.observations),
        )


def quantization_loss_metrics(
    output: QuantizeOutput,
    *,
    prefix: str | None = None,
) -> dict[str, Tensor]:
    if output.loss is None:
        return {}
    metrics = {
        "commitment": output.loss.commitment,
        "codebook": output.loss.codebook,
        "total": output.loss.total,
    }
    return _prefix_metrics(metrics, prefix)


def quantization_usage_metrics(
    usage: QuantizationUsage,
    *,
    metrics: Iterable[str] = ("usage", "active"),
) -> dict[str, Tensor]:
    """Flatten codebook usage into scalar, logging-friendly relative metric keys."""

    enabled = _metric_groups(metrics, allowed={"usage", "active"})
    observations = usage.observations
    entropy = usage.entropy
    unique_count = usage.unique_count
    denominator = usage.codebook_denominator
    perplexity = torch.where(
        observations > 0,
        entropy.exp(),
        torch.zeros_like(observations),
    )
    unique_ratio = _safe_divide(unique_count, denominator)
    dead_count = denominator - unique_count
    dead_ratio = _safe_divide(dead_count, denominator)
    active_count = usage.active_count
    active_ratio = usage.active_ratio

    values: dict[str, Tensor] = {}
    for book_index in range(usage.num_codebooks):
        book = f"book_{book_index}"
        if "usage" in enabled:
            values.update(
                {
                    f"{book}/usage/observations": observations[book_index],
                    f"{book}/usage/perplexity": perplexity[book_index],
                    f"{book}/usage/entropy": entropy[book_index],
                    f"{book}/usage/unique_count": unique_count[book_index],
                    f"{book}/usage/unique_ratio": unique_ratio[book_index],
                    f"{book}/usage/dead_count": dead_count[book_index],
                    f"{book}/usage/dead_ratio": dead_ratio[book_index],
                }
            )
        if "active" in enabled:
            values.update(
                {
                    f"{book}/active/count": active_count[book_index],
                    f"{book}/active/ratio": active_ratio[book_index],
                }
            )
    return values


def quantization_usage(
    output: QuantizeOutput,
    *,
    codebook_sizes: Sequence[int],
) -> QuantizationUsage:
    return codebook_usage(
        output.indices,
        codebook_sizes=codebook_sizes,
        active_codebook_mask=output.active_codebook_mask,
    )


def codebook_usage(
    indices: Tensor,
    *,
    codebook_sizes: Sequence[int],
    active_codebook_mask: Tensor | None = None,
) -> QuantizationUsage:
    """Summarize discrete codebook assignments.

    ``indices == -1`` is treated as inactive/unassigned. If
    ``active_codebook_mask`` is omitted, active counts are inferred from
    ``indices >= 0``; otherwise valid observations require both an active mask
    entry and a non-negative index.
    """

    flat_indices, active_mask, valid_mask = _flatten_indices(
        indices,
        codebook_sizes=codebook_sizes,
        active_codebook_mask=active_codebook_mask,
    )
    observed_codebooks = flat_indices.shape[1]
    observed_codebook_sizes = _normalize_codebook_sizes(codebook_sizes)[:observed_codebooks]
    max_codebook_size = max(observed_codebook_sizes)
    counts = torch.zeros(
        observed_codebooks,
        max_codebook_size,
        dtype=torch.float32,
        device=indices.device,
    )
    for book_index in range(observed_codebooks):
        selected = flat_indices[:, book_index][valid_mask[:, book_index]]
        if selected.numel() == 0:
            continue
        counts[book_index] = torch.bincount(
            selected,
            minlength=max_codebook_size,
        )[:max_codebook_size].to(dtype=torch.float32)

    return QuantizationUsage(
        counts=counts,
        active_count=active_mask.sum(dim=0).to(dtype=torch.float32),
        total_count=torch.full(
            (observed_codebooks,),
            flat_indices.shape[0],
            dtype=torch.float32,
            device=indices.device,
        ),
        codebook_sizes=observed_codebook_sizes,
    )


def valid_indices_by_codebook(
    indices: Tensor,
    *,
    codebook_sizes: Sequence[int],
    active_codebook_mask: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """Return valid non-negative assignment indices for each observed codebook."""

    flat_indices, _, valid_mask = _flatten_indices(
        indices,
        codebook_sizes=codebook_sizes,
        active_codebook_mask=active_codebook_mask,
    )
    return tuple(
        flat_indices[:, book_index][valid_mask[:, book_index]]
        for book_index in range(flat_indices.shape[1])
    )


def _flatten_indices(
    indices: Tensor,
    *,
    codebook_sizes: Sequence[int],
    active_codebook_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    codebook_sizes = _normalize_codebook_sizes(codebook_sizes)
    _validate_index_inputs(indices, codebook_sizes=codebook_sizes)
    flat_indices = _reshape_indices(indices, num_codebooks=len(codebook_sizes))
    observed_codebook_sizes = codebook_sizes[: flat_indices.shape[1]]
    if active_codebook_mask is None:
        active_mask = flat_indices >= 0
    else:
        if not isinstance(active_codebook_mask, Tensor):
            raise TypeError("active_codebook_mask must be a Tensor or None.")
        if active_codebook_mask.shape != indices.shape:
            raise ValueError(
                "active_codebook_mask must have the same shape as indices: "
                f"got {tuple(active_codebook_mask.shape)} and {tuple(indices.shape)}."
            )
        if active_codebook_mask.dtype != torch.bool:
            raise TypeError("active_codebook_mask must use torch.bool.")
        if active_codebook_mask.device != indices.device:
            raise ValueError("active_codebook_mask and indices must be on the same device.")
        active_mask = _reshape_indices(active_codebook_mask, num_codebooks=len(codebook_sizes))

    if flat_indices.numel() > 0:
        max_values = flat_indices.new_tensor(observed_codebook_sizes).view(1, -1)
        invalid = (flat_indices < -1) | (flat_indices >= max_values)
        if bool(invalid.any().item()):
            raise ValueError(
                "indices must contain active values in each codebook range "
                "or -1 for inactive codebooks."
            )
    valid_mask = active_mask & (flat_indices >= 0)
    return flat_indices, active_mask, valid_mask


def _reshape_indices(indices: Tensor, *, num_codebooks: int) -> Tensor:
    if num_codebooks == 1:
        return indices.reshape(-1, 1)
    if indices.ndim == 0:
        raise ValueError("multi-codebook indices must end with a codebook dimension.")
    observed_codebooks = indices.shape[-1]
    if observed_codebooks <= 0 or observed_codebooks > num_codebooks:
        raise ValueError(
            "indices last dimension must be in [1, num_codebooks]: "
            f"got {observed_codebooks}, num_codebooks={num_codebooks}."
        )
    return indices.reshape(-1, observed_codebooks)


def _validate_index_inputs(indices: Tensor, *, codebook_sizes: tuple[int, ...]) -> None:
    if not isinstance(indices, Tensor):
        raise TypeError("indices must be a Tensor.")
    _checks.index_dtype(indices)
    if not codebook_sizes:
        raise ValueError("codebook_sizes must contain at least one value.")


def _normalize_codebook_sizes(codebook_sizes: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(codebook_sizes)
    if not sizes:
        raise ValueError("codebook_sizes must contain at least one value.")
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError(f"each codebook size must be an integer, got {size!r}.")
        if size <= 0:
            raise ValueError(f"each codebook size must be positive, got {size}.")
    return tuple(int(size) for size in sizes)


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    numerator = numerator.to(dtype=torch.float32)
    denominator = denominator.to(dtype=torch.float32)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny),
        torch.zeros_like(numerator),
    )


def _metric_groups(values: Iterable[str], *, allowed: set[str]) -> set[str]:
    result = {value.strip() for value in values}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown quantization metrics: {sorted(unknown)!r}.")
    return result


def _prefix_metrics(
    metrics: dict[str, Tensor],
    prefix: str | None,
) -> dict[str, Tensor]:
    if prefix is None:
        return metrics
    prefix = prefix.strip("/")
    if not prefix:
        return metrics
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


__all__ = [
    "QuantizationUsage",
    "codebook_usage",
    "quantization_loss_metrics",
    "quantization_usage_metrics",
    "quantization_usage",
    "valid_indices_by_codebook",
]
