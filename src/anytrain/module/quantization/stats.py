from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .output import QuantizeOutput


@dataclass(eq=False)
class QuantizationUsage:
    counts: Tensor
    active_count: Tensor
    total_count: Tensor

    @property
    def num_codebooks(self) -> int:
        return int(self.counts.shape[0])

    @property
    def codebook_size(self) -> int:
        return int(self.counts.shape[1])

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
        denominator = self.counts.new_full(
            (self.num_codebooks,),
            float(self.codebook_size),
        )
        return _safe_divide(self.unique_count, denominator)

    @property
    def dead_count(self) -> Tensor:
        return self.counts.new_full(
            (self.num_codebooks,),
            float(self.codebook_size),
        ) - self.unique_count

    @property
    def dead_ratio(self) -> Tensor:
        denominator = self.counts.new_full(
            (self.num_codebooks,),
            float(self.codebook_size),
        )
        return _safe_divide(self.dead_count, denominator)

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


def quantization_loss_metrics(output: QuantizeOutput) -> dict[str, Tensor]:
    if output.loss is None:
        return {}
    return {
        "commitment": output.loss.commitment,
        "codebook": output.loss.codebook,
        "total": output.loss.total,
    }


def quantization_usage(
    output: QuantizeOutput,
    *,
    codebook_size: int,
    num_codebooks: int = 1,
) -> QuantizationUsage:
    return codebook_usage(
        output.indices,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        active_codebook_mask=output.active_codebook_mask,
    )


def codebook_usage(
    indices: Tensor,
    *,
    codebook_size: int,
    num_codebooks: int = 1,
    active_codebook_mask: Tensor | None = None,
) -> QuantizationUsage:
    flat_indices, active_mask, valid_mask = _flatten_indices(
        indices,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        active_codebook_mask=active_codebook_mask,
    )
    observed_codebooks = flat_indices.shape[1]
    counts = torch.zeros(
        observed_codebooks,
        codebook_size,
        dtype=torch.float32,
        device=indices.device,
    )
    for book_index in range(observed_codebooks):
        selected = flat_indices[:, book_index][valid_mask[:, book_index]]
        if selected.numel() == 0:
            continue
        counts[book_index] = torch.bincount(
            selected,
            minlength=codebook_size,
        ).to(dtype=torch.float32)

    return QuantizationUsage(
        counts=counts,
        active_count=active_mask.sum(dim=0).to(dtype=torch.float32),
        total_count=torch.full(
            (observed_codebooks,),
            flat_indices.shape[0],
            dtype=torch.float32,
            device=indices.device,
        ),
    )


def valid_indices_by_codebook(
    indices: Tensor,
    *,
    codebook_size: int,
    num_codebooks: int = 1,
    active_codebook_mask: Tensor | None = None,
) -> tuple[Tensor, ...]:
    flat_indices, _, valid_mask = _flatten_indices(
        indices,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        active_codebook_mask=active_codebook_mask,
    )
    return tuple(
        flat_indices[:, book_index][valid_mask[:, book_index]]
        for book_index in range(flat_indices.shape[1])
    )


def _flatten_indices(
    indices: Tensor,
    *,
    codebook_size: int,
    num_codebooks: int,
    active_codebook_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_index_inputs(indices, codebook_size=codebook_size, num_codebooks=num_codebooks)
    flat_indices = _reshape_indices(indices, num_codebooks=num_codebooks)
    if active_codebook_mask is None:
        active_mask = torch.ones_like(flat_indices, dtype=torch.bool)
    else:
        if active_codebook_mask.shape != indices.shape:
            raise ValueError(
                "active_codebook_mask must have the same shape as indices: "
                f"got {tuple(active_codebook_mask.shape)} and {tuple(indices.shape)}."
            )
        active_mask = _reshape_indices(
            active_codebook_mask.to(dtype=torch.bool),
            num_codebooks=num_codebooks,
        )

    if flat_indices.numel() > 0:
        invalid = (flat_indices < -1) | (flat_indices >= codebook_size)
        if bool(invalid.any().item()):
            raise ValueError(
                "indices must contain active values in [0, codebook_size) "
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


def _validate_index_inputs(indices: Tensor, *, codebook_size: int, num_codebooks: int) -> None:
    if torch.is_floating_point(indices) or torch.is_complex(indices):
        raise TypeError("indices must be an integer tensor.")
    if codebook_size <= 0:
        raise ValueError(f"codebook_size must be positive, got {codebook_size}.")
    if num_codebooks <= 0:
        raise ValueError(f"num_codebooks must be positive, got {num_codebooks}.")


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    numerator = numerator.to(dtype=torch.float32)
    denominator = denominator.to(dtype=torch.float32)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny),
        torch.zeros_like(numerator),
    )


__all__ = [
    "QuantizationUsage",
    "codebook_usage",
    "quantization_loss_metrics",
    "quantization_usage",
    "valid_indices_by_codebook",
]
