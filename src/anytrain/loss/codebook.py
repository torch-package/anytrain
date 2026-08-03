from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .output import LossItem


@dataclass(frozen=True)
class PackedCodebookLogits:
    """Valid-frame ``[N, K_q]`` logits, optional ``[N, Q]`` labels, and batch rows."""

    logits: tuple[Tensor, ...]
    labels: Tensor | None
    row_indices: Tensor
    batch_size: int


class MaskedCodebookCrossEntropyLoss(nn.Module):
    """Masked cross entropy for frame-aligned multi-codebook logits.

    ``include_details=False`` keeps only the per-row loss and avoids building
    codebook and top-1 metrics. ``validate=False`` is intended for trusted
    batches in a training hot path and skips all input checks.
    """

    def forward(
        self,
        logits: Sequence[Tensor],
        labels: Tensor,
        mask: Tensor,
        *,
        validate: bool = True,
        include_top1: bool = False,
        include_details: bool = True,
    ) -> LossItem:
        if validate:
            _validate_inputs(logits, labels, mask)

        frame_indices = mask.flatten().nonzero().flatten()
        packed = PackedCodebookLogits(
            logits=tuple(
                value.flatten(0, 1).index_select(0, frame_indices)
                for value in logits
            ),
            labels=labels.flatten(0, 1).index_select(0, frame_indices),
            row_indices=frame_indices.div(mask.size(1), rounding_mode="floor"),
            batch_size=labels.size(0),
        )
        return self.forward_packed(
            packed,
            validate=False,
            include_top1=include_top1,
            include_details=include_details,
        )

    def forward_packed(
        self,
        packed: PackedCodebookLogits,
        *,
        validate: bool = True,
        include_top1: bool = False,
        include_details: bool = True,
    ) -> LossItem:
        """Compute CE on valid frames and aggregate each frame back to its row."""
        if packed.labels is None:
            raise ValueError("packed codebook logits require labels for loss computation.")
        if not packed.logits:
            raise ValueError("codebook logits must provide at least one tensor.")
        if validate:
            _validate_packed_inputs(packed)

        labels = packed.labels.to(dtype=torch.long)
        frame_count = torch.bincount(
            packed.row_indices,
            minlength=packed.batch_size,
        )
        frame_denominator = frame_count.clamp_min(1)
        frame_losses: list[Tensor] = []
        total_frame_loss: Tensor | None = None
        accuracies: list[Tensor] = []
        for codebook, value in enumerate(packed.logits):
            target = labels[:, codebook]
            loss = F.cross_entropy(value, target, reduction="none")
            if include_details:
                frame_losses.append(loss)
            else:
                total_frame_loss = (
                    loss if total_frame_loss is None else total_frame_loss + loss
                )
            if include_details and include_top1:
                accuracies.append(value.argmax(dim=-1).eq(target).to(dtype=loss.dtype))

        if not include_details:
            if total_frame_loss is None:
                raise ValueError("codebook logits must provide at least one tensor.")
            row_loss = total_frame_loss.new_zeros(packed.batch_size).index_add(
                0,
                packed.row_indices,
                total_frame_loss,
            ) / frame_denominator
            return LossItem(loss=row_loss / len(packed.logits), details=None)

        frame_loss_tensor = torch.stack(frame_losses, dim=-1)
        codebook_loss_tensor = frame_loss_tensor.new_zeros(
            packed.batch_size,
            len(packed.logits),
        ).index_add(
            0,
            packed.row_indices,
            frame_loss_tensor,
        ) / frame_denominator[:, None]
        details = {
            f"codebook_{codebook}": codebook_loss_tensor[:, codebook]
            for codebook in range(len(packed.logits))
        }
        if include_top1:
            frame_top1 = torch.stack(accuracies, dim=-1)
            codebook_top1 = frame_top1.new_zeros(
                packed.batch_size,
                len(packed.logits),
            ).index_add(
                0,
                packed.row_indices,
                frame_top1,
            ) / frame_denominator[:, None]
            details.update(
                {
                    f"codebook_{codebook}_top1": codebook_top1[:, codebook]
                    for codebook in range(len(packed.logits))
                }
            )
        details["frames"] = frame_count.to(dtype=codebook_loss_tensor.dtype)
        return LossItem(loss=codebook_loss_tensor.mean(dim=-1), details=details)


def _validate_inputs(logits: Sequence[Tensor], labels: Tensor, mask: Tensor) -> None:
    if labels.ndim != 3 or mask.shape != labels.shape[:2]:
        raise ValueError(
            "codebook labels and mask must have shapes [B, F, Q] and [B, F]."
        )
    if not _is_signed_integer_dtype(labels.dtype):
        raise TypeError("codebook labels must use a signed integer dtype.")
    if mask.dtype != torch.bool:
        raise TypeError("codebook target mask must be boolean.")
    if len(logits) != labels.size(-1):
        raise ValueError("codebook logits must provide one tensor per codebook.")
    for value in logits:
        if value.ndim != 3 or value.shape[:2] != labels.shape[:2]:
            raise ValueError(
                "codebook logits must have shape [B, F, vocab] and align with labels."
            )
    if mask.device != labels.device or any(value.device != labels.device for value in logits):
        raise ValueError("codebook logits, labels, and mask must share a device.")
    valid_targets = labels[mask].to(dtype=torch.long)
    limits = torch.tensor(
        [value.size(-1) for value in logits],
        device=labels.device,
        dtype=torch.long,
    )
    if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
        raise ValueError("valid codebook target is outside its logits vocabulary.")


def _validate_packed_inputs(packed: PackedCodebookLogits) -> None:
    labels = packed.labels
    if labels is None:
        raise ValueError("packed codebook logits require labels for loss computation.")
    if packed.batch_size < 1:
        raise ValueError("packed codebook batch_size must be positive.")
    if labels.ndim != 2 or packed.row_indices.ndim != 1:
        raise ValueError(
            "packed codebook labels and row indices must have shapes [N, Q] and [N]."
        )
    if labels.size(0) != packed.row_indices.numel():
        raise ValueError("packed codebook labels and row indices must align.")
    if not _is_signed_integer_dtype(labels.dtype):
        raise TypeError("packed codebook labels must use a signed integer dtype.")
    if packed.row_indices.dtype != torch.long:
        raise TypeError("packed codebook row indices must use torch.long.")
    if len(packed.logits) != labels.size(-1):
        raise ValueError("packed codebook logits must provide one tensor per codebook.")
    if labels.device != packed.row_indices.device or any(
        value.device != labels.device for value in packed.logits
    ):
        raise ValueError("packed codebook logits, labels, and row indices must share a device.")
    for value in packed.logits:
        if value.ndim != 2 or value.size(0) != labels.size(0):
            raise ValueError(
                "packed codebook logits must have shape [N, vocab] and align with labels."
            )
    if bool(
        ((packed.row_indices < 0) | (packed.row_indices >= packed.batch_size)).any()
    ):
        raise ValueError("packed codebook row index is outside the batch.")
    limits = torch.tensor(
        [value.size(-1) for value in packed.logits],
        device=labels.device,
        dtype=torch.long,
    )
    valid_targets = labels.to(dtype=torch.long)
    if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
        raise ValueError("packed codebook target is outside its logits vocabulary.")


def _is_signed_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


__all__ = ["MaskedCodebookCrossEntropyLoss", "PackedCodebookLogits"]
