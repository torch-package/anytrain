from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .output import LossItem


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

        frame_count = mask.sum(dim=1).clamp_min(1)
        codebook_losses: list[Tensor] = []
        total_loss: Tensor | None = None
        accuracies: list[Tensor] = []
        for codebook, value in enumerate(logits):
            target = labels[..., codebook]
            safe_value = value.masked_fill(~mask[..., None], 0)
            safe_target = target.masked_fill(~mask, 0).to(dtype=torch.long)
            loss = F.cross_entropy(
                safe_value.movedim(-1, 1),
                safe_target,
                reduction="none",
            ).masked_fill(~mask, 0)
            row_loss = loss.sum(dim=1) / frame_count
            if include_details:
                codebook_losses.append(row_loss)
            else:
                total_loss = row_loss if total_loss is None else total_loss + row_loss
            if include_details and include_top1:
                correct = safe_value.argmax(dim=-1).eq(safe_target) & mask
                accuracies.append(correct.sum(dim=1))

        if not include_details:
            if total_loss is None:
                raise ValueError("codebook logits must provide at least one tensor.")
            return LossItem(loss=total_loss / len(logits), details=None)

        codebook_loss_tensor = torch.stack(codebook_losses, dim=-1)
        details = {
            f"codebook_{codebook}": codebook_loss_tensor[:, codebook]
            for codebook in range(len(logits))
        }
        if include_top1:
            codebook_top1 = torch.stack(accuracies, dim=-1) / frame_count[:, None]
            details.update(
                {
                    f"codebook_{codebook}_top1": codebook_top1[:, codebook]
                    for codebook in range(len(logits))
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
    valid_targets = labels[mask].to(dtype=torch.long)
    limits = torch.tensor(
        [value.size(-1) for value in logits],
        device=labels.device,
        dtype=torch.long,
    )
    if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
        raise ValueError("valid codebook target is outside its logits vocabulary.")


def _is_signed_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


__all__ = ["MaskedCodebookCrossEntropyLoss"]
