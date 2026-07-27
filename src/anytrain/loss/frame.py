from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .output import LossItem


def masked_frame_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    details: Mapping[str, Tensor] | None = None,
    detail_dtype: torch.dtype | None = None,
) -> LossItem:
    _validate_frame_tensors(prediction, target, mask, name="frame MSE")
    frame_mask = mask[..., None]
    safe_prediction = prediction.masked_fill(~frame_mask, 0)
    safe_target = target.masked_fill(~frame_mask, 0)
    frame_loss = F.mse_loss(safe_prediction, safe_target, reduction="none").mean(dim=-1)
    return _item(frame_loss, mask, details=details, detail_dtype=detail_dtype or target.dtype)


def masked_cosine_alignment_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detach_target: bool = True,
    detail_dtype: torch.dtype | None = None,
) -> LossItem:
    _validate_frame_tensors(prediction, target, mask, name="cosine alignment")
    frame_mask = mask[..., None]
    safe_prediction = prediction.masked_fill(~frame_mask, 0).float()
    safe_target = target.to(device=prediction.device)
    if detach_target:
        safe_target = safe_target.detach()
    safe_target = safe_target.masked_fill(~frame_mask, 0).float()
    normalized_prediction = F.normalize(safe_prediction, dim=-1)
    normalized_target = F.normalize(safe_target, dim=-1)
    frame_loss = (1 - (normalized_prediction * normalized_target).sum(dim=-1)).masked_fill(
        ~mask, 0
    )
    item = _item(frame_loss, mask, detail_dtype=detail_dtype or prediction.dtype)
    details = dict(item.details or {})
    details["cosine"] = 1 - item.loss
    return LossItem(item.loss, details)


class MaskedFrameMSELoss(nn.Module):
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        *,
        details: Mapping[str, Tensor] | None = None,
        detail_dtype: torch.dtype | None = None,
    ) -> LossItem:
        return masked_frame_mse_loss(
            prediction,
            target,
            mask,
            details=details,
            detail_dtype=detail_dtype,
        )


class MaskedCosineAlignmentLoss(nn.Module):
    def __init__(self, *, detach_target: bool = True) -> None:
        super().__init__()
        self.detach_target = detach_target

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        *,
        detail_dtype: torch.dtype | None = None,
    ) -> LossItem:
        return masked_cosine_alignment_loss(
            prediction,
            target,
            mask,
            detach_target=self.detach_target,
            detail_dtype=detail_dtype,
        )


def _validate_frame_tensors(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    name: str,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"{name} prediction and target shapes must match.")
    if prediction.dim() != 3 or mask.dim() != 2:
        raise ValueError(
            f"{name} prediction, target, and mask must have shapes [B, F, D], [B, F, D], and [B, F]."
        )
    if mask.shape != target.shape[:2]:
        raise ValueError(f"{name} mask must align with [batch, frame].")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} mask must be boolean.")


def _item(
    frame_loss: Tensor,
    mask: Tensor,
    *,
    details: Mapping[str, Tensor] | None = None,
    detail_dtype: torch.dtype,
) -> LossItem:
    weights = mask.to(dtype=frame_loss.dtype)
    frame_count = weights.sum(dim=1)
    loss = frame_loss.sum(dim=1) / frame_count.clamp_min(1)
    output_details = {"frames": frame_count.to(dtype=detail_dtype)}
    if details is not None:
        output_details.update({key: value.to(dtype=detail_dtype) for key, value in details.items()})
    return LossItem(loss=loss, details=output_details)
