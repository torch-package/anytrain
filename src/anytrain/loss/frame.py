from __future__ import annotations

import math
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
    safe_prediction, safe_target, _frame_mask = _prepared_pair(
        prediction,
        target,
        mask,
        detach_target=detach_target,
    )
    frame_loss = _cosine_distance(safe_prediction, safe_target).masked_fill(~mask, 0)
    item = _item(frame_loss, mask, detail_dtype=detail_dtype or prediction.dtype)
    details = dict(item.details or {})
    details["cosine"] = 1 - item.loss
    return LossItem(item.loss, details)


def teacher_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    smooth_l1_weight: float = 1.0,
    detach_target: bool = True,
    beta: float = 1.0,
    detail_dtype: torch.dtype | None = None,
) -> LossItem:
    """Masked teacher feature distill: ``1 - cosine`` plus optional Smooth L1.

    Both terms share the same ``[B, F]`` mask. Smooth L1 uses
    ``torch.nn.functional.smooth_l1_loss``. Details expose per-row
    ``cosine`` (similarity), ``smooth_l1``, and ``frames``.
    """
    smooth_l1_weight, beta = _teacher_options(smooth_l1_weight, beta)
    _validate_frame_tensors(prediction, target, mask, name="teacher")
    safe_prediction, safe_target, _frame_mask = _prepared_pair(
        prediction,
        target,
        mask,
        detach_target=detach_target,
    )
    cosine_distance = _cosine_distance(safe_prediction, safe_target).masked_fill(~mask, 0)
    smooth_l1 = (
        F.smooth_l1_loss(safe_prediction, safe_target, reduction="none", beta=beta)
        .mean(dim=-1)
        .masked_fill(~mask, 0)
    )
    frame_loss = cosine_distance + smooth_l1_weight * smooth_l1
    dtype = detail_dtype or prediction.dtype
    cosine_rows = _item(cosine_distance, mask, detail_dtype=dtype).loss
    smooth_rows = _item(smooth_l1, mask, detail_dtype=dtype).loss
    item = _item(frame_loss, mask, detail_dtype=dtype)
    details = dict(item.details or {})
    details["cosine"] = 1 - cosine_rows
    details["smooth_l1"] = smooth_rows
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


class TeacherLoss(nn.Module):
    """Teacher feature distillation with shared frame mask."""

    def __init__(
        self,
        smooth_l1_weight: float = 1.0,
        *,
        detach_target: bool = True,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.smooth_l1_weight, self.beta = _teacher_options(smooth_l1_weight, beta)
        self.detach_target = detach_target

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        *,
        detail_dtype: torch.dtype | None = None,
    ) -> LossItem:
        return teacher_loss(
            prediction,
            target,
            mask,
            smooth_l1_weight=self.smooth_l1_weight,
            detach_target=self.detach_target,
            beta=self.beta,
            detail_dtype=detail_dtype,
        )


def _teacher_options(smooth_l1_weight: object, beta: object) -> tuple[float, float]:
    return (
        _finite_float(smooth_l1_weight, name="smooth_l1_weight", positive=False),
        _finite_float(beta, name="beta", positive=True),
    )


def _finite_float(value: object, *, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive:
        if result <= 0:
            raise ValueError(f"{name} must be a positive float.")
    elif result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _prepared_pair(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detach_target: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    frame_mask = mask[..., None]
    safe_prediction = prediction.masked_fill(~frame_mask, 0).float()
    safe_target = target.to(device=prediction.device)
    if detach_target:
        safe_target = safe_target.detach()
    safe_target = safe_target.masked_fill(~frame_mask, 0).float()
    return safe_prediction, safe_target, frame_mask


def _cosine_distance(prediction: Tensor, target: Tensor) -> Tensor:
    normalized_prediction = F.normalize(prediction, dim=-1)
    normalized_target = F.normalize(target, dim=-1)
    return 1 - (normalized_prediction * normalized_target).sum(dim=-1)


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
