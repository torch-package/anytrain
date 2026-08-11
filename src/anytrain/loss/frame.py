from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .abc import LossABC
from .observation import emit_diagnostics, loss_with_diagnostics


def masked_frame_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    loss, _frame_count = _masked_frame_mse_loss(prediction, target, mask)
    return loss


def masked_cosine_alignment_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detach_target: bool = True,
) -> Tensor:
    loss, _cosine, _frame_count = _masked_cosine_alignment_loss(
        prediction,
        target,
        mask,
        detach_target=detach_target,
    )
    return loss


def teacher_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    smooth_l1_weight: float = 1.0,
    detach_target: bool = True,
    beta: float = 1.0,
) -> Tensor:
    smooth_l1_weight, beta = _teacher_options(smooth_l1_weight, beta)
    loss, _cosine, _smooth_l1, _frame_count = _teacher_loss(
        prediction,
        target,
        mask,
        smooth_l1_weight=smooth_l1_weight,
        detach_target=detach_target,
        beta=beta,
    )
    return loss


@loss_with_diagnostics
class MaskedFrameMSELoss(LossABC):
    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        loss, frame_count = _masked_frame_mse_loss(prediction, target, mask)
        emit_diagnostics(self, {"valid_frames": frame_count})
        return loss


@loss_with_diagnostics
class MaskedCosineAlignmentLoss(LossABC):
    def __init__(self, *, detach_target: bool = True) -> None:
        super().__init__()
        self.detach_target = detach_target

    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        loss, cosine, frame_count = _masked_cosine_alignment_loss(
            prediction,
            target,
            mask,
            detach_target=self.detach_target,
        )
        emit_diagnostics(
            self,
            {
                "cosine": cosine,
                "valid_frames": frame_count,
            },
        )
        return loss


@loss_with_diagnostics
class TeacherLoss(LossABC):
    """Teacher feature distillation with one shared frame-count reduction."""

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

    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        loss, cosine, smooth_l1, frame_count = _teacher_loss(
            prediction,
            target,
            mask,
            smooth_l1_weight=self.smooth_l1_weight,
            detach_target=self.detach_target,
            beta=self.beta,
        )
        emit_diagnostics(
            self,
            {
                "cosine": cosine,
                "smooth_l1": smooth_l1,
                "valid_frames": frame_count,
            },
        )
        return loss


def _masked_frame_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_frame_tensors(prediction, target, mask, name="frame MSE")
    frame_mask = mask[..., None]
    safe_prediction = prediction.masked_fill(~frame_mask, 0)
    safe_target = target.masked_fill(~frame_mask, 0)
    frame_loss = F.mse_loss(safe_prediction, safe_target, reduction="none").mean(dim=-1)
    return _masked_mean(frame_loss, mask)


def _masked_cosine_alignment_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detach_target: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_frame_tensors(prediction, target, mask, name="cosine alignment")
    safe_prediction, safe_target = _prepared_pair(
        prediction,
        target,
        mask,
        detach_target=detach_target,
    )
    frame_loss = _cosine_distance(safe_prediction, safe_target).masked_fill(~mask, 0)
    loss, frame_count = _masked_mean(frame_loss, mask)
    return loss, 1 - loss, frame_count


def _teacher_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    smooth_l1_weight: float,
    detach_target: bool,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_frame_tensors(prediction, target, mask, name="teacher")
    safe_prediction, safe_target = _prepared_pair(
        prediction,
        target,
        mask,
        detach_target=detach_target,
    )
    cosine_distance = _cosine_distance(safe_prediction, safe_target).masked_fill(~mask, 0)
    smooth_l1_frames = (
        F.smooth_l1_loss(safe_prediction, safe_target, reduction="none", beta=beta)
        .mean(dim=-1)
        .masked_fill(~mask, 0)
    )
    cosine_loss, frame_count = _masked_mean(cosine_distance, mask)
    smooth_l1, _frame_count = _masked_mean(smooth_l1_frames, mask)
    return (
        cosine_loss + smooth_l1_weight * smooth_l1,
        1 - cosine_loss,
        smooth_l1,
        frame_count,
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
) -> tuple[Tensor, Tensor]:
    frame_mask = mask[..., None]
    safe_prediction = prediction.masked_fill(~frame_mask, 0).float()
    safe_target = target.to(device=prediction.device)
    if detach_target:
        safe_target = safe_target.detach()
    safe_target = safe_target.masked_fill(~frame_mask, 0).float()
    return safe_prediction, safe_target


def _cosine_distance(prediction: Tensor, target: Tensor) -> Tensor:
    normalized_prediction = F.normalize(prediction, dim=-1)
    normalized_target = F.normalize(target, dim=-1)
    return 1 - (normalized_prediction * normalized_target).sum(dim=-1)


def _masked_mean(value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    frame_count = mask.sum()
    if frame_count.item() == 0:
        raise ValueError("loss mask must select at least one frame.")
    count = frame_count.to(dtype=value.dtype)
    return value.sum() / count, count


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
