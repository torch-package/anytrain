from __future__ import annotations

import torch
from torch import Tensor

from .._observation import loss as register_loss
from ..abc import LossABC


def _sum_channels_and_time(signal: Tensor) -> Tensor:
    if signal.ndim < 2:
        raise ValueError("audio tensors must include a time dimension.")
    if signal.ndim == 2:
        # Mono layout [batch, time].
        return signal.square().sum(dim=-1)
    return signal.square().sum(dim=(-2, -1))


def global_sdr(estimate: Tensor, reference: Tensor, *, eps: float = 1e-8) -> Tensor:
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape.")
    if estimate.ndim < 2:
        raise ValueError("estimate and reference must include batch and time dimensions.")

    error = estimate - reference
    numerator = _sum_channels_and_time(reference) + eps
    denominator = _sum_channels_and_time(error) + eps
    return 10 * torch.log10(numerator / denominator)


def scale_invariant_signal(
    estimate: Tensor,
    reference: Tensor,
    *,
    eps: float = 1e-8,
    min_energy: float = 1e-12,
    fallback_scale: float = 1.0,
) -> Tensor:
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape.")
    if estimate.ndim < 2:
        raise ValueError("estimate and reference must include batch and time dimensions.")

    if estimate.ndim == 2:
        reduce_dims: tuple[int, ...] = (-1,)
    else:
        reduce_dims = (-2, -1)

    reference_energy = (reference * reference).sum(dim=reduce_dims)
    estimate_energy = (estimate * estimate).sum(dim=reduce_dims)
    projection = (estimate * reference).sum(dim=reduce_dims)
    scale = projection / (reference_energy + eps)
    fallback = torch.full_like(scale, fallback_scale)
    scale = torch.where(estimate_energy < min_energy, fallback, scale)
    while scale.ndim < estimate.ndim:
        scale = scale.unsqueeze(-1)
    return scale * reference


def si_snr(estimate: Tensor, reference: Tensor, *, eps: float = 1e-8) -> Tensor:
    target = scale_invariant_signal(estimate, reference, eps=eps)
    return global_sdr(estimate, target, eps=eps)


@register_loss
class SDRLoss(LossABC):
    def __init__(
        self,
        *,
        scale_invariant: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.scale_invariant = scale_invariant
        self.eps = eps

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        score = (
            si_snr(input, target, eps=self.eps)
            if self.scale_invariant
            else global_sdr(input, target, eps=self.eps)
        )
        return -score.mean()
