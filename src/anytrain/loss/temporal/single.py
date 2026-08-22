from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor

from ..abc import LossABC
from ..observation import loss as register_loss


def _sum_channels_and_time(signal: Tensor) -> Tensor:
    if signal.ndim < 2:
        raise ValueError("audio tensors must include a time dimension.")
    if signal.ndim == 2:
        # Mono layout [batch, time].
        return signal.square().sum(dim=-1)
    return signal.square().sum(dim=(-2, -1))


def _validate_audio_pair(estimate: Tensor, reference: Tensor) -> None:
    if not isinstance(estimate, Tensor) or not isinstance(reference, Tensor):
        raise TypeError("estimate and reference must be tensors.")
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape.")
    if estimate.ndim < 2:
        raise ValueError("estimate and reference must include batch and time dimensions.")


def _finite_parameter(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def global_sdr(estimate: Tensor, reference: Tensor, *, eps: float = 1e-8) -> Tensor:
    _validate_audio_pair(estimate, reference)
    eps = _finite_parameter(eps, name="eps", positive=True)

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
    _validate_audio_pair(estimate, reference)
    eps = _finite_parameter(eps, name="eps", positive=True)
    min_energy = _finite_parameter(min_energy, name="min_energy", nonnegative=True)
    fallback_scale = _finite_parameter(fallback_scale, name="fallback_scale")

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
        if not isinstance(scale_invariant, bool):
            raise TypeError("scale_invariant must be a bool.")
        self.scale_invariant = scale_invariant
        self.eps = _finite_parameter(eps, name="eps", positive=True)

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        score = (
            si_snr(input, target, eps=self.eps)
            if self.scale_invariant
            else global_sdr(input, target, eps=self.eps)
        )
        return -score.mean()
