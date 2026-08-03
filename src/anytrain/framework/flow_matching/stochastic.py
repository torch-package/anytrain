from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

LogProbReduction = Literal["mean", "sum"]


@dataclass(eq=False)
class RectifiedFlowSDETransition:
    next_x: Tensor
    mean: Tensor
    noise_std: Tensor
    step_std: Tensor
    dt: Tensor
    log_prob: Tensor


class RectifiedFlowSDE:
    """Stochastic reverse step for Flow-GRPO style rectified-flow rollouts.

    Use this helper when a downstream sampler already has a velocity prediction
    and needs a Gaussian transition log-prob for online RL updates. The time
    convention follows reverse denoising with 0 < next_t < t < 1.

    Reference: Liu et al., "Flow-GRPO: Training Flow Matching Models via Online
    RL", arXiv:2505.05470, Section 4 ODE-to-SDE derivation.
    """

    def __init__(
        self,
        *,
        noise_level: float = 0.7,
        eps: float = 1e-5,
        log_prob_reduction: LogProbReduction = "mean",
    ) -> None:
        if not math.isfinite(noise_level) or noise_level <= 0:
            raise ValueError("noise_level must be finite and positive.")
        if not math.isfinite(eps) or eps <= 0 or eps >= 0.5:
            raise ValueError("eps must be in (0, 0.5).")
        if log_prob_reduction not in {"mean", "sum"}:
            raise ValueError("log_prob_reduction must be 'mean' or 'sum'.")
        self.noise_level = float(noise_level)
        self.eps = float(eps)
        self.log_prob_reduction: LogProbReduction = log_prob_reduction

    def step_with_log_prob(
        self,
        x_t: Tensor,
        t: Tensor,
        next_t: Tensor,
        velocity: Tensor,
        *,
        next_x: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> RectifiedFlowSDETransition:
        _validate_state_and_velocity(x_t, velocity)
        t = _batch_time(t, x_t, name="t")
        next_t = _batch_time(next_t, x_t, name="next_t")
        if not bool(torch.isfinite(t).all() and torch.isfinite(next_t).all()):
            raise ValueError("t and next_t must be finite.")
        if bool((t <= 0).any() or (t >= 1).any() or (next_t < 0).any()):
            raise ValueError("t must be in (0, 1) and next_t must be non-negative.")
        if bool((next_t >= t).any()):
            raise ValueError("next_t must be strictly less than t for reverse denoising.")

        t_view = _broadcast_time(t, x_t)
        next_t_view = _broadcast_time(next_t, x_t)
        dt = next_t_view - t_view
        noise_std = self.noise_std(t_view)
        mean = (
            x_t * (1.0 + noise_std.square() / (2.0 * t_view) * dt)
            + velocity * (1.0 + noise_std.square() * (1.0 - t_view) / (2.0 * t_view)) * dt
        )
        step_std = noise_std * torch.sqrt(-dt)

        if next_x is None:
            noise = torch.randn(
                mean.shape,
                generator=generator,
                device=mean.device,
                dtype=mean.dtype,
            )
            next_x = mean + step_std * noise
        else:
            _validate_next_x(next_x, mean)
        log_prob = gaussian_log_prob(
            next_x.detach(),
            mean,
            step_std,
            reduction=self.log_prob_reduction,
        )
        return RectifiedFlowSDETransition(
            next_x=next_x,
            mean=mean,
            noise_std=noise_std,
            step_std=step_std,
            dt=dt,
            log_prob=log_prob,
        )

    def noise_std(self, t: Tensor) -> Tensor:
        """Return sigma_t = noise_level * sqrt(t / (1 - t))."""

        if not isinstance(t, Tensor):
            raise TypeError("t must be a tensor.")
        if t.is_complex() or t.dtype == torch.bool:
            raise TypeError("t must be real-valued.")
        t = t.clamp(self.eps, 1.0 - self.eps)
        return t.new_tensor(self.noise_level) * torch.sqrt(t / (1.0 - t))


def gaussian_log_prob(
    value: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    reduction: LogProbReduction = "mean",
) -> Tensor:
    """Log-probability of a diagonal Gaussian reduced over non-batch axes."""

    _validate_gaussian_inputs(value, mean, std)
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")
    log_prob = (
        -0.5 * ((value - mean) / std).square()
        - torch.log(std)
        - 0.5 * mean.new_tensor(math.log(2.0 * math.pi))
    )
    dims = tuple(range(1, log_prob.ndim))
    if not dims:
        return log_prob
    if reduction == "mean":
        return log_prob.mean(dim=dims)
    return log_prob.sum(dim=dims)


def gaussian_kl_from_means(
    policy_mean: Tensor,
    reference_mean: Tensor,
    std: Tensor,
    *,
    reduction: LogProbReduction = "mean",
) -> Tensor:
    """Closed-form KL for equal-variance diagonal Gaussians.

    Pass the same standard deviation used for the transition distribution unless
    a downstream sampler intentionally follows a paper-specific convention.
    """

    _validate_gaussian_inputs(policy_mean, reference_mean, std)
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")
    kl = (policy_mean - reference_mean).square() / (2.0 * std.square())
    dims = tuple(range(1, kl.ndim))
    if not dims:
        return kl
    if reduction == "mean":
        return kl.mean(dim=dims)
    return kl.sum(dim=dims)


def _validate_state_and_velocity(x_t: Tensor, velocity: Tensor) -> None:
    if not isinstance(x_t, Tensor) or not isinstance(velocity, Tensor):
        raise TypeError("x_t and velocity must be tensors.")
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have the same shape.")
    if x_t.ndim < 1:
        raise ValueError("x_t must include a batch dimension.")
    if x_t.device != velocity.device:
        raise ValueError("x_t and velocity must be on the same device.")
    if x_t.dtype != velocity.dtype:
        raise TypeError("x_t and velocity must have the same dtype.")
    if x_t.is_complex() or velocity.is_complex():
        raise TypeError("x_t and velocity must be real-valued.")


def _validate_next_x(next_x: Tensor, mean: Tensor) -> None:
    if not isinstance(next_x, Tensor):
        raise TypeError("next_x must be a tensor.")
    if next_x.shape != mean.shape:
        raise ValueError("next_x must have the same shape as x_t.")
    if next_x.device != mean.device:
        raise ValueError("next_x must be on the same device as x_t.")
    if next_x.dtype != mean.dtype:
        raise TypeError("next_x must have the same dtype as x_t.")


def _validate_gaussian_inputs(value: Tensor, mean: Tensor, std: Tensor) -> None:
    if not isinstance(value, Tensor) or not isinstance(mean, Tensor) or not isinstance(std, Tensor):
        raise TypeError("value, mean, and std must be tensors.")
    if value.shape != mean.shape:
        raise ValueError("value and mean must have the same shape.")
    if value.device != mean.device or std.device != mean.device:
        raise ValueError("value, mean, and std must be on the same device.")
    if value.dtype != mean.dtype or std.dtype != mean.dtype:
        raise TypeError("value, mean, and std must have the same dtype.")
    if value.is_complex() or mean.is_complex() or std.is_complex():
        raise TypeError("value, mean, and std must be real-valued.")
    if bool((std <= 0).any()):
        raise ValueError("std must be positive.")
    if not bool(torch.isfinite(value).all() and torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("value, mean, and std must be finite.")


def _batch_time(t: Tensor, x_t: Tensor, *, name: str) -> Tensor:
    if not isinstance(t, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if t.is_complex() or t.dtype == torch.bool:
        raise TypeError(f"{name} must be real-valued.")
    if t.device != x_t.device:
        raise ValueError(f"{name} must be on the same device as x_t.")
    if t.ndim == 0:
        return t.to(dtype=x_t.dtype).repeat(x_t.shape[0])
    if t.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D tensor.")
    if t.numel() == 1:
        return t.to(dtype=x_t.dtype).repeat(x_t.shape[0])
    if t.numel() != x_t.shape[0]:
        raise ValueError(f"{name} must have shape [batch] or be scalar.")
    return t.to(dtype=x_t.dtype)


def _broadcast_time(t: Tensor, x_t: Tensor) -> Tensor:
    return t.view(t.shape[0], *([1] * (x_t.ndim - 1)))


__all__ = [
    "LogProbReduction",
    "RectifiedFlowSDE",
    "RectifiedFlowSDETransition",
    "gaussian_kl_from_means",
    "gaussian_log_prob",
]
