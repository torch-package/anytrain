from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anytrain import observation

from ._deps import MixturePathGeneralizedKL
from .continuous import ContinuousFlowRuntime
from .discrete import DiscreteFlowRuntime
from .types import FlowLossFn, ModelExtras


def mse_velocity_loss(
    prediction: Tensor,
    target: Tensor,
    extras: ModelExtras,
) -> Tensor:
    del extras
    return F.mse_loss(prediction, target)


def masked_mse_velocity_loss(
    prediction: Tensor,
    target: Tensor,
    extras: ModelExtras,
) -> Tensor:
    mask = extras.get("mask")
    if mask is None:
        return mse_velocity_loss(prediction, target, extras)
    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a tensor.")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean.")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if mask.ndim >= prediction.ndim or prediction.shape[: mask.ndim] != mask.shape:
        raise ValueError("mask must align with a leading prefix of prediction and target.")
    weights = mask.to(device=prediction.device, dtype=prediction.dtype)
    for _ in range(prediction.ndim - mask.ndim):
        weights = weights.unsqueeze(-1)
    unmasked_width = math.prod(prediction.shape[mask.ndim :])
    denominator = weights.sum() * unmasked_width
    if not bool(denominator > 0):
        raise ValueError("mask must contain at least one valid item.")
    return ((prediction - target).square() * weights).sum() / denominator


def _require_scalar_loss(loss: Tensor) -> Tensor:
    if loss.ndim != 0:
        raise ValueError("flow matching loss_fn must return a scalar tensor.")
    return loss


class ContinuousVelocityObjective(nn.Module):
    def __init__(
        self,
        runtime: ContinuousFlowRuntime,
        *,
        loss_fn: FlowLossFn = mse_velocity_loss,
    ):
        super().__init__()
        self.runtime = runtime
        self.loss_fn = loss_fn
        observation.registry.recommend(self)

    def forward(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        sample = self.runtime.training_sample(x_1, x_0=x_0)
        prediction = self.runtime.call_model(model, sample.x_t, sample.t, model_extras)
        return _require_scalar_loss(self.loss_fn(prediction, sample.velocity, model_extras))


class DiscreteGeneralizedKLObjective(nn.Module):
    def __init__(
        self,
        runtime: DiscreteFlowRuntime,
    ):
        super().__init__()
        self.runtime = runtime
        self.loss_fn = MixturePathGeneralizedKL(runtime.path)
        observation.registry.recommend(self)

    def forward(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        sample = self.runtime.training_sample(x_1, x_0=x_0)
        logits = self.runtime.call_model(model, sample.x_t, sample.t, model_extras)
        return _require_scalar_loss(
            self.loss_fn(
                logits=logits,
                x_1=x_1,
                x_t=sample.x_t,
                t=sample.t,
            )
        )


def _collect_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> observation.Curve:
    del module, inputs, kwargs
    if not isinstance(output, Tensor):
        raise TypeError("flow matching loss observation requires a Tensor module output.")
    return observation.Curve(output)


for _objective_type in (ContinuousVelocityObjective, DiscreteGeneralizedKLObjective):
    observation.registry.register(
        _objective_type,
        (
            observation.OutputObservation(
                "loss",
                _collect_loss,
                reduction=observation.Reduction.Mean,
                recommended=True,
            ),
        ),
    )


__all__ = [
    "ContinuousVelocityObjective",
    "DiscreteGeneralizedKLObjective",
    "masked_mse_velocity_loss",
    "mse_velocity_loss",
]
