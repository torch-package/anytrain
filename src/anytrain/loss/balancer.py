from __future__ import annotations

import math
from abc import ABC
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from anytrain._buffer import register_buffer

from ._observation import fixed_weights, recommend, uncertainty_weights
from ._validation import scalar

LossTensorDict = Mapping[str, Tensor]


def _validate_ordered_loss_names(loss_names: list[str], *, label: str) -> list[str]:
    if not isinstance(loss_names, list):
        raise TypeError(f"{label} must be a list of strings.")
    if not loss_names:
        raise ValueError(f"{label} must contain at least one name.")

    validated: list[str] = []
    source_by_observation_name: dict[str, str] = {}
    for name in loss_names:
        if not isinstance(name, str):
            raise TypeError("loss name must be a string.")
        if not name:
            raise ValueError("loss name must not be empty.")
        if name in validated:
            raise ValueError(f"Duplicate loss name {name!r}.")
        observation_name = name.strip("/")
        if not observation_name:
            raise ValueError(
                f"Loss name {name!r} must produce a non-empty observation tag."
            )
        previous = source_by_observation_name.get(observation_name)
        if previous is not None:
            raise ValueError(
                f"Loss names {previous!r} and {name!r} normalize to the same "
                f"observation tag {observation_name!r}."
            )
        source_by_observation_name[observation_name] = name
        validated.append(name)
    return validated


def _stack_named_losses(losses: LossTensorDict, loss_names: list[str]) -> Tensor:
    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a mapping of names to scalar tensors.")
    missing = [name for name in loss_names if name not in losses]
    if missing:
        raise ValueError(f"losses missing configured loss names: {missing}.")
    extra = [name for name in losses if name not in loss_names]
    if extra:
        raise ValueError(f"losses contains unknown loss names: {extra}.")
    return torch.stack(
        [scalar(losses[name], name=f"loss {name!r}") for name in loss_names]
    )


class LossBalancerABC(nn.Module, ABC):
    def forward(self, losses: LossTensorDict) -> Tensor:
        return scalar(self.compute_loss(losses), name="balanced loss")

    def compute_loss(self, losses: LossTensorDict) -> Tensor:
        raise NotImplementedError


class MeanLossBalancer(LossBalancerABC):
    def compute_loss(self, losses: LossTensorDict) -> Tensor:
        if not isinstance(losses, Mapping):
            raise TypeError("losses must be a mapping of names to scalar tensors.")
        if not losses:
            raise ValueError("losses must contain at least one scalar tensor.")
        return torch.stack(
            [scalar(value, name=f"loss {name!r}") for name, value in losses.items()]
        ).mean()


@fixed_weights
class FixedWeightLossBalancer(LossBalancerABC):
    weights: Tensor

    def __init__(
        self,
        loss_weights: Mapping[str, float],
        *,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.loss_names = _validate_ordered_loss_names(list(loss_weights), label="loss_weights")
        weights = torch.tensor(
            [_finite_weight(loss_weights[name], name=name) for name in self.loss_names]
        )
        if normalize:
            weight_sum = weights.sum()
            if weight_sum <= 0:
                raise ValueError("loss_weights must have positive sum when normalize=True.")
            weights = weights / weight_sum
        register_buffer(self, "weights", weights)
        recommend(self)

    def compute_loss(self, losses: LossTensorDict) -> Tensor:
        loss_tensor = self._stack_losses(losses)
        weights = self.weights.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
        return (loss_tensor * weights).sum()

    def _stack_losses(self, losses: LossTensorDict) -> Tensor:
        return _stack_named_losses(losses, self.loss_names)


def _finite_weight(value: float, *, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"loss weight {name!r} must be a real number.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"loss weight {name!r} must be finite.")
    return resolved


@uncertainty_weights
class UncertaintyLossBalancer(LossBalancerABC):
    def __init__(
        self,
        loss_names: list[str],
        *,
        normalized: bool = True,
    ) -> None:
        super().__init__()
        self.loss_names = _validate_ordered_loss_names(loss_names, label="loss_names")
        self.normalized = normalized
        self.log_var = nn.Parameter(torch.zeros(len(self.loss_names)))
        recommend(self)

    @property
    def _weight(self) -> Tensor:
        return torch.exp(-self.log_var)

    @property
    def weight(self) -> Tensor:
        if self.normalized:
            return self._weight / self._normalize_factor
        return self._weight

    @property
    def _normalize_factor(self) -> Tensor:
        with torch.no_grad():
            return self._weight.detach().sum()

    @property
    def _sum_log_var(self) -> Tensor:
        if self.normalized:
            return self.log_var.sum() / self._normalize_factor
        return self.log_var.sum()

    def compute_loss(self, losses: LossTensorDict) -> Tensor:
        loss_tensor = self._stack_losses(losses)
        return (loss_tensor * self.weight).sum() + self._sum_log_var

    def _stack_losses(self, losses: LossTensorDict) -> Tensor:
        return _stack_named_losses(losses, self.loss_names)
