from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import auto
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from anytrain._buffer import register_buffer
from anytrain._compat import StrEnum

from . import _balancer_checkpoint as checkpoint
from ._balancer_contract import stack_named_losses, validate_loss_names
from ._validation import scalar
from .observation import (
    dynamic_balancer,
    fixed_balancer,
    interpolated_balancer,
    recommend,
)

LossMapping = Mapping[str, Tensor]


class UncertaintyNormalization(StrEnum):
    NONE = auto()
    DETACHED_SUM = auto()


class LossBalancerABC(nn.Module, ABC):
    def forward(self, losses: LossMapping) -> Tensor:
        return scalar(self.compute_loss(losses), name="balanced loss")

    @abstractmethod
    def compute_loss(self, losses: LossMapping) -> Tensor:
        raise NotImplementedError


class WeightedLossBalancerABC(LossBalancerABC):
    """Named scalar balancer with an inspectable effective weight vector.

    Custom implementations must call ``super().__init__(loss_names)`` and
    implement ``weights`` plus ``compute_loss``. Static behavior that affects
    checkpoint interpretation belongs in ``_checkpoint_config()``.
    """

    loss_names: tuple[str, ...]

    def __init__(self, loss_names: list[str]) -> None:
        super().__init__()
        self.loss_names = validate_loss_names(loss_names, label="loss_names")
        self._loss_name_set = frozenset(self.loss_names)

    @property
    @abstractmethod
    def weights(self) -> Tensor:
        raise NotImplementedError

    def _stack_losses(self, losses: LossMapping) -> Tensor:
        return stack_named_losses(losses, self.loss_names, self._loss_name_set)

    def _checkpoint_config(self) -> dict[str, Any]:
        return {}

    def get_extra_state(self) -> dict[str, Any]:
        return checkpoint.extra_state(self.loss_names, self._checkpoint_config())

    def set_extra_state(self, state: object) -> None:
        checkpoint.validate_extra_state(
            state,
            loss_names=self.loss_names,
            config=self._checkpoint_config(),
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        blocked_key = f"{prefix}{checkpoint.BLOCKED_LOAD_KEY}"
        if state_dict.get(blocked_key) is checkpoint.BLOCKED_LOAD:
            state_dict.pop(blocked_key)
            self._block_checkpoint_descendants(state_dict, prefix)
            return
        try:
            self._validate_checkpoint_state(state_dict, prefix)
        except RuntimeError as error:
            error_msgs.append(str(error))
            self._block_checkpoint_descendants(state_dict, prefix)
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_checkpoint_state(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
    ) -> None:
        checkpoint.validate_checkpoint_state(
            self,
            state_dict,
            prefix,
            loss_names=self.loss_names,
            config=self._checkpoint_config(),
        )

    def _block_checkpoint_descendants(
        self,
        state_dict: dict[str, Any],
        prefix: str,
    ) -> None:
        del state_dict, prefix

class MeanLossBalancer(LossBalancerABC):
    def compute_loss(self, losses: LossMapping) -> Tensor:
        if not isinstance(losses, Mapping):
            raise TypeError("losses must be a mapping of names to scalar tensors.")
        if not losses:
            raise ValueError("losses must contain at least one scalar tensor.")
        return torch.stack(
            [scalar(value, name=f"loss {name!r}") for name, value in losses.items()]
        ).mean()


@fixed_balancer
class FixedWeightLossBalancer(WeightedLossBalancerABC):
    _weights: Tensor

    def __init__(
        self,
        loss_weights: Mapping[str, float],
        *,
        normalize_weights: bool = False,
    ) -> None:
        if not isinstance(loss_weights, Mapping):
            raise TypeError("loss_weights must be a mapping of names to real numbers.")
        if not loss_weights:
            raise ValueError("loss_weights must contain at least one name.")
        if not isinstance(normalize_weights, bool):
            raise TypeError("normalize_weights must be a bool.")
        super().__init__(list(loss_weights))
        weights = torch.tensor(
            [_finite_weight(loss_weights[name], name=name) for name in self.loss_names]
        )
        if not bool(torch.isfinite(weights).all()):
            raise ValueError(
                "loss_weights must remain finite in the default torch floating dtype."
            )
        if normalize_weights:
            weight_sum = weights.sum()
            if not bool(torch.isfinite(weight_sum)):
                raise ValueError(
                    "loss_weights must have finite sum when normalize_weights=True."
                )
            if weight_sum <= 0:
                raise ValueError(
                    "loss_weights must have positive sum when normalize_weights=True."
                )
            weights = weights / weight_sum
        register_buffer(self, "_weights", weights)
        recommend(self)

    @property
    def weights(self) -> Tensor:
        return self._weights

    def compute_loss(self, losses: LossMapping) -> Tensor:
        loss_values = self._stack_losses(losses)
        if self.weights.device != loss_values.device:
            raise RuntimeError(
                "loss balancer weights and losses must be on the same device; "
                "move the containing module before calling it."
            )
        return (loss_values * self.weights).sum()


def _finite_weight(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"loss weight {name!r} must be a real number.")
    weight = float(value)
    if not math.isfinite(weight):
        raise ValueError(f"loss weight {name!r} must be finite.")
    return weight


@dynamic_balancer
class UncertaintyLossBalancer(WeightedLossBalancerABC):
    def __init__(
        self,
        loss_names: list[str],
        *,
        normalization: UncertaintyNormalization | str = UncertaintyNormalization.DETACHED_SUM,
    ) -> None:
        if not isinstance(normalization, str):
            raise TypeError("normalization must be an UncertaintyNormalization or string.")
        try:
            mode = UncertaintyNormalization(normalization)
        except ValueError as error:
            supported = ", ".join(item.value for item in UncertaintyNormalization)
            raise ValueError(
                f"Unknown uncertainty normalization {normalization!r}. "
                f"Supported values: {supported}."
            ) from error
        super().__init__(loss_names)
        self.normalization = mode
        self.log_var = nn.Parameter(torch.zeros(len(self.loss_names)))
        recommend(self)

    @property
    def _raw_weights(self) -> Tensor:
        return torch.exp(-self.log_var)

    @property
    def weights(self) -> Tensor:
        raw_weights = self._raw_weights
        if self.normalization == UncertaintyNormalization.NONE:
            return raw_weights
        return raw_weights / raw_weights.detach().sum()

    def compute_loss(self, losses: LossMapping) -> Tensor:
        loss_values = self._stack_losses(losses)
        if self.log_var.device != loss_values.device:
            raise RuntimeError(
                "loss balancer log_var and losses must be on the same device; "
                "move the containing module before calling it."
            )
        raw_weights = self._raw_weights
        objective = (loss_values * raw_weights).sum() + self.log_var.sum()
        if self.normalization == UncertaintyNormalization.NONE:
            return objective

        # The detached factor keeps model-gradient weights normalized while
        # preserving the canonical uncertainty-gradient direction up to one
        # shared scale. It is intentionally a surrogate gradient.
        detached_weight_sum = raw_weights.detach().sum()
        return objective / detached_weight_sum

    def _checkpoint_config(self) -> dict[str, Any]:
        return {"normalization": self.normalization.value}


@interpolated_balancer
class InterpolatedLossBalancer(WeightedLossBalancerABC):
    blend: Tensor

    def __init__(
        self,
        *,
        fixed: WeightedLossBalancerABC,
        dynamic: WeightedLossBalancerABC,
        blend: float = 0.0,
    ) -> None:
        if not isinstance(fixed, WeightedLossBalancerABC):
            raise TypeError("fixed must inherit WeightedLossBalancerABC.")
        if not isinstance(dynamic, WeightedLossBalancerABC):
            raise TypeError("dynamic must inherit WeightedLossBalancerABC.")
        if fixed.loss_names != dynamic.loss_names:
            raise ValueError(
                "fixed and dynamic loss_names must match in the same order: "
                f"fixed={fixed.loss_names!r}, dynamic={dynamic.loss_names!r}."
            )
        with torch.no_grad():
            fixed_weights = fixed.weights
            dynamic_weights = dynamic.weights
        blend_tensor = fixed_weights.new_tensor(_blend(blend))
        _validate_interpolation(
            fixed_weights,
            dynamic_weights,
            blend_tensor,
            name="loss balancer weights",
        )
        super().__init__(list(fixed.loss_names))
        self.fixed = fixed
        self.dynamic = dynamic
        register_buffer(self, "blend", blend_tensor)
        recommend(self)

    def _validate_checkpoint_state(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
    ) -> None:
        extra_state_key = f"{prefix}_extra_state"
        if extra_state_key not in state_dict:
            raise RuntimeError(
                f"{type(self).__qualname__} checkpoint is missing loss-name metadata; "
                "it cannot be safely aligned with configured losses."
            )
        checkpoint.validate_extra_state(
            state_dict[extra_state_key],
            loss_names=self.loss_names,
            config=self._checkpoint_config(),
        )
        blend_key = f"{prefix}blend"
        if blend_key not in state_dict:
            raise RuntimeError(
                f"{type(self).__qualname__} checkpoint is missing blend state."
            )
        checkpoint.validate_blend(state_dict[blend_key], parse=_blend)
        self.fixed._validate_checkpoint_state(state_dict, f"{prefix}fixed.")
        self.dynamic._validate_checkpoint_state(
            state_dict,
            f"{prefix}dynamic.",
        )
        super()._validate_checkpoint_state(state_dict, prefix)

    def _block_checkpoint_descendants(
        self,
        state_dict: dict[str, Any],
        prefix: str,
    ) -> None:
        for name in ("fixed", "dynamic"):
            child_prefix = f"{prefix}{name}."
            for key in tuple(state_dict):
                if key.startswith(child_prefix):
                    state_dict.pop(key)
            state_dict[f"{child_prefix}{checkpoint.BLOCKED_LOAD_KEY}"] = (
                checkpoint.BLOCKED_LOAD
            )

    @property
    def weights(self) -> Tensor:
        return _interpolate(
            self.fixed.weights,
            self.dynamic.weights,
            self.blend,
            name="loss balancer weights",
        )

    def set_blend(self, blend: float) -> None:
        self.blend.fill_(_blend(blend))

    def compute_loss(self, losses: LossMapping) -> Tensor:
        fixed_objective = self.fixed(losses)
        dynamic_objective = self.dynamic(losses)
        return _interpolate(
            fixed_objective,
            dynamic_objective,
            self.blend,
            name="loss balancer objectives",
        )

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "fixed_type": _type_name(self.fixed),
            "fixed_config": self.fixed._checkpoint_config(),
            "dynamic_type": _type_name(self.dynamic),
            "dynamic_config": self.dynamic._checkpoint_config(),
        }


def _blend(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("blend must be a host real number, not bool or Tensor.")
    blend = float(value)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be finite and between 0 and 1.")
    return blend


def _interpolate(start: Tensor, end: Tensor, blend: Tensor, *, name: str) -> Tensor:
    _validate_interpolation(start, end, blend, name=name)
    return torch.lerp(start, end, blend)


def _validate_interpolation(
    start: Tensor,
    end: Tensor,
    blend: Tensor,
    *,
    name: str,
) -> None:
    if start.shape != end.shape:
        raise RuntimeError(
            f"{name} must have matching shapes, got {tuple(start.shape)} and "
            f"{tuple(end.shape)}."
        )
    if start.device != end.device or start.device != blend.device:
        raise RuntimeError(f"{name} and blend must be on the same device.")
    if start.dtype != end.dtype:
        raise RuntimeError(f"{name} must use the same dtype.")


def _type_name(module: nn.Module) -> str:
    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__qualname__}"
