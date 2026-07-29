from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from .abc import LossABC, LossDetails
from .balancer import LossBalancerABC, MeanLossBalancer


@runtime_checkable
class _SpectralCacheAware(Protocol):
    def _forward_with_spectral_cache(
        self,
        *args: Any,
        spectral_cache: Any,
        **kwargs: Any,
    ) -> Any: ...


class LossGroup(LossABC):
    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        balancer: LossBalancerABC | None = None,
    ) -> None:
        super().__init__()
        validated_losses = self._validate_losses(losses)
        self.losses = cast(dict[str, nn.Module], nn.ModuleDict(validated_losses))
        self.balancer = self._validate_balancer(balancer)

    def compute_loss(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, LossDetails]:
        spectral_cache = kwargs.pop("_spectral_cache", None)
        loss_values: dict[str, torch.Tensor] = {}
        details: LossDetails = {}
        for name, module in self.losses.items():
            loss, module_details = self._split_loss_result(
                self._call_loss(
                    module,
                    *args,
                    spectral_cache=spectral_cache,
                    **kwargs,
                )
            )
            loss_values[name] = loss
            self._add_detail(details, name, loss)
            for detail_name, detail_value in module_details.items():
                self._add_prefixed_detail(
                    details,
                    name,
                    detail_name,
                    detail_value,
                )

        if not loss_values:
            raise RuntimeError("LossGroup has no loss modules.")
        total, balancer_details = self._split_loss_result(self.balancer(loss_values))
        for detail_name, detail_value in balancer_details.items():
            self._add_prefixed_detail(
                details,
                "balancer",
                detail_name,
                detail_value,
            )
        return total, details

    def _call_loss(
        self,
        module: nn.Module,
        *args: Any,
        spectral_cache: Any,
        **kwargs: Any,
    ) -> Any:
        if spectral_cache is not None and isinstance(module, _SpectralCacheAware):
            return module._forward_with_spectral_cache(
                *args,
                spectral_cache=spectral_cache,
                **kwargs,
            )
        return module(*args, **kwargs)

    def _validate_losses(self, losses: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
        if not isinstance(losses, Mapping):
            raise TypeError("losses must be a mapping of names to loss modules.")
        if not losses:
            raise ValueError("losses must contain at least one loss.")

        validated: dict[str, nn.Module] = {}
        for raw_name, loss in losses.items():
            name = self._validate_loss_name(raw_name)
            if not isinstance(loss, nn.Module):
                raise TypeError(f"Loss {name!r} must be a torch.nn.Module.")
            validated[name] = loss
        return validated

    def _validate_balancer(self, balancer: LossBalancerABC | None) -> LossBalancerABC:
        if balancer is None:
            return MeanLossBalancer()
        if not isinstance(balancer, LossBalancerABC):
            raise TypeError("balancer must inherit LossBalancerABC.")
        return balancer

    def _validate_loss_name(self, name: object) -> str:
        name = self._validate_detail_name(name)
        if name == "balancer":
            raise ValueError(f"loss name {name!r} is reserved.")
        return name

    def _split_loss_result(self, result: object) -> tuple[torch.Tensor, LossDetails]:
        if isinstance(result, torch.Tensor):
            return self._validate_loss_tensor(result), {}

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("loss result must be a scalar tensor or (scalar tensor, details).")
        loss, details = result
        return self._validate_loss_tensor(loss), self._validate_nested_loss_details(
            details,
            detach=True,
        )

    def _add_detail(self, details: LossDetails, name: str, value: object) -> None:
        name = self._validate_detail_name(name)
        self._set_detail(details, name, value)

    def _add_prefixed_detail(
        self,
        details: LossDetails,
        loss_name: str,
        detail_name: str,
        value: object,
    ) -> None:
        name = self._join_detail_name(loss_name, detail_name)
        self._set_detail(details, name, value)

    def _set_detail(self, details: LossDetails, name: str, value: object) -> None:
        if name in details:
            raise ValueError(f"Duplicate loss detail key {name!r}.")
        details[name] = self._validate_detail_value(value, name=name, detach=True)

    def _join_detail_name(self, loss_name: str, detail_name: str) -> str:
        return self.detail_key_separator.join(
            [
                self._validate_detail_name(loss_name),
                self._validate_detail_path(detail_name),
            ]
        )

    def _validate_nested_loss_details(
        self,
        details: object,
        *,
        detach: bool,
    ) -> LossDetails:
        if not isinstance(details, Mapping):
            raise TypeError("loss details must be a mapping of string keys to detail values.")

        validated: LossDetails = {}
        for raw_name, raw_value in details.items():
            name = self._validate_detail_path(raw_name)
            validated[name] = self._validate_detail_value(raw_value, name=name, detach=detach)
        return validated

    def _validate_loss_details(
        self,
        details: object,
        *,
        detach: bool,
    ) -> LossDetails:
        return self._validate_nested_loss_details(details, detach=detach)

    def _validate_detail_path(self, name: object) -> str:
        if not isinstance(name, str):
            raise TypeError("loss detail key must be a string.")
        if not name:
            raise ValueError("loss detail key must not be empty.")
        for part in name.split(self.detail_key_separator):
            self._validate_detail_name(part)
        return name
