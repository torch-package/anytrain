from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from anytrain.loss.abc import LossDetails, LossDetailValue


def detach_details(details: Mapping[str, LossDetailValue]) -> LossDetails:
    detached: LossDetails = {}
    for name, value in details.items():
        if not isinstance(name, str):
            raise TypeError("loss detail key must be a string.")
        if not name:
            raise ValueError("loss detail key must not be empty.")
        if "/" in name:
            raise ValueError("loss detail key must not contain '/'.")
        if isinstance(value, bool):
            raise TypeError(f"Loss detail value {name!r} must be a float or 0-d tensor.")
        if isinstance(value, float):
            detached[name] = value
            continue
        if isinstance(value, Tensor):
            if value.ndim != 0:
                raise ValueError(f"Loss detail value {name!r} must be a 0-d tensor.")
            detached[name] = value.detach()
            continue
        raise TypeError(f"Loss detail value {name!r} must be a float or 0-d tensor.")
    return detached


__all__ = ["detach_details"]
