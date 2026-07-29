from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.optim import Optimizer


class CompositeOptimizer(Optimizer):
    """Expose multiple optimizers as one optimizer for schedulers and Lightning."""

    def __init__(self, optimizers: Mapping[str, Optimizer]) -> None:
        self.optimizers = self._validate_optimizers(optimizers)
        self._validate_disjoint_parameters(self.optimizers)
        self._allow_add_param_group = True
        super().__init__(self._collect_param_groups(), defaults={})
        self._allow_add_param_group = False

    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers.values():
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._allow_add_param_group:
            super().add_param_group(param_group)
            return
        raise RuntimeError(
            "CompositeOptimizer does not support add_param_group(). "
            "Add parameter groups to the child optimizer before composing."
        )

    def state_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            }
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping.")
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, Mapping):
            raise ValueError("Composite optimizer state must contain an 'optimizers' mapping.")

        expected_names = set(self.optimizers)
        actual_names = set(optimizer_states)
        if actual_names != expected_names:
            raise ValueError(
                "Composite optimizer state names must match current optimizers: "
                f"expected {sorted(expected_names)}, got {sorted(actual_names)}."
            )

        # Keep the checkpoint's child order so a separately serialized scheduler
        # (which addresses param groups positionally) still targets the same
        # named optimizer after a config mapping is reconstructed in another order.
        self.optimizers = {
            name: self.optimizers[name]
            for name in optimizer_states
        }
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(optimizer_states[name])
        self.param_groups = self._collect_param_groups()

    def _validate_optimizers(
        self,
        optimizers: Mapping[str, Optimizer],
    ) -> dict[str, Optimizer]:
        if not isinstance(optimizers, Mapping):
            raise TypeError("optimizers must be a mapping of names to optimizers.")
        if not optimizers:
            raise ValueError("optimizers must contain at least one optimizer.")

        validated: dict[str, Optimizer] = {}
        for name, optimizer in optimizers.items():
            if not isinstance(name, str) or not name:
                raise ValueError("optimizer names must be non-empty strings.")
            if not isinstance(optimizer, Optimizer):
                raise TypeError(f"Optimizer {name!r} must be a torch.optim.Optimizer.")
            validated[name] = optimizer
        return validated

    def _validate_disjoint_parameters(
        self,
        optimizers: Mapping[str, Optimizer],
    ) -> None:
        parameter_owner_by_id: dict[int, str] = {}
        for optimizer_name, optimizer in optimizers.items():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter_id = id(parameter)
                    previous_owner = parameter_owner_by_id.get(parameter_id)
                    if previous_owner is not None:
                        raise ValueError(
                            "Composite optimizer child optimizers must not share parameters: "
                            f"{previous_owner!r} and {optimizer_name!r} both contain one."
                        )
                    parameter_owner_by_id[parameter_id] = optimizer_name

    def _collect_param_groups(self) -> list[dict[str, Any]]:
        return [
            group
            for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]
