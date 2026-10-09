from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from torch.distributed.checkpoint.state_dict import StateDictOptions


def checkpoint_optimizers(optimizer: Optimizer) -> tuple[Optimizer, ...]:
    """Return leaf optimizers suitable for distributed checkpoint state dict APIs.

    ``CompositeOptimizer.state_dict()`` intentionally preserves child names for
    Lightning and scheduler checkpoint compatibility. PyTorch distributed
    checkpoint APIs instead require ordinary optimizer state dicts, so callers
    should pass this result to ``get_optimizer_state_dict()`` and
    ``set_optimizer_state_dict()``.
    """
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer.")

    optimizers: list[Optimizer] = []
    seen: set[int] = set()

    def visit(current: Optimizer) -> None:
        if id(current) in seen:
            raise ValueError(
                "An optimizer instance appears more than once in a composite optimizer."
            )
        seen.add(id(current))

        if isinstance(current, CompositeOptimizer):
            for child in current.optimizers.values():
                visit(child)
            return

        optimizers.append(current)

    visit(optimizer)
    return tuple(optimizers)


def checkpoint_leaf_states(
    optimizers: Sequence[Optimizer],
    optimizer_state: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Split one merged DCP optimizer state into per-optimizer leaf states.

    ``get_optimizer_state_dict(model, checkpoint_optimizers(optimizer))`` merges
    the leaf parameter groups and per-parameter state under one state dict;
    this helper re-splits it so each leaf can be restored independently.
    """
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, Sequence):
        raise ValueError(
            "Optimizer checkpoint state must contain 'state' and 'param_groups'."
        )
    if isinstance(groups, (str, bytes)):
        raise TypeError("Optimizer checkpoint param_groups must be a sequence.")

    offset = 0
    assigned_state_keys: set[Any] = set()
    leaf_states: list[dict[str, Any]] = []
    for optimizer in optimizers:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("checkpoint leaves must be torch.optim.Optimizer instances.")
        next_offset = offset + len(optimizer.param_groups)
        leaf_groups = list(groups[offset:next_offset])
        if len(leaf_groups) != len(optimizer.param_groups):
            raise ValueError(
                "Optimizer checkpoint parameter-group count does not match the configured optimizers."
            )

        parameter_keys: set[Any] = set()
        copied_groups: list[dict[str, Any]] = []
        for group in leaf_groups:
            if not isinstance(group, Mapping):
                raise TypeError("Optimizer checkpoint parameter groups must be mappings.")
            parameters = group.get("params")
            if not isinstance(parameters, Sequence) or isinstance(
                parameters, (str, bytes)
            ):
                raise TypeError(
                    "Optimizer checkpoint parameter groups must contain parameter sequences."
                )
            parameter_keys.update(parameters)
            copied_group = dict(group)
            copied_group["params"] = list(parameters)
            copied_groups.append(copied_group)

        leaf_state = {
            key: value for key, value in state.items() if key in parameter_keys
        }
        assigned_state_keys.update(leaf_state)
        leaf_states.append({"state": leaf_state, "param_groups": copied_groups})
        offset = next_offset

    if offset != len(groups):
        raise ValueError(
            "Optimizer checkpoint parameter-group count does not match the configured optimizers."
        )
    unassigned_state_keys = set(state) - assigned_state_keys
    if unassigned_state_keys:
        raise ValueError(
            "Optimizer checkpoint contains state that is not owned by any parameter group."
        )
    return tuple(leaf_states)


def restore_optimizer_states(
    model: nn.Module,
    optimizers: Sequence[Optimizer],
    optimizer_state: Mapping[str, Any],
    options: StateDictOptions,
) -> None:
    """Restore merged DCP state one optimizer at a time.

    DCP cannot load a merged state containing both stateful and stateless
    optimizers in strict mode, and full-state loading cannot infer a device
    from an entirely empty ``state`` mapping. Stateless leaves only need their
    parameter-group metadata restored, which the ordinary optimizer API can do
    without tensor redistribution.
    """
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    states = checkpoint_leaf_states(optimizers, optimizer_state)
    for optimizer, state in zip(optimizers, states, strict=True):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer leaves must be torch.optim.Optimizer instances.")
        if state["state"]:
            set_optimizer_state_dict(
                model,
                optimizer,
                optim_state_dict=state,
                options=options,
            )
        else:
            optimizer.load_state_dict(state)


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
        self.refresh_param_groups()

    def refresh_param_groups(self) -> None:
        """Re-collect child parameter groups after distributed state restore."""
        for optimizer in self.optimizers.values():
            if isinstance(optimizer, CompositeOptimizer):
                optimizer.refresh_param_groups()
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
