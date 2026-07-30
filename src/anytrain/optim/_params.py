"""Shared parameter planning helpers for optimizer group construction."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from torch import nn

from .rules import (
    ExcludedModules,
    LRScaleRule,
    LRScaleRules,
    is_embedding_module,
    is_muon_parameter_for_module,
    is_normalization_module,
    validate_excluded_modules,
)

_LR_SCALE_RULE_KEYS = frozenset({"name", "lr_scale"})
_DEFAULT_LR_SCALE = 1.0


@dataclass(frozen=True)
class ParameterPlan:
    """One traversal's result for all optimizer parameter groups."""

    all_params: tuple[nn.Parameter, ...]
    muon_params: tuple[nn.Parameter, ...]
    adamw_decay_params: tuple[nn.Parameter, ...]
    adamw_no_decay_params: tuple[nn.Parameter, ...]
    lr_scales: dict[int, float]

    @property
    def non_muon_params(self) -> tuple[nn.Parameter, ...]:
        muon_ids = {id(parameter) for parameter in self.muon_params}
        return tuple(parameter for parameter in self.all_params if id(parameter) not in muon_ids)


def validate_module(module: nn.Module) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError("`module` must be an instance of torch.nn.Module.")


def build_parameter_plan(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    decay_selected_params: bool = True,
    lr_scale_rules: LRScaleRules = (),
) -> ParameterPlan:
    """Resolve Muon, AdamW, and LR-scale groups in one module traversal."""
    validate_module(module)
    if not isinstance(decay_selected_params, bool):
        raise TypeError("decay_selected_params must be a bool.")

    named_modules = tuple(module.named_modules())
    selected_param_ids = _selected_param_ids(module, selected_params)
    excluded_module_ids = _excluded_module_ids(named_modules, excluded_modules)
    rules = _lr_scale_rules(
        {module_name for module_name, _ in named_modules},
        lr_scale_rules,
    )
    entries_by_id: dict[int, tuple[nn.Parameter, bool, bool]] = {}
    lr_scale_by_parameter_id: dict[int, tuple[int, float, str]] = {}

    for module_name, child_module in named_modules:
        matches = _matching_lr_scale_rules(module_name, rules)
        for parameter_name, parameter in child_module.named_parameters(recurse=False):
            parameter_id = id(parameter)
            for rule_name, lr_scale in matches:
                specificity = _module_name_specificity(rule_name)
                previous = lr_scale_by_parameter_id.get(parameter_id)
                if previous is None or specificity > previous[0]:
                    lr_scale_by_parameter_id[parameter_id] = (
                        specificity,
                        lr_scale,
                        rule_name,
                    )
                elif specificity == previous[0] and lr_scale != previous[1]:
                    raise ValueError(
                        "lr_scale rules assign conflicting scales with the same specificity: "
                        f"{previous[2]!r} and {rule_name!r}."
                    )

            if requires_grad_only and not parameter.requires_grad:
                continue
            if selected_param_ids is not None and parameter_id not in selected_param_ids:
                continue

            muon = is_muon_parameter_for_module(
                child_module,
                parameter_name,
                parameter,
                excluded_module_ids=excluded_module_ids,
            )
            decay = (
                decay_selected_params
                and id(child_module) not in excluded_module_ids
                and not is_embedding_module(child_module)
                and not is_normalization_module(child_module)
                and parameter_name == "weight"
                and parameter.ndim >= 2
            )
            previous_entry = entries_by_id.get(parameter_id)
            if previous_entry is None:
                entries_by_id[parameter_id] = (parameter, muon, decay)
            else:
                entries_by_id[parameter_id] = (
                    parameter,
                    previous_entry[1] and muon,
                    previous_entry[2] and decay,
                )

    all_params: list[nn.Parameter] = []
    muon_params: list[nn.Parameter] = []
    adamw_decay_params: list[nn.Parameter] = []
    adamw_no_decay_params: list[nn.Parameter] = []
    for parameter, is_muon, should_decay in entries_by_id.values():
        all_params.append(parameter)
        if is_muon:
            muon_params.append(parameter)
        if should_decay:
            adamw_decay_params.append(parameter)
        else:
            adamw_no_decay_params.append(parameter)

    return ParameterPlan(
        all_params=tuple(all_params),
        muon_params=tuple(muon_params),
        adamw_decay_params=tuple(adamw_decay_params),
        adamw_no_decay_params=tuple(adamw_no_decay_params),
        lr_scales={
            parameter_id: lr_scale
            for parameter_id, (_, lr_scale, _) in lr_scale_by_parameter_id.items()
        },
    )


def make_scaled_param_groups(
    groups: Sequence[tuple[Collection[nn.Parameter], Mapping[str, object]]],
    *,
    base_lr: float,
    lr_scales: Mapping[int, float],
) -> list[dict[str, object]]:
    param_groups: list[dict[str, object]] = []

    for params, options in groups:
        params_by_scale: dict[float, list[nn.Parameter]] = {}
        for parameter in params:
            lr_scale = lr_scales.get(id(parameter), _DEFAULT_LR_SCALE)
            params_by_scale.setdefault(lr_scale, []).append(parameter)

        for lr_scale, scaled_params in params_by_scale.items():
            group = dict(options)
            group["params"] = scaled_params
            group["lr"] = base_lr * lr_scale
            param_groups.append(group)

    return param_groups


def _selected_param_ids(
    module: nn.Module,
    selected_params: Collection[nn.Parameter] | None,
) -> set[int] | None:
    if selected_params is None:
        return None
    if not isinstance(selected_params, Collection):
        raise TypeError("selected_params must be a collection of nn.Parameter.")

    module_param_ids = {id(parameter) for parameter in module.parameters()}
    selected_param_ids: set[int] = set()
    for index, parameter in enumerate(selected_params):
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(f"selected_params[{index}] must be an nn.Parameter.")
        if id(parameter) not in module_param_ids:
            raise ValueError(f"selected_params[{index}] must belong to `module`.")
        selected_param_ids.add(id(parameter))
    return selected_param_ids


def _excluded_module_ids(
    named_modules: tuple[tuple[str, nn.Module], ...],
    excluded_modules: ExcludedModules,
) -> frozenset[int]:
    validate_excluded_modules(excluded_modules)
    module_ids = {id(child_module) for _, child_module in named_modules}
    excluded_module_ids: set[int] = set()
    for index, excluded_module in enumerate(excluded_modules):
        if id(excluded_module) not in module_ids:
            raise ValueError(f"excluded_modules[{index}] must belong to `module`.")
        excluded_module_ids.update(
            id(child_module) for child_module in excluded_module.modules()
        )
    return frozenset(excluded_module_ids)


def _lr_scale_rules(
    module_names: set[str],
    lr_scale_rules: LRScaleRules,
) -> tuple[tuple[str, float], ...]:
    if isinstance(lr_scale_rules, (str, bytes)) or not isinstance(lr_scale_rules, Sequence):
        raise TypeError("lr_scale_rules must be a sequence of LRScaleRule mappings.")

    lr_scale_by_name: dict[str, float] = {}
    for index, rule in enumerate(lr_scale_rules):
        name, lr_scale = _parse_lr_scale_rule(rule, index=index)
        if name not in module_names:
            raise ValueError(f"lr_scale_rules[{index}].name must belong to `module`.")

        previous_lr_scale = lr_scale_by_name.get(name)
        if previous_lr_scale is not None and previous_lr_scale != lr_scale:
            raise ValueError(f"lr_scale_rules contains conflicting scales for {name!r}.")
        lr_scale_by_name[name] = lr_scale

    return tuple(lr_scale_by_name.items())


def _parse_lr_scale_rule(rule: LRScaleRule, *, index: int) -> tuple[str, float]:
    if not isinstance(rule, Mapping):
        raise TypeError(f"lr_scale_rules[{index}] must be a mapping.")
    if set(rule) != _LR_SCALE_RULE_KEYS:
        raise ValueError("LRScaleRule must contain exactly 'name' and 'lr_scale'.")

    name = rule["name"]
    if not isinstance(name, str) or not name:
        raise TypeError(f"lr_scale_rules[{index}].name must be a non-empty string.")

    lr_scale = rule["lr_scale"]
    if isinstance(lr_scale, bool) or not isinstance(lr_scale, (int, float)):
        raise TypeError(f"lr_scale_rules[{index}].lr_scale must be a float.")
    resolved_scale = float(lr_scale)
    if not math.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError(
            f"lr_scale_rules[{index}].lr_scale must be finite and positive."
        )
    return name, resolved_scale


def _matching_lr_scale_rules(
    module_name: str,
    rules: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (rule_name, lr_scale)
        for rule_name, lr_scale in rules
        if module_name == rule_name or module_name.startswith(f"{rule_name}.")
    )


def _module_name_specificity(name: str) -> int:
    return name.count(".") + 1
