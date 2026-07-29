"""Shared parameter traversal helpers for optimizer group construction."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence

from torch import nn

from .rules import LRScaleRule, LRScaleRules

ParameterPredicate = Callable[[nn.Module, str, nn.Parameter], bool]
_LR_SCALE_RULE_KEYS = frozenset({"name", "lr_scale"})
_DEFAULT_LR_SCALE = 1.0


def validate_module(module: nn.Module) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError("`module` must be an instance of torch.nn.Module.")


def split_parameters_by_predicate(
    module: nn.Module,
    predicate: ParameterPredicate,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    validate_module(module)
    selected_param_ids = _selected_param_ids(module, selected_params)
    entries_by_id: dict[int, tuple[nn.Parameter, bool]] = {}

    for child_module in module.modules():
        for parameter_name, parameter in child_module.named_parameters(recurse=False):
            if requires_grad_only and not parameter.requires_grad:
                continue
            if selected_param_ids is not None and id(parameter) not in selected_param_ids:
                continue

            matches = predicate(child_module, parameter_name, parameter)
            parameter_id = id(parameter)
            previous_entry = entries_by_id.get(parameter_id)
            if previous_entry is None:
                entries_by_id[parameter_id] = (parameter, matches)
            else:
                entries_by_id[parameter_id] = (parameter, previous_entry[1] and matches)

    matched_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for parameter, matches in entries_by_id.values():
        if matches:
            matched_params.append(parameter)
        else:
            other_params.append(parameter)
    return matched_params, other_params


def make_scaled_param_groups(
    module: nn.Module,
    groups: Sequence[tuple[Collection[nn.Parameter], Mapping[str, object]]],
    *,
    base_lr: float,
    lr_scale_rules: LRScaleRules = (),
) -> list[dict[str, object]]:
    lr_scales = _parameter_lr_scales(module, lr_scale_rules)
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


def _parameter_lr_scales(
    module: nn.Module,
    lr_scale_rules: LRScaleRules = (),
) -> dict[int, float]:
    validate_module(module)
    rules = _lr_scale_rules(module, lr_scale_rules)
    if not rules:
        return {}

    lr_scale_by_parameter_id: dict[int, tuple[int, float, str]] = {}
    for module_name, child_module in module.named_modules():
        matches = _matching_lr_scale_rules(module_name, rules)
        if not matches:
            continue

        for parameter in child_module.parameters(recurse=False):
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

    return {
        parameter_id: lr_scale
        for parameter_id, (_, lr_scale, _) in lr_scale_by_parameter_id.items()
    }


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


def _lr_scale_rules(
    module: nn.Module,
    lr_scale_rules: LRScaleRules,
) -> tuple[tuple[str, float], ...]:
    if isinstance(lr_scale_rules, (str, bytes)) or not isinstance(lr_scale_rules, Sequence):
        raise TypeError("lr_scale_rules must be a sequence of LRScaleRule mappings.")

    module_names = set(dict(module.named_modules()))
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
