from __future__ import annotations

from collections.abc import Collection
from typing import Any, cast

import torch
from torch import nn

from ._params import ParameterPlan, build_parameter_plan, make_scaled_param_groups
from .options import AdamWOptions
from .rules import ExcludedModules, LRScaleRules


def create_adamw_optimizer(
    module: nn.Module,
    options: AdamWOptions,
    *,
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
) -> torch.optim.AdamW:
    plan = build_parameter_plan(
        module,
        excluded_modules=excluded_modules,
        lr_scale_rules=lr_scale_rules,
    )
    return create_adamw_optimizer_from_plan(plan, options)


def create_adamw_optimizer_from_plan(
    plan: ParameterPlan,
    options: AdamWOptions,
    *,
    include_param_ids: Collection[int] | None = None,
) -> torch.optim.AdamW:
    optimizer_options = cast(
        dict[str, Any],
        {
            key: value
            for key, value in options.items()
            if key not in ("lr", "weight_decay")
        },
    )
    lr = options["lr"]
    weight_decay = options.get("weight_decay", 0.1)

    decay_params = plan.adamw_decay_params
    no_decay_params = plan.adamw_no_decay_params
    if include_param_ids is not None:
        decay_params = tuple(
            parameter for parameter in decay_params if id(parameter) in include_param_ids
        )
        no_decay_params = tuple(
            parameter for parameter in no_decay_params if id(parameter) in include_param_ids
        )

    raw_param_groups: list[tuple[Collection[nn.Parameter], dict[str, object]]] = []
    if decay_params:
        raw_param_groups.append((decay_params, {"weight_decay": weight_decay}))
    if no_decay_params:
        raw_param_groups.append((no_decay_params, {"weight_decay": 0.0}))
    if not raw_param_groups:
        raise ValueError("No parameters are available for AdamW.")

    param_groups = make_scaled_param_groups(
        raw_param_groups,
        base_lr=lr,
        lr_scales=plan.lr_scales,
    )

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_options,
    )


def split_adamw_decay_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    decay_selected_params: bool = True,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    plan = build_parameter_plan(
        module,
        requires_grad_only=requires_grad_only,
        selected_params=selected_params,
        excluded_modules=excluded_modules,
        decay_selected_params=decay_selected_params,
    )
    return list(plan.adamw_decay_params), list(plan.adamw_no_decay_params)


__all__ = [
    "create_adamw_optimizer",
    "split_adamw_decay_params",
]
