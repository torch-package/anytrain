from __future__ import annotations

from collections.abc import Collection, Mapping

import torch
from torch import nn

from ._params import build_parameter_plan, make_scaled_param_groups
from .adamw import create_adamw_optimizer_from_plan
from .compose import CompositeOptimizer
from .options import (
    DEFAULT_MUON_ADJUST_LR_FN,
    AdamWOptions,
    MuonOptions,
)
from .rules import ExcludedModules, LRScaleRules


def muon_available() -> bool:
    return getattr(torch.optim, "Muon", None) is not None


def create_muon_adamw_optimizer(
    module: nn.Module,
    *,
    muon: MuonOptions,
    adamw: AdamWOptions,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
) -> CompositeOptimizer:
    plan = build_parameter_plan(
        module,
        requires_grad_only=requires_grad_only,
        excluded_modules=excluded_modules,
        lr_scale_rules=lr_scale_rules,
    )
    if not plan.muon_params:
        raise ValueError("No parameters are eligible for Muon.")

    optimizers: dict[str, torch.optim.Optimizer] = {
        "muon": _create_muon_optimizer(
            plan.muon_params,
            muon,
            lr_scales=plan.lr_scales,
        ),
    }
    non_muon_params = plan.non_muon_params
    if non_muon_params:
        optimizers["adamw"] = create_adamw_optimizer_from_plan(
            plan,
            adamw,
            include_param_ids={id(parameter) for parameter in non_muon_params},
        )

    return CompositeOptimizer(optimizers)


def split_muon_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split module parameters into Muon and non-Muon groups.

    By default, Muon receives trainable 2D weight matrices, except parameters
    owned by embedding or normalization modules. Modules passed in
    ``excluded_modules`` are also excluded. Shared parameters are only assigned
    to Muon when every owner is Muon-eligible.
    """

    plan = build_parameter_plan(
        module,
        requires_grad_only=requires_grad_only,
        excluded_modules=excluded_modules,
    )
    return list(plan.muon_params), list(plan.non_muon_params)


def _create_muon_optimizer(
    parameters: Collection[nn.Parameter],
    options: MuonOptions,
    *,
    lr_scales: Mapping[int, float],
) -> torch.optim.Optimizer:
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError(
            "torch.optim.Muon is not available in this PyTorch version. "
            "Muon requires a newer PyTorch."
        )

    optimizer_options = dict(options)
    lr = optimizer_options.pop("lr")
    weight_decay = optimizer_options.pop("weight_decay", 0.1)
    optimizer_options.setdefault("adjust_lr_fn", DEFAULT_MUON_ADJUST_LR_FN)

    param_groups = make_scaled_param_groups(
        ((parameters, {}),),
        base_lr=lr,
        lr_scales=lr_scales,
    )
    return muon_cls(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_options,
    )


__all__ = [
    "ExcludedModules",
    "create_muon_adamw_optimizer",
    "muon_available",
    "split_muon_params",
]
