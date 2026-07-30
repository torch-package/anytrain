from __future__ import annotations

from enum import auto

import torch
from torch import nn

from anytrain._compat import StrEnum

from .adamw import create_adamw_optimizer
from .muon import create_muon_adamw_optimizer
from .options import (
    AdamWOptions,
    MuonOptions,
)
from .rules import ExcludedModules, LRScaleRules

DEFAULT_ADAMW_WEIGHT_DECAY = 0.01
DEFAULT_MUON_WEIGHT_DECAY = 0.0


class _OptimizationPreset(StrEnum):
    PRETRAIN = auto()
    CPT = auto()
    SFT = auto()


class _OptimizerOption(StrEnum):
    ADAMW = auto()
    MUON = auto()


def create_optimizer(
    module: nn.Module,
    *,
    preset: str = "pretrain",
    optimizer: str = "adamw",
    lr: float | None = None,
    weight_decay: float | None = None,
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
    fused: bool | None = None,
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
) -> torch.optim.Optimizer:
    preset_option = _preset(preset)
    optimizer_option = _optimizer(optimizer)
    adamw = _adamw_options(
        preset_option,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
    )
    if optimizer_option is _OptimizerOption.ADAMW:
        return create_adamw_optimizer(
            module,
            adamw,
            excluded_modules=excluded_modules,
            lr_scale_rules=lr_scale_rules,
        )

    muon: MuonOptions = {
        "lr": adamw["lr"],
        "weight_decay": DEFAULT_MUON_WEIGHT_DECAY,
    }
    return create_muon_adamw_optimizer(
        module,
        muon=muon,
        adamw=adamw,
        excluded_modules=excluded_modules,
        lr_scale_rules=lr_scale_rules,
    )


def _preset(value: str) -> _OptimizationPreset:
    if not isinstance(value, str):
        raise TypeError("preset must be a string.")
    try:
        return _OptimizationPreset(value)
    except ValueError as error:
        raise ValueError("preset must be pretrain, cpt, or sft.") from error


def _optimizer(value: str) -> _OptimizerOption:
    if not isinstance(value, str):
        raise TypeError("optimizer must be a string.")
    try:
        return _OptimizerOption(value)
    except ValueError as error:
        raise ValueError("optimizer must be adamw or muon.") from error


def _adamw_options(
    preset: _OptimizationPreset,
    *,
    lr: float | None,
    weight_decay: float | None,
    betas: tuple[float, float] | None,
    eps: float | None,
    fused: bool | None,
) -> AdamWOptions:
    if preset is _OptimizationPreset.PRETRAIN:
        default_lr = 3e-4
        default_betas = (0.9, 0.95)
    elif preset is _OptimizationPreset.CPT:
        default_lr = 5e-5
        default_betas = (0.9, 0.95)
    elif preset is _OptimizationPreset.SFT:
        default_lr = 2e-5
        default_betas = (0.9, 0.999)
    else:
        raise ValueError("preset must be pretrain, cpt, or sft.")
    return {
        "lr": default_lr if lr is None else lr,
        "weight_decay": DEFAULT_ADAMW_WEIGHT_DECAY if weight_decay is None else weight_decay,
        "betas": default_betas if betas is None else betas,
        "eps": 1e-8 if eps is None else eps,
        "fused": fused,
    }


__all__ = ["create_optimizer"]
