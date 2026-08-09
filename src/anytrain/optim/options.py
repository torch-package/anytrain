from __future__ import annotations

from enum import auto
from typing import Final, Required, TypedDict

from anytrain._compat import StrEnum


class MuonAdjustLRFn(StrEnum):
    ORIGINAL = auto()
    MATCH_RMS_ADAMW = auto()


DEFAULT_MUON_ADJUST_LR_FN: Final[MuonAdjustLRFn] = MuonAdjustLRFn.MATCH_RMS_ADAMW


class AdamWOptions(TypedDict, total=False):
    lr: Required[float]
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    fused: bool | None


class MuonOptions(TypedDict, total=False):
    lr: Required[float]
    weight_decay: float
    momentum: float
    nesterov: bool
    ns_coefficients: tuple[float, float, float]
    eps: float
    ns_steps: int
    adjust_lr_fn: MuonAdjustLRFn | str | None


class MuonAdamWOptions(TypedDict):
    muon: MuonOptions
    adamw: AdamWOptions


OptimizerOptions = AdamWOptions | MuonAdamWOptions


__all__ = [
    "AdamWOptions",
    "DEFAULT_MUON_ADJUST_LR_FN",
    "MuonAdjustLRFn",
    "MuonAdamWOptions",
    "MuonOptions",
    "OptimizerOptions",
]
