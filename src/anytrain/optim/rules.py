from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

from torch import nn

ExcludedModules = tuple[nn.Module, ...]


class LRScaleRule(TypedDict):
    name: str
    lr_scale: float


LRScaleRules = Sequence[LRScaleRule]

_EMBEDDING_MODULE_TYPES: Final[tuple[type[nn.Module], ...]] = (nn.Embedding, nn.EmbeddingBag)
_NORMALIZATION_MODULE_TYPES: Final[tuple[type[nn.Module], ...]] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.SyncBatchNorm,
)


def validate_excluded_modules(excluded_modules: ExcludedModules) -> None:
    if not isinstance(excluded_modules, tuple):
        raise TypeError("excluded_modules must be a tuple of nn.Module instances.")
    for index, excluded_module in enumerate(excluded_modules):
        if not isinstance(excluded_module, nn.Module):
            raise TypeError(f"excluded_modules[{index}] must be an nn.Module instance.")


def is_muon_parameter_for_module(
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
    *,
    excluded_module_ids: frozenset[int] | None = None,
) -> bool:
    excluded_ids = frozenset() if excluded_module_ids is None else excluded_module_ids
    if (
        id(module) in excluded_ids
        or is_embedding_module(module)
        or is_normalization_module(module)
    ):
        return False
    return parameter_name == "weight" and parameter.ndim == 2


def is_embedding_module(module: nn.Module) -> bool:
    return isinstance(module, _EMBEDDING_MODULE_TYPES)


def is_normalization_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, _NORMALIZATION_MODULE_TYPES) or class_name.endswith("norm")


__all__ = [
    "ExcludedModules",
    "LRScaleRule",
    "LRScaleRules",
    "is_embedding_module",
    "is_muon_parameter_for_module",
    "is_normalization_module",
    "validate_excluded_modules",
]
