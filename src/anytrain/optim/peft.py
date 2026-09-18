"""PEFT adapters for the LoRA-Muon optimizer."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

try:
    from peft.tuners.lora import Linear as PeftLinear
    from peft.tuners.lora import LoraLayer
except ImportError as exc:  # pragma: no cover - depends on the optional environment.
    raise ImportError(
        "`anytrain.optim.peft` requires a compatible Hugging Face PEFT release. "
        "Install it with `python -m pip install 'anytrain[peft]'`."
    ) from exc

from ._adapter import is_exact_or_fsdp
from ._params import build_parameter_plan
from .adamw import create_adamw_optimizer_from_plan
from .compose import CompositeOptimizer
from .lora_muon import LoRAFactorPair, LoRAMuon
from .options import AdamWOptions
from .rules import ExcludedModules, LRScaleRules


def lora_factor_pairs(
    module: nn.Module,
    *,
    adapter_names: Sequence[str] | None = None,
) -> tuple[LoRAFactorPair, ...]:
    """Collect active vanilla PEFT Linear adapters as LoRA factor pairs.

    When ``adapter_names`` is omitted, each layer's active adapters are used.
    Explicitly selected adapters must also be active so their parameters
    participate in the model's forward pass.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module.")
    requested = _adapter_names(adapter_names)
    found: set[str] = set()
    pairs: list[LoRAFactorPair] = []
    parameter_ids: set[int] = set()
    storage_keys: set[tuple[torch.device, int]] = set()

    for path, layer in module.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        installed = set(layer.scaling)
        active = tuple(layer.active_adapters)
        candidates = active if requested is None else requested
        selected = tuple(name for name in candidates if name in installed)
        if not selected:
            continue

        label = path or "<root>"
        if layer.disable_adapters:
            raise ValueError(f"PEFT LoRA layer {label!r} has adapters disabled.")
        if layer.merged:
            raise ValueError(f"PEFT LoRA layer {label!r} must be unmerged for LoRA-Muon.")
        if _uses_tensor_parallel(layer):
            raise ValueError(
                f"PEFT LoRA layer {label!r} uses tensor parallelism, which LoRA-Muon "
                "does not support."
            )
        if not is_exact_or_fsdp(layer, PeftLinear):
            raise TypeError(
                f"PEFT LoRA layer {label!r} has unsupported type "
                f"{type(layer).__name__}; only vanilla PEFT Linear is supported."
            )

        for adapter_name in selected:
            found.add(adapter_name)
            if adapter_name not in active:
                raise ValueError(
                    f"PEFT adapter {adapter_name!r} on layer {label!r} must be active."
                )
            pair = _factor_pair(layer, path=label, adapter_name=adapter_name)
            _register_parameter(
                pair.left,
                label=f"{label}.{adapter_name}.lora_B.weight",
                parameter_ids=parameter_ids,
                storage_keys=storage_keys,
            )
            _register_parameter(
                pair.right,
                label=f"{label}.{adapter_name}.lora_A.weight",
                parameter_ids=parameter_ids,
                storage_keys=storage_keys,
            )
            pairs.append(pair)

    if requested is not None:
        missing = [name for name in requested if name not in found]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(f"PEFT adapters were not found: {names}.")
    if not pairs:
        raise ValueError("No active supported PEFT Linear LoRA adapters were found.")
    return tuple(pairs)


def create_lora_muon_optimizer(
    module: nn.Module,
    *,
    lr: float,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    adapter_names: Sequence[str] | None = None,
) -> LoRAMuon:
    """Create LoRA-Muon when all trainable parameters are selected LoRA factors."""
    pairs = lora_factor_pairs(module, adapter_names=adapter_names)
    selected_ids = _pair_parameter_ids(pairs)
    uncovered = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and id(parameter) not in selected_ids
    ]
    if uncovered:
        preview = ", ".join(uncovered[:5])
        if len(uncovered) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "PEFT LoRA-Muon would leave trainable parameters without an optimizer: "
            f"{preview}. Use create_lora_muon_adamw_optimizer() for mixed training."
        )
    return LoRAMuon(
        pairs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def create_lora_muon_adamw_optimizer(
    module: nn.Module,
    *,
    lr: float,
    adamw: AdamWOptions,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    adapter_names: Sequence[str] | None = None,
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
) -> LoRAMuon | CompositeOptimizer:
    """Optimize PEFT LoRA factors with LoRA-Muon and remaining params with AdamW."""
    pairs = lora_factor_pairs(module, adapter_names=adapter_names)
    selected_ids = _pair_parameter_ids(pairs)
    remaining = tuple(
        parameter
        for _, parameter in module.named_parameters()
        if parameter.requires_grad and id(parameter) not in selected_ids
    )
    lora_muon = LoRAMuon(
        pairs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    if not remaining:
        return lora_muon

    plan = build_parameter_plan(
        module,
        selected_params=remaining,
        excluded_modules=excluded_modules,
        lr_scale_rules=lr_scale_rules,
    )
    return CompositeOptimizer(
        {
            "lora_muon": lora_muon,
            "adamw": create_adamw_optimizer_from_plan(plan, adamw),
        }
    )


def _pair_parameter_ids(pairs: Sequence[LoRAFactorPair]) -> set[int]:
    return {
        id(parameter)
        for pair in pairs
        for parameter in (pair.left, pair.right)
    }


def _adapter_names(adapter_names: Sequence[str] | None) -> tuple[str, ...] | None:
    if adapter_names is None:
        return None
    if isinstance(adapter_names, (str, bytes)) or not isinstance(adapter_names, Sequence):
        raise TypeError("adapter_names must be a sequence of strings or None.")

    result: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(adapter_names):
        if not isinstance(name, str) or not name:
            raise TypeError(f"adapter_names[{index}] must be a non-empty string.")
        if name in seen:
            raise ValueError(f"adapter_names must not contain duplicate {name!r}.")
        seen.add(name)
        result.append(name)
    if not result:
        raise ValueError("adapter_names must not be empty.")
    return tuple(result)


def _factor_pair(
    layer: PeftLinear,
    *,
    path: str,
    adapter_name: str,
) -> LoRAFactorPair:
    if adapter_name in layer.lora_variant or layer.use_dora.get(adapter_name, False):
        raise ValueError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} is a LoRA variant; "
            "LoRA-Muon only supports vanilla LoRA."
        )
    if adapter_name not in layer.lora_A or adapter_name not in layer.lora_B:
        raise ValueError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} has no Linear A/B factors."
        )

    right_module = layer.lora_A[adapter_name]
    left_module = layer.lora_B[adapter_name]
    if not _is_supported_linear_factor(right_module) or not _is_supported_linear_factor(
        left_module
    ):
        raise TypeError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} must use plain nn.Linear "
            "A/B factors."
        )
    if right_module.bias is not None or left_module.bias is not None:
        raise ValueError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} must not use LoRA bias."
        )

    left = left_module.weight
    right = right_module.weight
    if not isinstance(left, nn.Parameter) or not isinstance(right, nn.Parameter):
        raise TypeError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} factors must be nn.Parameter."
        )
    if type(left) is not nn.Parameter or type(right) is not nn.Parameter:
        raise TypeError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} must use local, "
            "unsharded nn.Parameter factors."
        )
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} must have 2D factors."
        )
    if not left.requires_grad or not right.requires_grad:
        raise ValueError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} must train both factors."
        )

    scale = layer.scaling[adapter_name]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} must have a numeric scale."
        )
    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(
            f"PEFT adapter {adapter_name!r} on layer {path!r} scale must be positive and finite."
        )
    return LoRAFactorPair(left=left, right=right, scale=scale_value)


def _is_supported_linear_factor(module: nn.Module) -> bool:
    return is_exact_or_fsdp(module, nn.Linear)


def _uses_tensor_parallel(layer: LoraLayer) -> bool:
    base_layer = layer.get_base_layer()
    return (
        getattr(layer, "_tp_info", None) is not None
        or getattr(base_layer, "_hf_tp_plan", None) is not None
        or getattr(base_layer, "_hf_device_mesh", None) is not None
    )


def _register_parameter(
    parameter: nn.Parameter,
    *,
    label: str,
    parameter_ids: set[int],
    storage_keys: set[tuple[torch.device, int]],
) -> None:
    if id(parameter) in parameter_ids:
        raise ValueError(f"PEFT LoRA factor {label!r} is shared by multiple adapters.")
    if parameter.device.type == "meta":
        raise ValueError(f"PEFT LoRA factor {label!r} must be materialized before optimization.")
    storage_key = (parameter.device, parameter.untyped_storage().data_ptr())
    if storage_key in storage_keys:
        raise ValueError(f"PEFT LoRA factor {label!r} shares storage with another factor.")
    parameter_ids.add(id(parameter))
    storage_keys.add(storage_key)


__all__ = [
    "create_lora_muon_adamw_optimizer",
    "create_lora_muon_optimizer",
    "lora_factor_pairs",
]
