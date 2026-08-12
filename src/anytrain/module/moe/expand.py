"""Compile ``expert`` / ``weight`` / ``bias`` / ``grid`` into an internal MoE runtime."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from .bank import ParamExpertBank
from .expert import Expert, LinearExpert
from .grid import MoEGrid, NativeGrid
from .partitioner import MoEPartitioner
from .routing import validate_num_experts, validate_top_k


def coerce_expert(expert: object) -> Expert:
    """Normalize legacy ``F.linear`` to :class:`LinearExpert`; require :class:`Expert`."""

    if expert is F.linear:
        return LinearExpert()
    if isinstance(expert, Expert):
        return expert
    raise TypeError(
        "MoE expert must be an Expert object (stock Linear/Conv*Expert or "
        f"CallableExpert); got {type(expert)!r}."
    )


def build_expert_runtime(
    expert: Expert | object,
    weight: Tensor,
    bias: Tensor | None,
    *,
    num_experts: int,
    grid: MoEGrid | None,
) -> tuple[ParamExpertBank, MoEPartitioner]:
    """Build an internal expert bank and partitioner via :meth:`Expert.bind`."""

    validate_num_experts(num_experts)
    resolved: MoEGrid = NativeGrid() if grid is None else grid
    bound = coerce_expert(expert).bind(
        weight,
        bias,
        num_experts=num_experts,
        grid=resolved,
    )
    return bound.bank, bound.partitioner


def validate_router_num_experts(router: object, num_experts: int) -> None:
    validate_num_experts(num_experts)
    declared = declared_num_experts(router)
    if declared is None:
        raise ValueError("router must expose num_experts matching MoE.num_experts.")
    if declared != num_experts:
        raise ValueError(
            "router.num_experts must match MoE.num_experts: "
            f"got {declared}, expected {num_experts}."
        )


def validate_gate_num_experts(gate: object | None, num_experts: int) -> None:
    if gate is None:
        return
    declared = declared_num_experts(gate)
    if declared is None:
        raise ValueError("gate must expose num_experts matching MoE.num_experts.")
    if declared != num_experts:
        raise ValueError(
            "gate.num_experts must match MoE.num_experts: "
            f"got {declared}, expected {num_experts}."
        )


def validate_moe_top_k(top_k: int | None, *, num_experts: int) -> None:
    if top_k is None:
        return
    validate_top_k(top_k, num_experts=num_experts)


def declared_num_experts(module: object) -> int | None:
    value = getattr(module, "num_experts", None)
    if value is not None:
        return _declared_count(value, name="declared num_experts")
    config = getattr(module, "config", None)
    if config is not None:
        config_value = getattr(config, "num_experts", None)
        if config_value is not None:
            return _declared_count(
                config_value,
                name="declared config.num_experts",
            )
    return None


def _declared_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    validate_num_experts(value, name=name)
    return value


__all__ = [
    "build_expert_runtime",
    "coerce_expert",
    "declared_num_experts",
    "validate_gate_num_experts",
    "validate_moe_top_k",
    "validate_router_num_experts",
]
