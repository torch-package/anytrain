from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from anytrain import observation

from .bank import ParamExpertBank
from .expand import (
    build_expert_runtime,
    validate_gate_num_experts,
    validate_moe_top_k,
    validate_router_num_experts,
)
from .expert import Expert
from .grid import MoEGrid
from .metrics import moe_routing_metrics, tokens_per_expert
from .params import ExpertInit, expand_expert
from .partitioner import MoEPartitioner, PartitionLayout
from .routing import ExpertRouting, materialize_routing, validate_num_experts
from .types import MoEOutput


class MoE(nn.Module):
    """Unified MoE entry: partition, route, execute experts, restore.

    ``num_experts`` is the single public source of truth. Callers pass an
    :class:`Expert` object plus expert-major ``weight`` / ``bias`` Parameters
    and an optional thin ``grid`` descriptor; MoE calls ``expert.bind(...)`` to
    build the internal bank and partitioner.

    ``top_k=None`` selects a dense mixture; ``top_k=K`` materializes sparse
    top-k routes. Both paths return ``MoEOutput``. Runtime ``forward(..., top_k=)``
    may override the default density.
    """

    router: nn.Module
    expert_bank: ParamExpertBank
    partitioner: MoEPartitioner
    gate: nn.Module | None
    num_experts: int
    top_k: int | None

    def __init__(
        self,
        *,
        num_experts: int,
        router: nn.Module,
        expert: Expert,
        weight: Tensor,
        bias: Tensor | None = None,
        grid: MoEGrid | None = None,
        gate: nn.Module | None = None,
        top_k: int | None = None,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        validate_num_experts(num_experts)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        validate_moe_top_k(top_k, num_experts=num_experts)
        validate_router_num_experts(router, num_experts)
        validate_gate_num_experts(gate, num_experts)

        expert_bank, partitioner = build_expert_runtime(
            expert,
            weight,
            bias,
            num_experts=num_experts,
            grid=grid,
        )
        self.num_experts = num_experts
        self.router = router
        self.expert_bank = expert_bank
        self.partitioner = partitioner
        self.gate = gate
        self.top_k = top_k
        self.eps = eps
        self._gate_accepts_mask = _module_accepts_keyword(gate, "mask")
        self._gate_accepts_collect_stats = _module_accepts_keyword(gate, "collect_stats")
        self._router_accepts_mask = _module_accepts_keyword(router, "mask")
        observation.registry.recommend(self)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        num_experts: int,
        router: nn.Module | None = None,
        gate: nn.Module | None = None,
        grid: MoEGrid | None = None,
        top_k: int | None = None,
        init: ExpertInit = "copy",
        eps: float = 1e-9,
    ) -> MoE:
        """Build MoE from a supported single-op module.

        This is a convenience wrapper around expand_expert, not a general
        nn.Module graph compiler. Supported modules are the same stock
        single-op modules accepted by expand_expert. When router is omitted,
        a matching default router is created from the module input shape.
        """

        expert, weight, bias = expand_expert(module, num_experts, init=init)
        if router is None:
            router = _default_router_for_module(module, num_experts)
        return cls(
            num_experts=num_experts,
            router=router,
            expert=expert,
            weight=weight,
            bias=bias,
            grid=grid,
            gate=gate,
            top_k=top_k,
            eps=eps,
        )

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        top_k: int | None = None,
        **router_kwargs: Any,
    ) -> MoEOutput:
        cells, router_features, layout = self.partitioner.partition(x, mask=mask)
        features = cells if router_features is None else router_features
        runtime_top_k = self.top_k if top_k is None else top_k
        if runtime_top_k is not None:
            validate_moe_top_k(runtime_top_k, num_experts=self.num_experts)
        routing = self.route(
            features,
            mask=layout.mask,
            top_k=runtime_top_k,
            **router_kwargs,
        )
        return self._execute(
            cells,
            routing,
            layout,
            check_index_bounds=False,
        )

    def forward_manually(self, x: Tensor, routing: ExpertRouting) -> MoEOutput:
        cells, _router_features, layout = self.partitioner.partition(x, mask=routing.mask)
        return self._execute(cells, routing, layout, check_index_bounds=True)

    def route(
        self,
        features: Tensor,
        *,
        mask: Tensor | None = None,
        top_k: int | None = None,
        **router_kwargs: Any,
    ) -> ExpertRouting:
        router_kwargs = dict(router_kwargs)
        collect_stats = router_kwargs.pop("collect_stats", None)
        if mask is not None and self._router_accepts_mask:
            router_kwargs.setdefault("mask", mask)
        logits = self.router(features, **router_kwargs)
        if not isinstance(logits, Tensor):
            raise TypeError(
                f"router must return a logits Tensor, got {type(logits)!r}."
            )
        if logits.size(-1) != self.num_experts:
            raise ValueError(
                "router logits last dimension must match num_experts: "
                f"got {logits.size(-1)}, expected {self.num_experts}."
            )
        probabilities = self._gate_logits(
            logits,
            mask=mask,
            collect_stats=collect_stats,
        )
        return materialize_routing(
            logits,
            probabilities,
            top_k=top_k,
            mask=mask,
            eps=self.eps,
        )

    def _execute(
        self,
        cells: Tensor,
        routing: ExpertRouting,
        layout: PartitionLayout,
        *,
        check_index_bounds: bool,
    ) -> MoEOutput:
        cell_outputs = self.expert_bank(
            cells,
            routing,
            check_index_bounds=check_index_bounds,
        )
        if not isinstance(cell_outputs, Tensor):
            raise TypeError("expert_bank must return a Tensor.")
        restored_obj = layout.restore(cell_outputs)
        restored_obj = self.expert_bank.apply_bias(
            restored_obj,
            routing,
            layout,
            cell_outputs=cell_outputs,
        )
        if not isinstance(restored_obj, Tensor):
            raise TypeError("layout.restore / expert_bank.apply_bias must return a Tensor.")
        restored: Tensor = restored_obj
        assignment_counts = tokens_per_expert(routing)
        dropped = None if routing.is_dense else _dropped_mask(routing)
        return MoEOutput(
            hidden_states=restored,
            routing=routing,
            tokens_per_expert=assignment_counts,
            dropped_mask=dropped,
        )

    def _gate_logits(
        self,
        logits: Tensor,
        *,
        mask: Tensor | None,
        collect_stats: bool | None,
    ) -> Tensor:
        if self.gate is None:
            return logits.softmax(dim=-1)
        kwargs: dict[str, Any] = {}
        if mask is not None and self._gate_accepts_mask:
            kwargs["mask"] = mask
        if self._gate_accepts_collect_stats:
            kwargs["collect_stats"] = collect_stats
        probabilities = self.gate(logits, **kwargs)
        if not isinstance(probabilities, Tensor):
            raise TypeError("gate must return a Tensor of probabilities.")
        if probabilities.shape != logits.shape:
            raise ValueError(
                "gate must return probabilities with the same shape as logits: "
                f"got {tuple(probabilities.shape)} and {tuple(logits.shape)}."
            )
        return probabilities


def _dropped_mask(routing: ExpertRouting) -> Tensor:
    indices, weights, _top_k = routing.require_topk()
    del indices
    return weights.sum(dim=-1) <= 0


def _module_accepts_keyword(module: nn.Module | None, keyword: str) -> bool:
    if module is None:
        return False
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )


def _default_router_for_module(module: nn.Module, num_experts: int) -> nn.Module:
    if isinstance(module, nn.Linear):
        from .router import LinearRouter

        return LinearRouter(module.in_features, num_experts)
    if isinstance(module, nn.Conv1d):
        from .conv import ConvRouter1d

        return ConvRouter1d(module.in_channels, num_experts)
    if isinstance(module, nn.ConvTranspose1d):
        from .conv import ConvRouter1d

        return ConvRouter1d(module.in_channels, num_experts)
    if isinstance(module, nn.Conv2d):
        from .conv import ConvRouter2d

        return ConvRouter2d(module.in_channels, num_experts)
    raise TypeError(
        "MoE.from_module default router supports nn.Linear, nn.Conv1d, "
        "nn.Conv2d, and nn.ConvTranspose1d modules. Pass router=... for "
        f"custom modules; got {type(module)!r}."
    )


def _collect_observation(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> dict[str, observation.Observation]:
    del module, inputs, kwargs
    if not isinstance(output, MoEOutput):
        raise TypeError("MoE observation requires MoEOutput from module forward.")
    return {
        name: observation.Curve(value.detach())
        for name, value in moe_routing_metrics(
            output.routing,
            tokens=output.tokens_per_expert,
            dropped_mask=output.dropped_mask,
        ).items()
    }


observation.registry.register(
    MoE,
    (
        observation.OutputObservation(
            name="routing",
            collect=_collect_observation,
            recommended=True,
        ),
    ),
)


__all__ = ["MoE"]
