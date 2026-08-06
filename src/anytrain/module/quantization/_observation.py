from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast

from torch import nn

from anytrain import observation
from anytrain.observation import Bins, Curve, EMAHistory, OutputObservation

from .output import QuantizeOutput


class _Quantizer(Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...


_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def register(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(module_type, _QUANTIZATION_OBSERVATIONS)
    return module_type


def recommend(module: nn.Module) -> None:
    observation.registry.recommend(module)


def collect_codes(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> dict[str, observation.Observation]:
    del inputs, kwargs
    if not isinstance(output, QuantizeOutput):
        raise TypeError("quantization observation requires QuantizeOutput from module forward.")
    sizes = tuple(cast(_Quantizer, module).codebook_sizes)
    return {
        "codes": Bins(
            output.indices,
            sizes=sizes,
            group_dim=None if len(sizes) == 1 else -1,
            active_mask=output.active_codebook_mask,
            group_names=tuple(f"book_{index}" for index in range(len(sizes))),
        )
    }


def collect_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> dict[str, observation.Observation]:
    del module, inputs, kwargs
    if not isinstance(output, QuantizeOutput):
        raise TypeError("quantization observation requires QuantizeOutput from module forward.")
    loss = output.loss
    if loss is None:
        return {}
    return {
        "commitment": Curve(loss.commitment),
        "codebook": Curve(loss.codebook),
        "total": Curve(loss.total),
    }


_QUANTIZATION_OBSERVATIONS = (
    OutputObservation(
        name="quantization",
        collect=collect_codes,
        history=EMAHistory(half_life_steps=1_000),
        recommended=True,
        covers_descendants=True,
    ),
    OutputObservation(
        name="quantization/loss",
        collect=collect_loss,
    ),
)


__all__ = ["recommend", "register"]
