from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from torch import nn

from anytrain import observation
from anytrain.observation import Bins, Curve, EMAHistory, ForwardObservation

from .output import QuantizeOutput


class _Quantizer(Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...


class ObservedQuantizer:
    def observation_specs(self) -> tuple[observation.ObservationSpec, ...]:
        return (_QUANTIZATION_OBSERVATION,)


def register(module: nn.Module) -> None:
    observation.recommend(module)


def collect(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> dict[str, observation.Observation]:
    del inputs, kwargs
    if not isinstance(output, QuantizeOutput):
        raise TypeError("quantization observation requires QuantizeOutput from module forward.")
    sizes = tuple(cast(_Quantizer, module).codebook_sizes)
    values: dict[str, observation.Observation] = {
        "codes": Bins(
            output.indices,
            sizes=sizes,
            group_dim=None if len(sizes) == 1 else -1,
            active_mask=output.active_codebook_mask,
            group_names=tuple(f"book_{index}" for index in range(len(sizes))),
        )
    }
    loss = output.loss
    if loss is not None:
        values.update(
            {
                "loss/commitment": Curve(loss.commitment),
                "loss/codebook": Curve(loss.codebook),
                "loss/total": Curve(loss.total),
            }
        )
    return values


_QUANTIZATION_OBSERVATION = ForwardObservation(
    name="quantization",
    collect=collect,
    history=EMAHistory(half_life_steps=1_000),
    recommended=True,
    covers_descendants=True,
)


__all__ = ["ObservedQuantizer", "register"]
