from __future__ import annotations

from collections.abc import Mapping, Set

import torch
from torch import Tensor

from ._validation import scalar


def validate_loss_names(loss_names: list[str], *, label: str) -> tuple[str, ...]:
    if not isinstance(loss_names, list):
        raise TypeError(f"{label} must be a list of strings.")
    if not loss_names:
        raise ValueError(f"{label} must contain at least one name.")

    ordered_names: list[str] = []
    unique_names: set[str] = set()
    loss_by_observation_tag: dict[str, str] = {}
    for name in loss_names:
        if not isinstance(name, str):
            raise TypeError("loss name must be a string.")
        if not name:
            raise ValueError("loss name must not be empty.")
        if name in unique_names:
            raise ValueError(f"Duplicate loss name {name!r}.")
        observation_tag = name.strip("/")
        if not observation_tag:
            raise ValueError(
                f"Loss name {name!r} must produce a non-empty observation tag."
            )
        previous_name = loss_by_observation_tag.get(observation_tag)
        if previous_name is not None:
            raise ValueError(
                f"Loss names {previous_name!r} and {name!r} normalize to the same "
                f"observation tag {observation_tag!r}."
            )
        loss_by_observation_tag[observation_tag] = name
        ordered_names.append(name)
        unique_names.add(name)
    return tuple(ordered_names)


def stack_named_losses(
    losses: Mapping[str, Tensor],
    loss_names: tuple[str, ...],
    loss_name_set: Set[str],
) -> Tensor:
    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a mapping of names to scalar tensors.")
    missing_names = [name for name in loss_names if name not in losses]
    if missing_names:
        raise ValueError(f"losses missing configured loss names: {missing_names}.")
    unknown_names = [name for name in losses if name not in loss_name_set]
    if unknown_names:
        raise ValueError(f"losses contains unknown loss names: {unknown_names}.")
    return torch.stack(
        [scalar(losses[name], name=f"loss {name!r}") for name in loss_names]
    )
