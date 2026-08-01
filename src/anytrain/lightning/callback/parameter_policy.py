from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lightning import pytorch as pl
from torch import nn


class ParameterTrainability(Protocol):
    def __call__(
        self,
        module: nn.Module,
        name: str,
        parameter: nn.Parameter,
    ) -> bool: ...


class ModuleSelector(Protocol):
    def __call__(self, pl_module: pl.LightningModule) -> nn.Module: ...


@dataclass(frozen=True)
class ParameterTrainabilitySummary:
    tensors: int
    trainable_tensors: int
    frozen_tensors: int
    changed_tensors: int
    parameters: int
    trainable_parameters: int
    frozen_parameters: int


def apply_parameter_trainability(
    module: nn.Module,
    trainable: ParameterTrainability,
) -> ParameterTrainabilitySummary:
    if not isinstance(module, nn.Module):
        raise TypeError("parameter trainability target must be an nn.Module.")
    if not callable(trainable):
        raise TypeError("parameter trainability policy must be callable.")

    tensors = 0
    trainable_tensors = 0
    frozen_tensors = 0
    changed_tensors = 0
    parameters = 0
    trainable_parameters = 0
    frozen_parameters = 0

    for name, parameter in module.named_parameters():
        should_train = trainable(module, name, parameter)
        if not isinstance(should_train, bool):
            raise TypeError(
                "parameter trainability policy must return a boolean for "
                f"{name!r}."
            )
        numel = parameter.numel()
        tensors += 1
        parameters += numel
        if parameter.requires_grad != should_train:
            changed_tensors += 1
        parameter.requires_grad_(should_train)
        if should_train:
            trainable_tensors += 1
            trainable_parameters += numel
        else:
            frozen_tensors += 1
            frozen_parameters += numel

    return ParameterTrainabilitySummary(
        tensors=tensors,
        trainable_tensors=trainable_tensors,
        frozen_tensors=frozen_tensors,
        changed_tensors=changed_tensors,
        parameters=parameters,
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
    )


class ParameterPolicyCallback(pl.Callback):
    def __init__(
        self,
        trainable: ParameterTrainability,
        *,
        module: ModuleSelector | None = None,
    ) -> None:
        if not callable(trainable):
            raise TypeError("parameter policy trainability must be callable.")
        if module is not None and not callable(module):
            raise TypeError("parameter policy module selector must be callable.")
        self.trainable = trainable
        self.module = module
        self.summary: ParameterTrainabilitySummary | None = None
        self.applied = False

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str | None,
    ) -> None:
        del trainer
        if stage != "fit":
            return
        if self.applied:
            return
        module = self.module(pl_module) if self.module is not None else pl_module
        self.summary = apply_parameter_trainability(module, self.trainable)
        self.applied = True


__all__ = [
    "ModuleSelector",
    "ParameterPolicyCallback",
    "ParameterTrainability",
    "ParameterTrainabilitySummary",
    "apply_parameter_trainability",
]
