from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

import torch
from lightning import pytorch as pl
from torch import Tensor

from anytrain.loss import LossItem


class LossOutputProvider(Protocol):
    def current_loss_outputs(self) -> Mapping[str, Any]: ...


class GradientLoggerCallback(pl.Callback):
    def __init__(
        self,
        loss_pair: tuple[str, str],
        parameter_name: str,
        every_n_steps: int | None = 5_000,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.loss_pair = loss_pair
        self.parameter_name = parameter_name
        self.every_n_steps = every_n_steps
        self.eps = eps

    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: Tensor,
    ) -> None:
        del loss
        if not self.should_run(trainer, pl_module):
            return
        self.log_gradients(trainer, pl_module)

    def should_run(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> bool:
        del pl_module
        if self.every_n_steps is None:
            return True
        return int(trainer.global_step) % self.every_n_steps == 0

    def log_gradients(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer
        parameter = dict(pl_module.named_parameters()).get(self.parameter_name)
        if parameter is None:
            raise KeyError(
                f"unknown parameter {self.parameter_name!r}; "
                "use a name from pl_module.named_parameters()"
            )

        provider = cast(LossOutputProvider, cast(object, pl_module))
        outputs = provider.current_loss_outputs()
        loss_a = loss_value(outputs, self.loss_pair[0])
        loss_b = loss_value(outputs, self.loss_pair[1])

        grad_a = torch.autograd.grad(
            loss_a,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grad_b = torch.autograd.grad(
            loss_b,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]

        norm_a = gradient_norm(grad_a, parameter)
        norm_b = gradient_norm(grad_b, parameter)
        log_ratio = torch.log(norm_a.clamp_min(self.eps)) - torch.log(
            norm_b.clamp_min(self.eps)
        )
        cosine = gradient_cosine(grad_a, grad_b, self.eps, parameter)

        prefix = f"grad/{self.loss_pair[0]}_{self.loss_pair[1]}"
        for name, value in (
            (f"norm/{self.loss_pair[0]}", norm_a),
            (f"norm/{self.loss_pair[1]}", norm_b),
            ("log_ratio", log_ratio),
            ("cosine", cosine),
        ):
            pl_module.log(
                f"{prefix}/{name}",
                value.detach(),
                on_step=True,
                logger=True,
                sync_dist=True,
            )


class GradientNormLoggerCallback(pl.Callback):
    def __init__(
        self,
        tag: str = "train/grad_norm",
        every_n_steps: int | None = 100,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        self.tag = tag
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del optimizer
        if not self.should_run(trainer, pl_module):
            return
        self.on_log_attempt(trainer, pl_module)
        self.log_grad_norm(pl_module)

    def should_run(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> bool:
        del pl_module
        if self.every_n_steps is None:
            return True
        return int(trainer.global_step) % self.every_n_steps == 0

    def on_log_attempt(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module

    def log_grad_norm(self, pl_module: pl.LightningModule) -> None:
        gradients = [
            parameter.grad.detach().norm(2)
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            pl_module.log(
                self.tag,
                torch.stack(gradients).norm(2),
                on_step=True,
                sync_dist=True,
            )


def loss_value(outputs: Mapping[str, Any], name: str) -> Tensor:
    value = outputs.get(name)
    if not isinstance(value, LossItem):
        raise KeyError(f"output {name!r} is not a LossItem")
    return value.loss.mean()


def gradient_norm(grad: Tensor | None, parameter: Tensor) -> Tensor:
    if grad is None:
        return parameter.new_zeros(())
    return grad.detach().norm()


def gradient_cosine(
    grad_a: Tensor | None,
    grad_b: Tensor | None,
    eps: float,
    parameter: Tensor,
) -> Tensor:
    if grad_a is None or grad_b is None:
        return parameter.new_tensor(float("nan"))
    flat_a = grad_a.detach().flatten()
    flat_b = grad_b.detach().flatten()
    denominator = flat_a.norm() * flat_b.norm()
    if denominator < eps:
        return parameter.new_tensor(float("nan"))
    return torch.dot(flat_a, flat_b) / denominator


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "GradientLoggerCallback",
    "GradientNormLoggerCallback",
    "LossOutputProvider",
    "gradient_cosine",
    "gradient_norm",
    "loss_value",
]
