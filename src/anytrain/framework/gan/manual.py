from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

from torch import Tensor, nn

from .._validation import positive_int
from .module import Loss


class _OptimizerLike(Protocol):
    def zero_grad(self, *, set_to_none: bool = False) -> None: ...

    def step(self) -> None: ...


@contextmanager
def freeze_parameters(module: nn.Module, *, eval_mode: bool = False) -> Iterator[None]:
    """Temporarily disable parameter gradients and optionally enter eval mode."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module.")
    if not isinstance(eval_mode, bool):
        raise TypeError("eval_mode must be a bool.")
    states = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    training_states = (
        [(submodule, submodule.training) for submodule in module.modules()] if eval_mode else ()
    )
    for parameter, _ in states:
        parameter.requires_grad_(False)
    if eval_mode:
        module.eval()
    try:
        yield
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)
        if eval_mode:
            for submodule, training in training_states:
                submodule.train(training)


def discriminator_step(
    loss_fn: Loss,
    fake: Tensor,
    real: Tensor,
    optimizer: _OptimizerLike,
    backward: Callable[[Tensor], None],
    *,
    combine_inputs: bool = True,
    gradient_penalty: bool | None = None,
    gradient_penalty_scale: float = 1.0,
) -> Tensor:
    """Run one discriminator update without imposing a Lightning dependency."""

    _require_loss(loss_fn)
    _require_optimizer(optimizer)
    _require_backward(backward)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn.discriminator_loss(
        fake,
        real,
        combine_inputs=combine_inputs,
        gradient_penalty=gradient_penalty,
        gradient_penalty_scale=gradient_penalty_scale,
    )
    backward(loss)
    optimizer.step()
    return loss


def generator_step(
    loss_fn: Loss,
    optimizer: _OptimizerLike,
    backward: Callable[[Tensor], None],
    objective: Callable[[], Tensor],
    *,
    eval_discriminator: bool = False,
) -> Tensor:
    """Run a generator objective with discriminator gradients optionally disabled."""

    _require_loss(loss_fn)
    _require_optimizer(optimizer)
    _require_backward(backward)
    if not callable(objective):
        raise TypeError("objective must be callable.")
    if not isinstance(eval_discriminator, bool):
        raise TypeError("eval_discriminator must be a bool.")
    optimizer.zero_grad(set_to_none=True)
    with freeze_parameters(loss_fn.discriminator, eval_mode=eval_discriminator):
        loss = objective()
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ValueError("generator objective must return a scalar Tensor.")
        backward(loss)
    optimizer.step()
    return loss


def should_apply_gradient_penalty(step: int, *, interval: int = 1) -> bool:
    """Return whether a zero-based discriminator step should run GP."""

    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer.")
    if step < 0:
        raise ValueError("step must be non-negative.")
    interval = positive_int(interval, "interval")
    return step % interval == 0


def _require_optimizer(optimizer: object) -> None:
    zero_grad = getattr(optimizer, "zero_grad", None)
    step = getattr(optimizer, "step", None)
    if not callable(zero_grad) or not callable(step):
        raise TypeError("optimizer must provide zero_grad() and step().")


def _require_loss(loss_fn: object) -> None:
    if not isinstance(loss_fn, Loss):
        raise TypeError("loss_fn must be an anytrain.framework.gan.Loss.")


def _require_backward(backward: object) -> None:
    if not callable(backward):
        raise TypeError("backward must be callable.")


__all__ = [
    "discriminator_step",
    "freeze_parameters",
    "generator_step",
    "should_apply_gradient_penalty",
]
