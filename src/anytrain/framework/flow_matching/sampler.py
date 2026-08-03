from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from ._deps import (
    MixtureDiscreteEulerSolver,
    MixtureDiscreteProbPath,
    ODESolver,
)
from .time import DEFAULT_TIME_EPS
from .types import FlowSampleOutput, ModelCaller, default_call_model


class _ModelAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        call_model: ModelCaller,
        model_extras: dict[str, object],
    ):
        super().__init__()
        self.model = model
        self.call_model = call_model
        self.model_extras = model_extras

    def forward(
        self,
        x: Tensor | None = None,
        t: Tensor | None = None,
        **solver_extras: object,
    ) -> Tensor:
        if x is None:
            raise TypeError("x is required.")
        if t is None:
            raise TypeError("t is required.")

        extras = dict(self.model_extras)
        extras.update(solver_extras)
        return self.call_model(self.model, x, _expand_time(t, x), extras)


def _expand_time(t: Tensor, x: Tensor) -> Tensor:
    if t.ndim == 0:
        return t.repeat(x.shape[0])
    if t.ndim == 1 and t.numel() == 1 and x.shape[0] != 1:
        return t.repeat(x.shape[0])
    return t


def _final_state(sample: Tensor, return_intermediates: bool) -> Tensor:
    if return_intermediates and sample.ndim > 0:
        return sample[-1]
    return sample


class ODESampler:
    def __init__(
        self,
        *,
        solver_factory: Callable[[nn.Module], ODESolver] = ODESolver,
        method: str = "midpoint",
        nfe: int = 20,
        num_steps: int = 10,
        return_intermediates: bool = False,
    ):
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}.")
        if num_steps < 2:
            raise ValueError(f"num_steps must be at least 2, got {num_steps}.")

        self.solver_factory = solver_factory
        self.method = method
        self.nfe = nfe
        self.num_steps = num_steps
        self.return_intermediates = return_intermediates

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        *,
        time_grid: Tensor | None = None,
        call_model: ModelCaller = default_call_model,
        **model_extras: object,
    ) -> FlowSampleOutput:
        adapter = _ModelAdapter(model, call_model, dict(model_extras))
        solver = self.solver_factory(adapter)
        if time_grid is None:
            time_grid = torch.linspace(
                0,
                1,
                self.num_steps,
                device=x_0.device,
                dtype=x_0.dtype,
            )
        elif time_grid.ndim != 1 or time_grid.numel() != self.num_steps:
            raise ValueError("time_grid must have shape (num_steps,).")
        else:
            time_grid = time_grid.to(device=x_0.device, dtype=x_0.dtype)
        states = solver.sample(
            x_init=x_0,
            method=self.method,
            step_size=1 / self.nfe,
            time_grid=time_grid,
            return_intermediates=self.return_intermediates,
        )
        return FlowSampleOutput(
            final=_final_state(states, self.return_intermediates),
            states=states if self.return_intermediates else None,
            time_grid=time_grid,
        )


class DiscreteEulerSampler:
    def __init__(
        self,
        *,
        solver_factory: Callable[..., MixtureDiscreteEulerSolver] = MixtureDiscreteEulerSolver,
        nfe: int = 64,
        num_steps: int = 10,
        eps: float = DEFAULT_TIME_EPS,
        return_intermediates: bool = False,
        verbose: bool = False,
    ):
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}.")
        if num_steps < 2:
            raise ValueError(f"num_steps must be at least 2, got {num_steps}.")
        if not 0 <= eps < 1:
            raise ValueError(f"eps must be in [0, 1), got {eps}.")

        self.solver_factory = solver_factory
        self.nfe = nfe
        self.num_steps = num_steps
        self.eps = eps
        self.return_intermediates = return_intermediates
        self.verbose = verbose

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        *,
        vocab_size: int,
        path: MixtureDiscreteProbPath,
        call_model: ModelCaller = default_call_model,
        **model_extras: object,
    ) -> FlowSampleOutput:
        if x_0.dtype != torch.long:
            raise TypeError(f"x_0 must have dtype torch.long, got {x_0.dtype}.")

        extras = dict(model_extras)

        def prob_fn(
            x: Tensor | None = None,
            t: Tensor | None = None,
            **solver_extras: object,
        ) -> Tensor:
            if x is None:
                raise TypeError("x is required.")
            if t is None:
                raise TypeError("t is required.")

            merged = dict(extras)
            merged.update(solver_extras)
            logits = call_model(model, x, _expand_time(t, x), merged)
            return logits.softmax(dim=-1)

        solver = self.solver_factory(
            prob_fn,
            path=path,
            vocabulary_size=vocab_size,
        )
        time_grid = torch.linspace(0, 1 - self.eps, self.num_steps, device=x_0.device)
        states = solver.sample(
            x_init=x_0,
            step_size=1 / self.nfe,
            time_grid=time_grid,
            return_intermediates=self.return_intermediates,
            verbose=self.verbose,
        )
        return FlowSampleOutput(
            final=_final_state(states, self.return_intermediates),
            states=states if self.return_intermediates else None,
            time_grid=time_grid,
        )


__all__ = [
    "DiscreteEulerSampler",
    "ODESampler",
]
