from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from lightning import pytorch as pl
from torch import Tensor

from .. import _capability


class LossOutputProvider(Protocol):
    def current_loss_outputs(self) -> Mapping[str, Tensor]: ...


class GradientLossGroupProvider(Protocol):
    def current_gradient_loss_groups(self) -> Mapping[str, Mapping[str, Tensor]]: ...


GradientLossGroupsFn = Callable[
    [pl.LightningModule],
    Mapping[str, Mapping[str, Tensor]],
]


@dataclass
class GradientProbe:
    name: str
    parameters: tuple[str, ...]
    match: str = "exact"
    trainable_only: bool = True

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "probe name")
        parameters = tuple(self.parameters)
        if not parameters:
            raise ValueError("gradient probe parameters must not be empty.")
        for index, parameter in enumerate(parameters):
            _non_empty_string(parameter, f"probe {self.name!r} parameter {index}")
        if self.match not in {"exact", "regex"}:
            raise ValueError("gradient probe match must be 'exact' or 'regex'.")
        if not isinstance(self.trainable_only, bool):
            raise TypeError("gradient probe trainable_only must be a boolean.")
        self.parameters = parameters


@dataclass(frozen=True)
class GradientTarget:
    loss: str
    group: str = "batch"

    def __post_init__(self) -> None:
        _non_empty_string(self.loss, "gradient target loss")
        _non_empty_string(self.group, "gradient target group")

    @property
    def tag(self) -> str:
        return f"{self.loss}@{self.group}"


@dataclass(frozen=True)
class GradientComparison:
    left: GradientTarget
    right: GradientTarget

    def __post_init__(self) -> None:
        if not isinstance(self.left, GradientTarget):
            raise TypeError("gradient comparison left must be a GradientTarget.")
        if not isinstance(self.right, GradientTarget):
            raise TypeError("gradient comparison right must be a GradientTarget.")

    @property
    def tag(self) -> str:
        return f"{self.left.tag}__{self.right.tag}"


class GradientProbeLoggerCallback(pl.Callback):
    def __init__(
        self,
        comparisons: Sequence[GradientComparison],
        probes: Sequence[GradientProbe] | Mapping[str, Sequence[str]],
        *,
        loss_groups: GradientLossGroupsFn,
        every_n_steps: int | None = 5_000,
        eps: float = 1e-12,
        tag: str = "grad",
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        _non_empty_string(tag, "tag")
        self.comparisons = _comparisons(comparisons)
        self.probes = _probes(probes)
        if not callable(loss_groups):
            raise TypeError("loss_groups must be callable.")
        self.loss_groups = loss_groups
        self.every_n_steps = every_n_steps
        self.eps = eps
        self.tag = tag.strip("/")
        self._last_logged_step: int | None = None

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
        step = _capability.global_step(trainer)
        if self.every_n_steps is not None and step % self.every_n_steps != 0:
            return False
        # Under grad accumulation, global_step is flat across microbatches.
        # Log once per optimizer step window to avoid repeated autograd.grad.
        if self._last_logged_step == step:
            return False
        self._last_logged_step = step
        return True

    def log_gradients(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer
        parameters = dict(pl_module.named_parameters())
        probe_parameters = {
            probe.name: _resolve_probe(parameters, probe) for probe in self.probes
        }
        loss_groups = _validate_loss_groups(
            self.loss_groups(pl_module),
            source="loss_groups provider",
        )

        for comparison in self.comparisons:
            left_outputs = _loss_group(loss_groups, comparison.left.group)
            right_outputs = _loss_group(loss_groups, comparison.right.group)
            self._log_comparison(
                pl_module,
                probe_parameters,
                prefix=f"{self.tag}/{comparison.tag}",
                left_name=comparison.left.tag,
                left_loss=loss_value(left_outputs, comparison.left.loss),
                right_name=comparison.right.tag,
                right_loss=loss_value(right_outputs, comparison.right.loss),
            )

    def _log_comparison(
        self,
        pl_module: pl.LightningModule,
        probe_parameters: Mapping[str, tuple[Tensor, ...]],
        *,
        prefix: str,
        left_name: str,
        left_loss: Tensor,
        right_name: str,
        right_loss: Tensor,
    ) -> None:
        for probe_name, parameters in probe_parameters.items():
            gradients_a = torch.autograd.grad(
                left_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            gradients_b = torch.autograd.grad(
                right_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            norm_a = gradient_group_norm(gradients_a, parameters)
            norm_b = gradient_group_norm(gradients_b, parameters)
            log_ratio = torch.log(norm_a.clamp_min(self.eps)) - torch.log(
                norm_b.clamp_min(self.eps)
            )
            cosine = gradient_group_cosine(
                gradients_a,
                gradients_b,
                self.eps,
                parameters,
            )

            for name, value in (
                (f"{probe_name}/norm/{left_name}", norm_a),
                (f"{probe_name}/norm/{right_name}", norm_b),
                (f"{probe_name}/log_ratio", log_ratio),
                (f"{probe_name}/cosine", cosine),
            ):
                pl_module.log(
                    f"{prefix}/{name}",
                    value.detach(),
                    on_step=True,
                    logger=True,
                    sync_dist=True,
                )


def loss_value(outputs: Mapping[str, Tensor], name: str) -> Tensor:
    try:
        value = outputs[name]
    except KeyError as error:
        raise KeyError(f"missing scalar loss output {name!r}") from error
    if not isinstance(value, Tensor):
        raise TypeError(f"loss output {name!r} must be a tensor.")
    if value.ndim != 0:
        raise ValueError(f"loss output {name!r} must be a scalar tensor.")
    return value


def gradient_loss_groups(
    pl_module: pl.LightningModule,
) -> Mapping[str, Mapping[str, Tensor]]:
    provider = cast(object, pl_module)
    group_provider = getattr(provider, "current_gradient_loss_groups", None)
    if callable(group_provider):
        return _validate_loss_groups(
            group_provider(),
            source="current_gradient_loss_groups()",
        )
    output_provider = getattr(provider, "current_loss_outputs", None)
    if not callable(output_provider):
        raise TypeError(
            "gradient_loss_groups() requires current_gradient_loss_groups() "
            "or current_loss_outputs()."
        )
    outputs = output_provider()
    if not isinstance(outputs, Mapping):
        raise TypeError("current_loss_outputs() must return a mapping.")
    return {"batch": cast(Mapping[str, Tensor], outputs)}


def _validate_loss_groups(
    value: object,
    *,
    source: str,
) -> Mapping[str, Mapping[str, Tensor]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must return a mapping.")
    for name, outputs in value.items():
        _non_empty_string(name, f"{source} group name")
        if not isinstance(outputs, Mapping):
            raise TypeError(f"{source} group {name!r} must be a mapping.")
    return cast(Mapping[str, Mapping[str, Tensor]], value)


def _loss_group(
    loss_groups: Mapping[str, Mapping[str, Tensor]],
    name: str,
) -> Mapping[str, Tensor]:
    try:
        return loss_groups[name]
    except KeyError as error:
        raise KeyError(
            f"unknown gradient loss group {name!r}; "
            "use a group returned by the configured loss_groups provider"
        ) from error


def gradient_norm(grad: Tensor | None, parameter: Tensor) -> Tensor:
    if grad is None:
        return parameter.new_zeros(())
    return grad.detach().norm()


def gradient_group_norm(
    gradients: Sequence[Tensor | None],
    parameters: Sequence[Tensor],
) -> Tensor:
    total = parameters[0].new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            detached = gradient.detach()
            total = total + torch.sum(detached * detached)
    return torch.sqrt(total)


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


def gradient_group_cosine(
    gradients_a: Sequence[Tensor | None],
    gradients_b: Sequence[Tensor | None],
    eps: float,
    parameters: Sequence[Tensor],
) -> Tensor:
    dot = parameters[0].new_zeros(())
    norm_a2 = parameters[0].new_zeros(())
    norm_b2 = parameters[0].new_zeros(())
    for grad_a, grad_b in zip(gradients_a, gradients_b, strict=False):
        detached_a = None if grad_a is None else grad_a.detach()
        detached_b = None if grad_b is None else grad_b.detach()
        if detached_a is not None:
            norm_a2 = norm_a2 + torch.sum(detached_a * detached_a)
        if detached_b is not None:
            norm_b2 = norm_b2 + torch.sum(detached_b * detached_b)
        if detached_a is not None and detached_b is not None:
            dot = dot + torch.sum(detached_a * detached_b)
    denominator = torch.sqrt(norm_a2) * torch.sqrt(norm_b2)
    if denominator < eps:
        return parameters[0].new_tensor(float("nan"))
    return dot / denominator


def _resolve_probe(
    parameters: Mapping[str, Tensor],
    probe: GradientProbe,
) -> tuple[Tensor, ...]:
    matches: list[Tensor] = []
    seen: set[int] = set()
    if probe.match == "exact":
        for name in probe.parameters:
            try:
                parameter = parameters[name]
            except KeyError as error:
                raise KeyError(
                    f"unknown parameter {name!r} for gradient probe {probe.name!r}"
                ) from error
            if probe.trainable_only and not parameter.requires_grad:
                continue
            key = id(parameter)
            if key not in seen:
                seen.add(key)
                matches.append(parameter)
    else:
        patterns = [(pattern, re.compile(pattern)) for pattern in probe.parameters]
        matched_patterns: set[str] = set()
        for name, parameter in parameters.items():
            if probe.trainable_only and not parameter.requires_grad:
                continue
            for pattern, compiled in patterns:
                if compiled.search(name):
                    matched_patterns.add(pattern)
                    key = id(parameter)
                    if key not in seen:
                        seen.add(key)
                        matches.append(parameter)
                    break
        missing = [pattern for pattern, _ in patterns if pattern not in matched_patterns]
        if missing:
            raise KeyError(
                f"gradient probe {probe.name!r} matched no parameters for: "
                + ", ".join(repr(pattern) for pattern in missing)
            )
    if not matches:
        raise KeyError(
            f"gradient probe {probe.name!r} matched no trainable parameters"
        )
    return tuple(matches)


def _comparisons(
    values: Sequence[GradientComparison],
) -> tuple[GradientComparison, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("gradient logger requires at least one comparison.")
    for index, value in enumerate(result):
        if not isinstance(value, GradientComparison):
            raise TypeError(
                f"gradient comparison {index} must be a GradientComparison."
            )
    return result


def _probes(
    probes: Sequence[GradientProbe] | Mapping[str, Sequence[str]],
) -> tuple[GradientProbe, ...]:
    if isinstance(probes, Mapping):
        result = tuple(
            GradientProbe(name=name, parameters=tuple(parameters))
            for name, parameters in probes.items()
        )
    else:
        result = tuple(probes)
    if not result:
        raise ValueError("gradient logger requires at least one probe.")
    names = [probe.name for probe in result]
    if len(set(names)) != len(names):
        raise ValueError("gradient probe names must be unique.")
    return result


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "GradientComparison",
    "GradientLossGroupProvider",
    "GradientLossGroupsFn",
    "GradientProbe",
    "GradientProbeLoggerCallback",
    "GradientTarget",
    "LossOutputProvider",
    "gradient_group_cosine",
    "gradient_group_norm",
    "gradient_loss_groups",
    "loss_value",
]
