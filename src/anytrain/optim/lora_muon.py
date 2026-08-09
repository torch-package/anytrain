"""Paper-faithful reference implementation of LoRA-Muon."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class LoRAFactorPair:
    """A factor pair whose effective weight is ``scale * left @ right``."""

    left: nn.Parameter
    right: nn.Parameter
    scale: float = 1.0


class LoRAMuon(Optimizer):
    """LoRA-Muon using exact SVD and eigendecomposition reference kernels.

    This implements Algorithm 1 from "LoRA-Muon: Spectral Steepest Descent
    on the Low-Rank Manifold" (arXiv:2606.12921). Both factors must remain
    numerically full rank; rank-deficient initialization is outside the
    fixed-rank manifold assumed by the paper and is rejected explicitly.
    """

    def __init__(
        self,
        pairs: Sequence[LoRAFactorPair],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
    ) -> None:
        pair_items = _validate_pairs(pairs)
        defaults = {
            "lr": _option("lr", lr, minimum=0.0),
            "momentum": _option("momentum", momentum, minimum=0.0, maximum=1.0),
            "weight_decay": _option("weight_decay", weight_decay, minimum=0.0),
        }
        _validate_decay(defaults["lr"], defaults["weight_decay"])

        param_groups = [
            {
                "params": [pair.left, pair.right],
                "scale": _option(f"pairs[{index}].scale", pair.scale, minimum=0.0),
            }
            for index, pair in enumerate(pair_items)
        ]
        for index, group in enumerate(param_groups):
            if group["scale"] == 0.0:
                raise ValueError(f"pairs[{index}].scale must be positive.")

        self._allow_add_param_group = True
        super().__init__(param_groups, defaults)
        self._allow_add_param_group = False

        for index, pair in enumerate(pair_items):
            _inverse_gram_root(pair.left, name=f"pairs[{index}].left")
            _inverse_gram_root(pair.right.mT, name=f"pairs[{index}].right.T")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        updates: list[
            tuple[
                nn.Parameter,
                nn.Parameter,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for index, group in enumerate(self.param_groups):
            left, right = _group_factors(group, index=index)
            left_grad = left.grad
            right_grad = right.grad
            if left_grad is None and right_grad is None:
                continue
            if left_grad is None or right_grad is None:
                raise RuntimeError(
                    f"LoRA-Muon factor pair {index} must receive gradients for both factors."
                )
            if left_grad.is_sparse or right_grad.is_sparse:
                raise RuntimeError("LoRA-Muon does not support sparse gradients.")

            lr = _option("lr", group["lr"], minimum=0.0)
            momentum = _option("momentum", group["momentum"], minimum=0.0, maximum=1.0)
            weight_decay = _option("weight_decay", group["weight_decay"], minimum=0.0)
            scale = _option("scale", group["scale"], minimum=0.0)
            if scale == 0.0:
                raise ValueError("LoRA-Muon scale must be positive.")
            _validate_decay(lr, weight_decay)

            work_dtype = _work_dtype(left.dtype)
            sqrt_scale = math.sqrt(scale)
            left_value = left.detach().to(dtype=work_dtype)
            right_value = right.detach().to(dtype=work_dtype)
            paper_left = left_value * sqrt_scale
            paper_right = right_value.mT * sqrt_scale
            left_invroot = _inverse_gram_root(
                paper_left,
                name=f"factor pair {index} left",
            )
            right_invroot = _inverse_gram_root(
                paper_right,
                name=f"factor pair {index} right.T",
            )

            left_work_grad = left_grad.detach().to(dtype=work_dtype) / sqrt_scale
            right_work_grad = right_grad.detach().to(dtype=work_dtype) / sqrt_scale
            _validate_finite_pair(
                left_work_grad,
                name=f"factor pair {index} left gradient",
                other=right_work_grad,
                other_name=f"factor pair {index} right gradient",
            )
            left_momentum = _next_momentum(
                self.state.get(left, {}),
                left_work_grad,
                momentum=momentum,
            )
            right_momentum_value = _next_momentum(
                self.state.get(right, {}),
                right_work_grad,
                momentum=momentum,
            )
            right_momentum = right_momentum_value.mT

            left_increment = (
                -lr
                / 2.0
                * _matrix_sign(left_momentum @ right_invroot)
                @ right_invroot
            )
            right_increment = (
                -lr
                / 2.0
                * _matrix_sign(right_momentum @ left_invroot)
                @ left_invroot
            )

            decay_scale = math.sqrt(1.0 - lr * weight_decay)
            increment_scale = 1.0 / (sqrt_scale * decay_scale)
            next_left = (
                left_value * decay_scale + left_increment * increment_scale
            ).to(dtype=left.dtype)
            next_right = (
                right_value * decay_scale + right_increment.mT * increment_scale
            ).to(dtype=right.dtype)
            _inverse_gram_root(
                next_left,
                name=f"factor pair {index} candidate left",
            )
            _inverse_gram_root(
                next_right.mT,
                name=f"factor pair {index} candidate right.T",
            )

            updates.append(
                (
                    left,
                    right,
                    next_left,
                    next_right,
                    left_momentum,
                    right_momentum_value,
                )
            )

        for left, right, next_left, next_right, left_momentum, right_momentum in updates:
            left.copy_(next_left)
            right.copy_(next_right)
            self.state[left]["momentum_buffer"] = left_momentum
            self.state[right]["momentum_buffer"] = right_momentum

        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_momenta: list[torch.Tensor | None] = []

        def capture_momenta(
            optimizer: Optimizer,
            loaded_state_dict: dict[str, Any],
        ) -> None:
            del optimizer
            captured_momenta = _saved_momenta(loaded_state_dict)
            self._validate_saved_momenta(captured_momenta)
            saved_momenta[:] = captured_momenta

        def restore_momenta(optimizer: Optimizer) -> None:
            del optimizer
            self._restore_momenta(saved_momenta)

        pre_handle = self.register_load_state_dict_pre_hook(capture_momenta)
        post_handle = self.register_load_state_dict_post_hook(restore_momenta, prepend=True)
        try:
            super().load_state_dict(state_dict)
        finally:
            pre_handle.remove()
            post_handle.remove()

    def _validate_saved_momenta(
        self,
        saved_momenta: Sequence[torch.Tensor | None],
    ) -> None:
        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if len(saved_momenta) != len(parameters):
            return

        for pair_index in range(len(self.param_groups)):
            pair_offset = pair_index * 2
            pair_momenta = saved_momenta[pair_offset : pair_offset + 2]
            if (pair_momenta[0] is None) != (pair_momenta[1] is None):
                raise ValueError(
                    f"LoRA-Muon checkpoint factor pair {pair_index} must contain "
                    "both momentum buffers or neither."
                )

        for parameter, saved_momentum in zip(parameters, saved_momenta, strict=False):
            if saved_momentum is None:
                continue
            if saved_momentum.shape != parameter.shape:
                raise ValueError(
                    "LoRA-Muon checkpoint momentum buffer shape must match its factor."
                )
            if not saved_momentum.is_floating_point():
                raise TypeError("LoRA-Muon checkpoint momentum buffers must be floating point.")
            _validate_finite(saved_momentum, name="LoRA-Muon checkpoint momentum buffer")

    def _restore_momenta(self, saved_momenta: Sequence[torch.Tensor | None]) -> None:
        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if len(saved_momenta) != len(parameters):
            return
        for parameter, saved_momentum in zip(parameters, saved_momenta, strict=False):
            if not isinstance(parameter, nn.Parameter) or saved_momentum is None:
                continue
            self.state[parameter]["momentum_buffer"] = saved_momentum.detach().to(
                device=parameter.device,
                dtype=_work_dtype(parameter.dtype),
                copy=True,
            )

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._allow_add_param_group:
            super().add_param_group(param_group)
            return
        raise RuntimeError(
            "LoRAMuon does not support add_param_group(); construct it with all factor pairs."
        )


def _validate_pairs(pairs: Sequence[LoRAFactorPair]) -> tuple[LoRAFactorPair, ...]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError("pairs must be a sequence of LoRAFactorPair instances.")
    if not pairs:
        raise ValueError("pairs must contain at least one LoRAFactorPair.")

    parameter_ids: set[int] = set()
    validated: list[LoRAFactorPair] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, LoRAFactorPair):
            raise TypeError(f"pairs[{index}] must be a LoRAFactorPair.")
        _validate_pair(pair, index=index, parameter_ids=parameter_ids)
        validated.append(pair)
    return tuple(validated)


def _validate_pair(
    pair: LoRAFactorPair,
    *,
    index: int,
    parameter_ids: set[int],
) -> None:
    for name, parameter in (("left", pair.left), ("right", pair.right)):
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(f"pairs[{index}].{name} must be an nn.Parameter.")
        if parameter.ndim != 2:
            raise ValueError(f"pairs[{index}].{name} must be two-dimensional.")
        if not parameter.is_floating_point():
            raise TypeError(f"pairs[{index}].{name} must have a floating-point dtype.")
        if not parameter.requires_grad:
            raise ValueError(f"pairs[{index}].{name} must require gradients.")
        if id(parameter) in parameter_ids:
            raise ValueError("LoRA-Muon factor parameters must be unique across pairs.")
        parameter_ids.add(id(parameter))

    if pair.left.shape[1] != pair.right.shape[0]:
        raise ValueError(
            f"pairs[{index}] has incompatible rank dimensions: "
            f"left.shape={tuple(pair.left.shape)}, right.shape={tuple(pair.right.shape)}."
        )
    if pair.left.shape[1] == 0:
        raise ValueError(f"pairs[{index}] rank dimension must be positive.")
    if pair.left.device != pair.right.device:
        raise ValueError(f"pairs[{index}] factors must be on the same device.")
    if pair.left.dtype != pair.right.dtype:
        raise ValueError(f"pairs[{index}] factors must have the same dtype.")


def _group_factors(group: dict[str, Any], *, index: int) -> tuple[nn.Parameter, nn.Parameter]:
    parameters = group["params"]
    if len(parameters) != 2:
        raise RuntimeError(f"LoRA-Muon param group {index} must contain exactly two factors.")
    left, right = parameters
    if not isinstance(left, nn.Parameter) or not isinstance(right, nn.Parameter):
        raise RuntimeError(f"LoRA-Muon param group {index} must contain nn.Parameter factors.")
    return left, right


def _next_momentum(
    state: dict[str, Any],
    gradient: torch.Tensor,
    *,
    momentum: float,
) -> torch.Tensor:
    buffer = state.get("momentum_buffer")
    if buffer is None:
        previous = torch.zeros_like(gradient)
    elif not isinstance(buffer, torch.Tensor) or buffer.shape != gradient.shape:
        raise RuntimeError("Invalid LoRA-Muon momentum buffer in optimizer state.")
    else:
        previous = buffer.to(device=gradient.device, dtype=gradient.dtype)

    if momentum == 0.0:
        return gradient.clone()
    return previous * momentum + gradient * (1.0 - momentum)


def _saved_momenta(state_dict: Mapping[str, Any]) -> list[torch.Tensor | None]:
    saved_state = state_dict.get("state")
    saved_groups = state_dict.get("param_groups")
    if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, Sequence):
        return []

    buffers: list[torch.Tensor | None] = []
    for group in saved_groups:
        if not isinstance(group, Mapping):
            return []
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, Sequence):
            return []
        for parameter_id in parameter_ids:
            if parameter_id not in saved_state:
                buffers.append(None)
                continue
            parameter_state = saved_state[parameter_id]
            if not isinstance(parameter_state, Mapping):
                raise TypeError("LoRA-Muon checkpoint parameter state must be a mapping.")
            if "momentum_buffer" not in parameter_state:
                buffers.append(None)
                continue
            buffer = parameter_state["momentum_buffer"]
            if not isinstance(buffer, torch.Tensor):
                raise TypeError("LoRA-Muon checkpoint momentum buffers must be tensors.")
            buffers.append(buffer)
    return buffers


def _inverse_gram_root(factor: torch.Tensor, *, name: str) -> torch.Tensor:
    work = factor.detach().to(dtype=_work_dtype(factor.dtype))
    _validate_finite(work, name=name)

    gram = work.mT @ work
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = eigenvalues[-1]
    tolerance = largest * max(factor.shape) * torch.finfo(work.dtype).eps
    full_rank = (largest > 0) & (eigenvalues[0] > tolerance)
    if not bool(full_rank):
        raise ValueError(
            f"{name} must be numerically full column rank for paper-faithful LoRA-Muon."
        )

    inverse_sqrt = eigenvalues.rsqrt()
    return (eigenvectors * inverse_sqrt.unsqueeze(0)) @ eigenvectors.mT


def _matrix_sign(matrix: torch.Tensor) -> torch.Tensor:
    _validate_finite(matrix, name="LoRA-Muon momentum")
    left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        # The paper's Newton-Schulz realization fixes an all-zero input at zero.
        return torch.zeros_like(matrix)
    # Keep the zero-input branch on device.  Calling bool() here would force a
    # host synchronization for every factor pair even though the SVD already
    # produced the predicate needed to mask the result.
    return (left @ right) * (singular_values[0] > 0)


def _validate_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_finite_pair(
    value: torch.Tensor,
    *,
    name: str,
    other: torch.Tensor,
    other_name: str,
) -> None:
    flags = torch.stack(
        (
            torch.isfinite(value).all(),
            torch.isfinite(other).all(),
        )
    )
    if bool(flags.all()):
        return
    if not bool(flags[0]):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool(flags[1]):
        raise ValueError(f"{other_name} must contain only finite values.")


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _option(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result < minimum or (maximum is not None and result >= maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}.")
        raise ValueError(f"{name} must be in [{minimum}, {maximum}).")
    return result


def _validate_decay(lr: float, weight_decay: float) -> None:
    if lr * weight_decay >= 1.0:
        raise ValueError("LoRA-Muon requires lr * weight_decay < 1 for split weight decay.")


__all__ = ["LoRAFactorPair", "LoRAMuon"]
