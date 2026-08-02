"""Expert-major parameter helpers.

``expand_param`` is the reusable E-axis expander for any single-expert tensor.
``expand_expert`` is a narrow stock-only factory for ``nn.Linear`` /
``nn.Conv1d`` / ``nn.Conv2d`` / ``nn.ConvTranspose1d``; it is not a general
``nn.Module`` → :class:`Expert` compiler.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Size, Tensor, nn

from .conv.segment import PaddingMode
from .conv.shape import pair_2d, scalar_1d, size_1d, size_2d
from .expert import (
    Conv1dExpert,
    Conv2dExpert,
    ConvTranspose1dExpert,
    Expert,
    linear,
)
from .routing import validate_num_experts

ExpertInit = Literal["copy", "kaiming", "empty"]


def bank_weight_shape(
    module_or_weight: nn.Module | Tensor,
    num_experts: int,
) -> Size:
    """Return expert-major weight shape ``(num_experts, *single_expert_shape)``."""

    validate_num_experts(num_experts)
    single = _single_weight(module_or_weight)
    return Size((num_experts, *single.shape))


def expand_param(
    single_weight: Tensor,
    num_experts: int,
    *,
    init: ExpertInit = "copy",
) -> nn.Parameter:
    """Expand a single-expert parameter tensor to expert-major ``Parameter``."""

    validate_num_experts(num_experts)
    _validate_init(init)
    if single_weight.ndim < 1:
        raise ValueError(
            "single_weight must be a non-scalar tensor, "
            f"got shape {tuple(single_weight.shape)}."
        )
    bank = torch.empty(
        (num_experts, *single_weight.shape),
        dtype=single_weight.dtype,
        device=single_weight.device,
    )
    if init == "empty":
        return nn.Parameter(bank)
    if init == "copy":
        with torch.no_grad():
            bank.copy_(single_weight.detach().unsqueeze(0).expand_as(bank))
        return nn.Parameter(bank)
    with torch.no_grad():
        for expert_index in range(num_experts):
            _kaiming_weight_(bank[expert_index])
    return nn.Parameter(bank)


def expand_expert(
    module: nn.Module,
    num_experts: int,
    *,
    init: ExpertInit = "copy",
) -> tuple[Expert, nn.Parameter, nn.Parameter | None]:
    """Stock-only helper: copy geometry + expand params from Linear/Conv*.

    Supported modules: ``nn.Linear``, ``nn.Conv1d`` (incl. ``moe.conv.Conv1d``),
    ``nn.Conv2d``, ``nn.ConvTranspose1d``. Custom ``nn.Module`` graphs are
    rejected; write an :class:`Expert` and call :func:`expand_param` instead.

    ``weight`` / ``bias`` are expert-major Parameters shaped
    ``(num_experts, *module.weight.shape)`` and
    ``(num_experts, *module.bias.shape)`` (or ``bias is None``).

    With ``init="copy"``, every expert copies the template parameters so dense
    mixtures and sparse one-hot routes match the template op output.
    """

    validate_num_experts(num_experts)
    _validate_init(init)
    expert = expert_from_module(module)
    weight_tensor = getattr(module, "weight", None)
    if not isinstance(weight_tensor, Tensor):
        raise TypeError(
            "expand_expert requires a module with a weight Tensor, "
            f"got {type(module)!r}."
        )
    weight = expand_param(weight_tensor, num_experts, init=init)
    bias_tensor = getattr(module, "bias", None)
    if bias_tensor is None:
        return expert, weight, None
    if not isinstance(bias_tensor, Tensor):
        raise TypeError(
            "expand_expert requires module.bias to be a Tensor or None, "
            f"got {type(bias_tensor)!r}."
        )
    bias_param = expand_param(bias_tensor, num_experts, init=_bias_init(init))
    if init == "kaiming":
        with torch.no_grad():
            bias_param.zero_()
    return expert, weight, bias_param


def expert_from_module(module: nn.Module) -> Expert:
    """Map a stock Linear/Conv module to geometry. Not for custom modules."""

    if isinstance(module, nn.Linear):
        return linear
    if isinstance(module, nn.Conv1d):
        return _conv1d_expert(module)
    if isinstance(module, nn.ConvTranspose1d):
        return _conv_transpose1d_expert(module)
    if isinstance(module, nn.Conv2d):
        return _conv2d_expert(module)
    raise TypeError(
        "expand_expert supports nn.Linear, nn.Conv1d, nn.Conv2d, "
        f"nn.ConvTranspose1d (and moe.conv.Conv1d); got {type(module)!r}."
    )


def _bias_init(init: ExpertInit) -> ExpertInit:
    # Bias has no kaiming geometry; copy/empty stay, kaiming zeros after expand.
    if init == "kaiming":
        return "empty"
    return init


def _validate_init(init: ExpertInit) -> None:
    if init not in {"copy", "kaiming", "empty"}:
        raise ValueError(f"init must be 'copy', 'kaiming', or 'empty', got {init!r}.")


def _single_weight(module_or_weight: nn.Module | Tensor) -> Tensor:
    if isinstance(module_or_weight, Tensor):
        return module_or_weight
    weight = getattr(module_or_weight, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError(
            "bank_weight_shape expects a Tensor or a module with a weight Tensor, "
            f"got {type(module_or_weight)!r}."
        )
    return weight


def _kaiming_weight_(weight: Tensor) -> None:
    if weight.ndim >= 2:
        nn.init.kaiming_normal_(weight)
    else:
        nn.init.normal_(weight)


def _conv1d_expert(module: nn.Conv1d) -> Conv1dExpert:
    padding_value = module.padding
    if isinstance(padding_value, str):
        raise TypeError(
            "expand_expert requires integer Conv1d.padding; "
            f"got padding={padding_value!r}."
        )
    padding = scalar_1d(size_1d(padding_value, name="padding"), name="padding")
    causal = bool(getattr(module, "causal", False))
    padding_mode = module.padding_mode
    if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
        raise ValueError(f"unsupported conv padding_mode {padding_mode!r}.")
    return Conv1dExpert(
        stride=scalar_1d(size_1d(module.stride, name="stride"), name="stride"),
        padding=padding,
        dilation=scalar_1d(size_1d(module.dilation, name="dilation"), name="dilation"),
        groups=module.groups,
        causal=causal,
        padding_mode=cast(PaddingMode, padding_mode),
    )


def _conv_transpose1d_expert(module: nn.ConvTranspose1d) -> ConvTranspose1dExpert:
    padding_value = module.padding
    if isinstance(padding_value, str):
        raise TypeError(
            "expand_expert requires integer ConvTranspose1d.padding; "
            f"got padding={padding_value!r}."
        )
    return ConvTranspose1dExpert(
        stride=scalar_1d(size_1d(module.stride, name="stride"), name="stride"),
        padding=scalar_1d(size_1d(padding_value, name="padding"), name="padding"),
        output_padding=scalar_1d(
            size_1d(module.output_padding, name="output_padding"),
            name="output_padding",
        ),
        dilation=scalar_1d(size_1d(module.dilation, name="dilation"), name="dilation"),
        groups=module.groups,
    )


def _conv2d_expert(module: nn.Conv2d) -> Conv2dExpert:
    if isinstance(module.padding, str):
        raise TypeError(
            "expand_expert requires integer Conv2d.padding; "
            f"got padding={module.padding!r}."
        )
    return Conv2dExpert(
        stride=pair_2d(size_2d(module.stride, name="stride"), name="stride"),
        padding=pair_2d(size_2d(module.padding, name="padding"), name="padding"),
        dilation=pair_2d(size_2d(module.dilation, name="dilation"), name="dilation"),
        groups=module.groups,
    )


__all__ = [
    "ExpertInit",
    "bank_weight_shape",
    "expand_expert",
    "expand_param",
    "expert_from_module",
]
