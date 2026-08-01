from __future__ import annotations

import torch
from einops import rearrange
from torch import Size, Tensor, nn
from torch.nn import functional as F

from ..routing import DenseExpertWeights
from .bank import (
    expand_expert_weights as _expand_expert_weights,
)
from .bank import (
    mix_expert_bias as _mix_expert_bias,
)
from .bank import (
    mix_expert_weights as _mix_expert_weights,
)
from .bank import (
    validate_channels as _validate_channels,
)
from .bank import (
    validate_expert_weights as _validate_expert_weights,
)
from .segment import PaddingMode
from .shape import (
    SizeLike,
    effective_kernel_size_2d,
    infer_padding_2d,
    pair_2d,
    size_2d,
    validate_conv2d_args,
)


class DynamicConv2d(nn.Module):
    router: nn.Module | None
    kernel_size: Size
    stride: Size
    dilation: Size
    padding: Size
    padding_mode: PaddingMode
    weight: nn.Parameter
    bias: nn.Parameter | None
    _effective_kernel_size: Size | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeLike,
        num_experts: int,
        *,
        stride: SizeLike = 1,
        dilation: SizeLike = 1,
        groups: int = 1,
        bias: bool = True,
        padding: SizeLike | None = None,
        padding_mode: PaddingMode = "zeros",
        router: nn.Module | None = None,
    ) -> None:
        super().__init__()
        _validate_channels(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            num_experts=num_experts,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.kernel_size = size_2d(kernel_size, name="kernel_size")
        self.stride = size_2d(stride, name="stride")
        self.dilation = size_2d(dilation, name="dilation")
        self.groups = groups
        if padding is None:
            padding = infer_padding_2d(
                effective_kernel_size_2d(self.kernel_size, self.dilation),
                self.stride,
            )
        self.padding = size_2d(padding, name="padding")
        if padding_mode != "zeros":
            raise ValueError("DynamicConv2d currently only supports padding_mode='zeros'.")
        self.padding_mode = padding_mode
        self.router = router
        self._effective_kernel_size = None

        validate_conv2d_args(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        self.weight = nn.Parameter(
            torch.empty(
                num_experts,
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_channels))
        else:
            self.register_parameter("bias", None)
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert_index in range(self.num_experts):
            nn.init.kaiming_normal_(self.weight[expert_index])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def effective_kernel_size(self) -> Size:
        effective_kernel_size = self._effective_kernel_size
        if effective_kernel_size is None:
            effective_kernel_size = effective_kernel_size_2d(self.kernel_size, self.dilation)
            self._effective_kernel_size = effective_kernel_size
        return effective_kernel_size

    def forward(self, x: Tensor) -> Tensor:
        _validate_input_2d(x, channels=self.in_channels, name="DynamicConv2d")
        expert_weights = self.compute_expert_weights(x)
        return self.apply_conv(x, expert_weights)

    def forward_manually(self, x: Tensor, expert_weights: Tensor | DenseExpertWeights) -> Tensor:
        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        _validate_input_2d(x, channels=self.in_channels, name="DynamicConv2d")
        return self.apply_conv(x, expert_weights)

    def compute_expert_weights(self, x: Tensor) -> Tensor:
        router = self.router
        if router is None:
            raise ValueError("router must be provided when calling DynamicConv2d.forward().")
        expert_weights = router(x)
        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        return expert_weights.reshape(-1, self.num_experts)

    def apply_conv(self, x: Tensor, expert_weights: Tensor | DenseExpertWeights) -> Tensor:
        expert_weight_tensor: Tensor = _validate_expert_weights(
            expert_weights,
            num_experts=self.num_experts,
        )
        expert_weight_tensor = _expand_expert_weights(expert_weight_tensor, batch_size=x.size(0))
        weight = _mix_expert_weights(expert_weight_tensor, self.weight)
        x = rearrange(x, "b c h w -> 1 (b c) h w")
        weight = rearrange(weight, "b o c kh kw -> (b o) c kh kw")

        output = F.conv2d(
            x,
            weight,
            bias=None,
            stride=pair_2d(self.stride, name="stride"),
            padding=pair_2d(self.padding, name="padding"),
            dilation=pair_2d(self.dilation, name="dilation"),
            groups=expert_weight_tensor.size(0) * self.groups,
        )
        output = rearrange(output, "1 (b o) h w -> b o h w", b=expert_weight_tensor.size(0))

        if self.bias is not None:
            bias = _mix_expert_bias(expert_weight_tensor, self.bias)
            output = output + bias[..., None, None]
        return output


def _validate_input_2d(x: Tensor, *, channels: int, name: str) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} expects input shape (B, C, H, W), got {tuple(x.shape)}.")
    if x.size(1) != channels:
        raise ValueError(f"{name} channel mismatch: got {x.size(1)}, expected {channels}.")
