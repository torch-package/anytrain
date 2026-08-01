from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Size, Tensor, nn
from torch.nn import functional as F

from ..routing import DenseExpertWeights
from ._expert import (
    expand_expert_weights as _expand_expert_weights,
)
from ._expert import (
    mix_expert_bias as _mix_expert_bias,
)
from ._expert import (
    mix_expert_weights as _mix_expert_weights,
)
from ._expert import (
    validate_channels as _validate_channels,
)
from ._expert import (
    validate_expert_weights as _validate_expert_weights,
)
from .segment import (
    PaddingMode,
    fold_transposed_segments_1d,
    pad_context_1d,
    pad_conv1d_input,
    pad_tail_to_length_1d,
    pad_tail_to_multiple_1d,
    trim_1d,
    unfold_complete_windows_1d,
    unfold_segments_1d,
)
from .shape import (
    SizeLike,
    effective_kernel_size_1d,
    infer_padding_1d,
    scalar_1d,
    size_1d,
    validate_conv1d_args,
    validate_dynamic_conv1d_args,
)


@dataclass(frozen=True)
class PreprocessCache:
    batch_size: int
    num_segments: tuple[int, ...]
    output_size: tuple[int, ...]


class DynamicConv1d(nn.Module):
    router: nn.Module | None
    kernel_size: Size
    stride: Size
    dilation: Size
    padding: Size
    padding_mode: PaddingMode
    segment_size: Size | None
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
        segment_size: SizeLike | None = None,
        causal: bool = False,
        boundary_aware: bool = True,
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
        self.kernel_size = size_1d(kernel_size, name="kernel_size")
        self.stride = size_1d(stride, name="stride")
        self.dilation = size_1d(dilation, name="dilation")
        self.groups = groups
        if padding is None:
            padding = infer_padding_1d(
                effective_kernel_size_1d(self.kernel_size, self.dilation),
                self.stride,
            )
        self.padding = size_1d(padding, name="padding")
        self.padding_mode = padding_mode
        self.segment_size = size_1d(segment_size, name="segment_size")
        self.causal = causal
        self.boundary_aware = boundary_aware
        self.router = router
        self._effective_kernel_size = None

        validate_dynamic_conv1d_args(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            segment_size=self.segment_size,
        )

        self.weight = nn.Parameter(
            torch.empty(num_experts, out_channels, in_channels // groups, self.kernel_size[0])
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
            effective_kernel_size = effective_kernel_size_1d(self.kernel_size, self.dilation)
            self._effective_kernel_size = effective_kernel_size
        return effective_kernel_size

    def forward(self, x: Tensor) -> Tensor:
        x, cache = self.preprocess(x)
        expert_weights = self.compute_expert_weights(x)
        output = self.apply_conv(x, expert_weights)
        return self.postprocess(output, cache)

    def forward_manually(self, x: Tensor, expert_weights: Tensor | DenseExpertWeights) -> Tensor:
        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        x, cache = self.preprocess(x)
        expert_weights = _expand_expert_weights(
            expert_weights,
            batch_size=x.size(0),
            original_batch_size=cache.batch_size,
            num_segments=cache.num_segments[0],
        )
        output = self.apply_conv(x, expert_weights)
        return self.postprocess(output, cache)

    def compute_expert_weights(self, x: Tensor) -> Tensor:
        router = self.router
        if router is None:
            raise ValueError("router must be provided when calling DynamicConv1d.forward().")
        expert_weights = router(x)
        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        return expert_weights.reshape(-1, self.num_experts)

    def preprocess(self, x: Tensor) -> tuple[Tensor, PreprocessCache]:
        _validate_input_1d(x, channels=self.in_channels, name="DynamicConv1d")
        batch_size = x.size(0)
        raw_input_length = x.size(-1)
        padding = scalar_1d(self.padding, name="padding")

        segment_size_config = self.segment_size
        if segment_size_config is None:
            x = pad_conv1d_input(
                x,
                padding=padding,
                causal=self.causal,
                padding_mode=self.padding_mode,
            )
            output_size = (self._conv_output_length(x.size(-1), conv_padding=0),)
            return x, PreprocessCache(
                batch_size=batch_size,
                num_segments=(1,),
                output_size=output_size,
            )

        segment_size = scalar_1d(segment_size_config, name="segment_size")
        if self.boundary_aware:
            num_segments = (raw_input_length + segment_size - 1) // segment_size
            padded_input_length = num_segments * segment_size
            x = pad_context_1d(
                x,
                left=self._left_context_size(),
                right=self._right_context_size(),
                padding_mode=self.padding_mode,
            )
            x = pad_tail_to_length_1d(
                x,
                length=padded_input_length
                + scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
                - 1,
                padding_mode="zeros",
            )
            output_size = (
                self._conv_output_length(self._padded_input_length(raw_input_length), conv_padding=0),
            )
            segments = unfold_complete_windows_1d(
                x,
                window_size=segment_size
                + scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
                - 1,
                step=segment_size,
            )
        else:
            x = pad_tail_to_multiple_1d(
                x,
                multiple=segment_size,
                padding_mode="zeros",
            )
            output_size = (self._conv_output_length(raw_input_length, conv_padding=padding),)
            segments = unfold_complete_windows_1d(
                x,
                window_size=segment_size,
                step=segment_size,
            )
        num_segments = int(segments.size(2))
        x = rearrange(segments, "b c k t -> (b k) c t")
        if self.causal and not self.boundary_aware:
            x = pad_context_1d(
                x,
                left=self._left_context_size(),
                right=0,
                padding_mode="zeros",
            )
        return x, PreprocessCache(
            batch_size=batch_size,
            num_segments=(num_segments,),
            output_size=output_size,
        )

    def apply_conv(self, x: Tensor, expert_weights: Tensor | DenseExpertWeights) -> Tensor:
        expert_weight_tensor: Tensor = _validate_expert_weights(
            expert_weights,
            num_experts=self.num_experts,
        )
        expert_weight_tensor = _expand_expert_weights(expert_weight_tensor, batch_size=x.size(0))
        weight = _mix_expert_weights(expert_weight_tensor, self.weight)
        x = rearrange(x, "b c t -> 1 (b c) t")
        weight = rearrange(weight, "b o c k -> (b o) c k")

        conv_padding = (
            scalar_1d(self.padding, name="padding")
            if self.segment_size is not None and not self.boundary_aware and not self.causal
            else 0
        )
        output = F.conv1d(
            x,
            weight,
            bias=None,
            stride=scalar_1d(self.stride, name="stride"),
            padding=conv_padding,
            dilation=scalar_1d(self.dilation, name="dilation"),
            groups=expert_weight_tensor.size(0) * self.groups,
        )
        output = rearrange(output, "1 (b o) t -> b o t", b=expert_weight_tensor.size(0))

        if self.bias is not None:
            bias = _mix_expert_bias(expert_weight_tensor, self.bias)
            output = output + bias[..., None]
        return output

    def postprocess(self, output: Tensor, cache: PreprocessCache) -> Tensor:
        if self.segment_size is None:
            return trim_1d(output, cache.output_size[0])

        num_segments = cache.num_segments[0]
        output = rearrange(output, "(b k) c t -> b c (k t)", k=num_segments)
        return trim_1d(output, cache.output_size[0])

    def _conv_output_length(self, input_length: int, *, conv_padding: int) -> int:
        effective_kernel = scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
        stride = scalar_1d(self.stride, name="stride")
        return (input_length + 2 * conv_padding - effective_kernel) // stride + 1

    def _left_context_size(self) -> int:
        if self.causal:
            return scalar_1d(self.effective_kernel_size, name="effective_kernel_size") - 1
        return scalar_1d(self.padding, name="padding")

    def _right_context_size(self) -> int:
        if self.causal:
            return 0
        effective_kernel = scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
        return effective_kernel - 1 - scalar_1d(self.padding, name="padding")

    def _padded_input_length(self, grouped_length: int) -> int:
        # Output shape follows PyTorch Conv1d with this module's configured padding.
        padding = scalar_1d(self.padding, name="padding")
        return grouped_length + 2 * padding


class DynamicConvTranspose1d(nn.Module):
    router: nn.Module | None
    kernel_size: Size
    stride: Size
    dilation: Size
    padding: Size
    output_padding: Size
    segment_size: Size | None
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
        output_padding: SizeLike = 0,
        segment_size: SizeLike | None = None,
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
        self.kernel_size = size_1d(kernel_size, name="kernel_size")
        self.stride = size_1d(stride, name="stride")
        self.dilation = size_1d(dilation, name="dilation")
        self.groups = groups
        if padding is None:
            padding = infer_padding_1d(
                effective_kernel_size_1d(self.kernel_size, self.dilation),
                self.stride,
            )
        self.padding = size_1d(padding, name="padding")
        self.output_padding = size_1d(output_padding, name="output_padding")
        self.segment_size = size_1d(segment_size, name="segment_size")
        self.router = router
        self._effective_kernel_size = None

        validate_conv1d_args(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        _validate_output_padding(
            output_padding=scalar_1d(self.output_padding, name="output_padding"),
            stride=scalar_1d(self.stride, name="stride"),
            dilation=scalar_1d(self.dilation, name="dilation"),
        )
        segment_size_config = self.segment_size
        if segment_size_config is not None and segment_size_config[0] <= 0:
            raise ValueError(f"segment_size must be positive, got {segment_size_config[0]}.")

        self.weight = nn.Parameter(
            torch.empty(num_experts, in_channels, out_channels // groups, self.kernel_size[0])
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
            effective_kernel_size = effective_kernel_size_1d(self.kernel_size, self.dilation)
            self._effective_kernel_size = effective_kernel_size
        return effective_kernel_size

    def forward(self, x: Tensor) -> Tensor:
        x, cache = self.preprocess(x)
        expert_weights = self.compute_expert_weights(x)
        output = self.apply_conv(x, expert_weights)
        output = self.postprocess(output, cache)
        return self.apply_bias(output, expert_weights, cache)

    def forward_manually(self, x: Tensor, expert_weights: Tensor | DenseExpertWeights) -> Tensor:
        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        x, cache = self.preprocess(x)
        expert_weights = _expand_expert_weights(
            expert_weights,
            batch_size=x.size(0),
            original_batch_size=cache.batch_size,
            num_segments=cache.num_segments[0],
        )
        output = self.apply_conv(x, expert_weights)
        output = self.postprocess(output, cache)
        return self.apply_bias(output, expert_weights, cache)

    def compute_expert_weights(self, x: Tensor) -> Tensor:
        router = self.router
        if router is None:
            raise ValueError(
                "router must be provided when calling DynamicConvTranspose1d.forward()."
            )
        expert_weights = router(x)
        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        return expert_weights.reshape(-1, self.num_experts)

    def preprocess(self, x: Tensor) -> tuple[Tensor, PreprocessCache]:
        _validate_input_1d(x, channels=self.in_channels, name="DynamicConvTranspose1d")
        batch_size = x.size(0)
        output_size = (self._output_length(x.size(-1)),)
        segment_size_config = self.segment_size
        if segment_size_config is None:
            return x, PreprocessCache(
                batch_size=batch_size,
                num_segments=(1,),
                output_size=output_size,
            )

        segments = unfold_segments_1d(
            x,
            segment_size=segment_size_config,
            overlap=None,
            padding_mode="zeros",
        )
        num_segments = int(segments.size(2))
        x = rearrange(segments, "b c k t -> (b k) c t")
        return x, PreprocessCache(
            batch_size=batch_size,
            num_segments=(num_segments,),
            output_size=output_size,
        )

    def apply_conv(self, x: Tensor, expert_weights: Tensor | DenseExpertWeights) -> Tensor:
        expert_weight_tensor: Tensor = _validate_expert_weights(
            expert_weights,
            num_experts=self.num_experts,
        )
        expert_weight_tensor = _expand_expert_weights(expert_weight_tensor, batch_size=x.size(0))
        weight = _mix_expert_weights(expert_weight_tensor, self.weight)
        x = rearrange(x, "b c t -> 1 (b c) t")
        weight = rearrange(weight, "b i o k -> (b i) o k")

        output = F.conv_transpose1d(
            x,
            weight,
            bias=None,
            stride=scalar_1d(self.stride, name="stride"),
            padding=0,
            output_padding=0,
            dilation=scalar_1d(self.dilation, name="dilation"),
            groups=expert_weight_tensor.size(0) * self.groups,
        )
        output = rearrange(output, "1 (b o) t -> b o t", b=expert_weight_tensor.size(0))

        return output

    def postprocess(self, output: Tensor, cache: PreprocessCache) -> Tensor:
        padding = scalar_1d(self.padding, name="padding")
        output_padding = scalar_1d(self.output_padding, name="output_padding")

        segment_size_config = self.segment_size
        if segment_size_config is None:
            end = output.size(-1) - padding + output_padding
            return output[..., padding:end]

        num_segments = cache.num_segments[0]
        segment_size = scalar_1d(segment_size_config, name="segment_size")
        folded = fold_transposed_segments_1d(
            output,
            num_segments=num_segments,
            segment_size=segment_size,
            stride=scalar_1d(self.stride, name="stride"),
            raw_segment_output_size=self._raw_output_length(segment_size),
            raw_full_output_size=self._raw_output_length(num_segments * segment_size),
            padding=padding,
        )
        return trim_1d(folded, cache.output_size[0])

    def apply_bias(
        self,
        output: Tensor,
        expert_weights: Tensor | DenseExpertWeights,
        cache: PreprocessCache,
    ) -> Tensor:
        if self.bias is None:
            return output

        expert_weights = _validate_expert_weights(expert_weights, num_experts=self.num_experts)
        num_segments = cache.num_segments[0]
        expert_weights = _expand_expert_weights(
            expert_weights,
            batch_size=cache.batch_size * num_segments,
        )
        bias = _mix_expert_bias(expert_weights, self.bias)
        if num_segments == 1:
            return output + bias[..., None]

        segment_size_config = self.segment_size
        if segment_size_config is None:
            raise RuntimeError("segment_size must be set when cache has multiple segments.")
        segment_size = scalar_1d(segment_size_config, name="segment_size")
        padding = scalar_1d(self.padding, name="padding")
        raw_segment_output_size = self._raw_output_length(segment_size)
        raw_full_output_size = self._raw_output_length(num_segments * segment_size)
        segment_bias = bias[..., None].expand(-1, -1, raw_segment_output_size)
        folded_bias = fold_transposed_segments_1d(
            segment_bias,
            num_segments=num_segments,
            segment_size=segment_size,
            stride=scalar_1d(self.stride, name="stride"),
            raw_segment_output_size=raw_segment_output_size,
            raw_full_output_size=raw_full_output_size,
            padding=padding,
        )
        coverage = fold_transposed_segments_1d(
            segment_bias.new_ones((num_segments, 1, raw_segment_output_size)),
            num_segments=num_segments,
            segment_size=segment_size,
            stride=scalar_1d(self.stride, name="stride"),
            raw_segment_output_size=raw_segment_output_size,
            raw_full_output_size=raw_full_output_size,
            padding=padding,
        )
        folded_bias = trim_1d(folded_bias, cache.output_size[0])
        coverage = trim_1d(coverage, cache.output_size[0]).clamp_min(1)
        return output + folded_bias / coverage

    def _raw_output_length(self, input_length: int) -> int:
        stride = scalar_1d(self.stride, name="stride")
        effective_kernel = scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
        return (input_length - 1) * stride + effective_kernel

    def _output_length(self, input_length: int) -> int:
        padding = scalar_1d(self.padding, name="padding")
        output_padding = scalar_1d(self.output_padding, name="output_padding")
        return self._raw_output_length(input_length) - 2 * padding + output_padding


def _validate_input_1d(x: Tensor, *, channels: int, name: str) -> None:
    if x.ndim != 3:
        raise ValueError(f"{name} expects input shape (B, C, T), got {tuple(x.shape)}.")
    if x.size(1) != channels:
        raise ValueError(f"{name} channel mismatch: got {x.size(1)}, expected {channels}.")


def _validate_output_padding(*, output_padding: int, stride: int, dilation: int) -> None:
    if output_padding < 0:
        raise ValueError(f"output_padding must be non-negative, got {output_padding}.")
    if output_padding >= stride and output_padding >= dilation:
        raise ValueError(
            "output_padding must be smaller than either stride or dilation: "
            f"got output_padding={output_padding}, stride={stride}, dilation={dilation}."
        )
