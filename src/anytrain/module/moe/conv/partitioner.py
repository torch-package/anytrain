"""Convolutional grid partitioners owned by MoE, not expert banks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Size, Tensor

from ..grid import Grid1dContext
from ..routing import broadcast_routing_mask
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
    scalar_1d,
    size_1d,
)


@dataclass(eq=False)
class Conv1dGridLayout:
    mask: Tensor | None
    batch_size: int
    num_segments: int
    output_length: int
    segmented: bool

    def restore(self, cell_outputs: Tensor) -> Tensor:
        if not self.segmented:
            return trim_1d(cell_outputs, self.output_length)
        output = rearrange(cell_outputs, "(b k) c t -> b c (k t)", k=self.num_segments)
        return trim_1d(output, self.output_length)


def conv1d_execution_padding(
    *,
    cell_size: Size | None,
    context: Grid1dContext,
    causal: bool,
    padding: SizeLike,
) -> int:
    """Padding applied inside the conv expert bank after partitioner preprocess."""

    padding_size = size_1d(padding, name="padding")
    if cell_size is not None and context == "local" and not causal:
        return scalar_1d(padding_size, name="padding")
    return 0


class Conv1dGridPartitioner:
    """Partition ``(B, C, T)`` inputs into conv execution cells for MoE."""

    def __init__(
        self,
        *,
        kernel_size: SizeLike,
        stride: SizeLike = 1,
        dilation: SizeLike = 1,
        padding: SizeLike = 0,
        fill: PaddingMode = "zeros",
        cell_size: SizeLike | None = None,
        causal: bool = False,
        context: Grid1dContext = "global",
        in_channels: int | None = None,
    ) -> None:
        self.kernel_size = size_1d(kernel_size, name="kernel_size")
        self.stride = size_1d(stride, name="stride")
        self.dilation = size_1d(dilation, name="dilation")
        self.padding = size_1d(padding, name="padding")
        if context not in {"global", "local"}:
            raise ValueError(f"unsupported Conv1dGridPartitioner.context {context!r}.")
        self.fill: PaddingMode = fill
        self.cell_size = None if cell_size is None else size_1d(cell_size, name="cell_size")
        self.causal = causal
        self.context = context
        self.in_channels = in_channels
        self._effective_kernel_size = effective_kernel_size_1d(self.kernel_size, self.dilation)

    @property
    def effective_kernel_size(self) -> Size:
        return self._effective_kernel_size

    def partition(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Conv1dGridLayout]:
        if x.ndim != 3:
            raise ValueError(f"Conv1dGridPartitioner expects (B, C, T), got {tuple(x.shape)}.")
        if self.in_channels is not None and x.size(1) != self.in_channels:
            raise ValueError(
                "input channel mismatch: "
                f"got {x.size(1)}, expected {self.in_channels}."
            )

        batch_size = x.size(0)
        raw_input_length = x.size(-1)
        padding = scalar_1d(self.padding, name="padding")
        cell_size_config = self.cell_size

        if cell_size_config is None:
            cells = pad_conv1d_input(
                x,
                padding=padding,
                causal=self.causal,
                padding_mode=self.fill,
            )
            output_length = self._conv_output_length(cells.size(-1), conv_padding=0)
            layout = Conv1dGridLayout(
                mask=_cell_mask(mask, batch_size=batch_size, num_segments=1, device=x.device),
                batch_size=batch_size,
                num_segments=1,
                output_length=output_length,
                segmented=False,
            )
            return cells, None, layout

        segment_size = scalar_1d(cell_size_config, name="cell_size")
        if self.context == "global":
            num_segments = (raw_input_length + segment_size - 1) // segment_size
            padded_input_length = num_segments * segment_size
            cells = pad_context_1d(
                x,
                left=self._left_context_size(),
                right=self._right_context_size(),
                padding_mode=self.fill,
            )
            cells = pad_tail_to_length_1d(
                cells,
                length=padded_input_length
                + scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
                - 1,
                padding_mode="zeros",
            )
            output_length = self._conv_output_length(
                self._padded_input_length(raw_input_length),
                conv_padding=0,
            )
            segments = unfold_complete_windows_1d(
                cells,
                window_size=segment_size
                + scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
                - 1,
                step=segment_size,
            )
        else:
            cells = pad_tail_to_multiple_1d(
                x,
                multiple=segment_size,
                padding_mode="zeros",
            )
            output_length = self._conv_output_length(raw_input_length, conv_padding=padding)
            segments = unfold_complete_windows_1d(
                cells,
                window_size=segment_size,
                step=segment_size,
            )
        num_segments = int(segments.size(2))
        cells = rearrange(segments, "b c k t -> (b k) c t")
        if self.causal and self.context == "local":
            cells = pad_context_1d(
                cells,
                left=self._left_context_size(),
                right=0,
                padding_mode="zeros",
            )
        layout = Conv1dGridLayout(
            mask=_cell_mask(
                mask,
                batch_size=batch_size,
                num_segments=num_segments,
                device=x.device,
            ),
            batch_size=batch_size,
            num_segments=num_segments,
            output_length=output_length,
            segmented=True,
        )
        return cells, None, layout

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
        padding = scalar_1d(self.padding, name="padding")
        return grouped_length + 2 * padding


@dataclass(eq=False)
class ConvTranspose1dGridLayout:
    mask: Tensor | None
    batch_size: int
    num_segments: int
    output_length: int
    cell_size: int | None
    padding: int
    output_padding: int
    stride: int
    raw_segment_output_size: int
    raw_full_output_size: int

    def restore(self, cell_outputs: Tensor) -> Tensor:
        if self.cell_size is None:
            end = cell_outputs.size(-1) - self.padding + self.output_padding
            return cell_outputs[..., self.padding : end]

        folded = fold_transposed_segments_1d(
            cell_outputs,
            num_segments=self.num_segments,
            segment_size=self.cell_size,
            stride=self.stride,
            raw_segment_output_size=self.raw_segment_output_size,
            raw_full_output_size=self.raw_full_output_size,
            padding=self.padding,
        )
        return trim_1d(folded, self.output_length)


class ConvTranspose1dGridPartitioner:
    """Partition ``(B, C, T)`` inputs for transposed dynamic convolution."""

    def __init__(
        self,
        *,
        kernel_size: SizeLike,
        stride: SizeLike = 1,
        dilation: SizeLike = 1,
        padding: SizeLike = 0,
        output_padding: SizeLike = 0,
        cell_size: SizeLike | None = None,
        in_channels: int | None = None,
    ) -> None:
        self.kernel_size = size_1d(kernel_size, name="kernel_size")
        self.stride = size_1d(stride, name="stride")
        self.dilation = size_1d(dilation, name="dilation")
        self.padding = size_1d(padding, name="padding")
        self.output_padding = size_1d(output_padding, name="output_padding")
        self.cell_size = None if cell_size is None else size_1d(cell_size, name="cell_size")
        self.in_channels = in_channels
        self._effective_kernel_size = effective_kernel_size_1d(self.kernel_size, self.dilation)

    @property
    def effective_kernel_size(self) -> Size:
        return self._effective_kernel_size

    def partition(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, ConvTranspose1dGridLayout]:
        if x.ndim != 3:
            raise ValueError(
                f"ConvTranspose1dGridPartitioner expects (B, C, T), got {tuple(x.shape)}."
            )
        if self.in_channels is not None and x.size(1) != self.in_channels:
            raise ValueError(
                "input channel mismatch: "
                f"got {x.size(1)}, expected {self.in_channels}."
            )

        batch_size = x.size(0)
        output_length = self._output_length(x.size(-1))
        padding = scalar_1d(self.padding, name="padding")
        output_padding = scalar_1d(self.output_padding, name="output_padding")
        stride = scalar_1d(self.stride, name="stride")
        cell_size_config = self.cell_size

        if cell_size_config is None:
            layout = ConvTranspose1dGridLayout(
                mask=_cell_mask(mask, batch_size=batch_size, num_segments=1, device=x.device),
                batch_size=batch_size,
                num_segments=1,
                output_length=output_length,
                cell_size=None,
                padding=padding,
                output_padding=output_padding,
                stride=stride,
                raw_segment_output_size=self._raw_output_length(x.size(-1)),
                raw_full_output_size=self._raw_output_length(x.size(-1)),
            )
            return x, None, layout

        segments = unfold_segments_1d(
            x,
            segment_size=cell_size_config,
            overlap=None,
            padding_mode="zeros",
        )
        num_segments = int(segments.size(2))
        cells = rearrange(segments, "b c k t -> (b k) c t")
        segment_size = scalar_1d(cell_size_config, name="cell_size")
        layout = ConvTranspose1dGridLayout(
            mask=_cell_mask(
                mask,
                batch_size=batch_size,
                num_segments=num_segments,
                device=x.device,
            ),
            batch_size=batch_size,
            num_segments=num_segments,
            output_length=output_length,
            cell_size=segment_size,
            padding=padding,
            output_padding=output_padding,
            stride=stride,
            raw_segment_output_size=self._raw_output_length(segment_size),
            raw_full_output_size=self._raw_output_length(num_segments * segment_size),
        )
        return cells, None, layout

    def _raw_output_length(self, input_length: int) -> int:
        stride = scalar_1d(self.stride, name="stride")
        effective_kernel = scalar_1d(self.effective_kernel_size, name="effective_kernel_size")
        return (input_length - 1) * stride + effective_kernel

    def _output_length(self, input_length: int) -> int:
        padding = scalar_1d(self.padding, name="padding")
        output_padding = scalar_1d(self.output_padding, name="output_padding")
        return self._raw_output_length(input_length) - 2 * padding + output_padding


@dataclass(eq=False)
class Conv2dLayout:
    mask: Tensor | None = None

    def restore(self, cell_outputs: Tensor) -> Tensor:
        return cell_outputs


class Conv2dIdentityPartitioner:
    """Pass-through partitioner for per-sample dynamic Conv2d."""

    def __init__(self, *, in_channels: int) -> None:
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        self.in_channels = in_channels

    def partition(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Conv2dLayout]:
        if x.ndim != 4:
            raise ValueError(
                f"Conv2dIdentityPartitioner expects (B, C, H, W), got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                "input channel mismatch: "
                f"got {x.size(1)}, expected {self.in_channels}."
            )
        return x, None, Conv2dLayout(
            mask=_cell_mask(mask, batch_size=x.size(0), num_segments=1, device=x.device)
        )


def _cell_mask(
    mask: Tensor | None,
    *,
    batch_size: int,
    num_segments: int,
    device: torch.device,
) -> Tensor | None:
    if mask is None:
        return None
    cell_count = batch_size * num_segments
    if mask.ndim == 1 and mask.numel() == cell_count:
        return broadcast_routing_mask(
            mask,
            leading_shape=(cell_count,),
            device=device,
            target_name="flattened convolution routing-cell shape",
        )
    return broadcast_routing_mask(
        mask,
        leading_shape=(batch_size, num_segments),
        device=device,
        target_name="convolution routing-cell shape",
    ).reshape(-1)


__all__ = [
    "Conv1dGridLayout",
    "Conv1dGridPartitioner",
    "Conv2dIdentityPartitioner",
    "Conv2dLayout",
    "ConvTranspose1dGridLayout",
    "ConvTranspose1dGridPartitioner",
    "conv1d_execution_padding",
]
