from __future__ import annotations

from typing import Literal

from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from .shape import SizeLike, scalar_1d, size_1d

PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
TorchPaddingMode = Literal["constant", "reflect", "replicate", "circular"]


def torch_padding_mode(mode: PaddingMode) -> TorchPaddingMode:
    return "constant" if mode == "zeros" else mode


def pad_conv1d_input(
    x: Tensor,
    *,
    padding: int,
    causal: bool,
    padding_mode: PaddingMode,
) -> Tensor:
    if padding == 0:
        return x
    mode = torch_padding_mode(padding_mode)
    left_padding = 2 * padding if causal else padding
    right_padding = 0 if causal else padding
    return F.pad(x, (left_padding, right_padding), mode=mode)


def pad_context_1d(
    x: Tensor,
    *,
    left: int,
    right: int,
    padding_mode: PaddingMode,
) -> Tensor:
    if left < 0 or right < 0:
        raise ValueError(f"context padding must be non-negative, got left={left}, right={right}.")
    if left == 0 and right == 0:
        return x
    return F.pad(x, (left, right), mode=torch_padding_mode(padding_mode))


def pad_tail_to_multiple_1d(
    x: Tensor,
    *,
    multiple: int,
    padding_mode: PaddingMode,
) -> Tensor:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}.")
    remainder = x.size(-1) % multiple
    if remainder == 0:
        return x
    tail_padding = multiple - remainder
    return F.pad(x, (0, tail_padding), mode=torch_padding_mode(padding_mode))


def pad_tail_to_length_1d(
    x: Tensor,
    *,
    length: int,
    padding_mode: PaddingMode,
) -> Tensor:
    if length < x.size(-1):
        raise ValueError(
            f"length must be at least input length: got length={length}, input={x.size(-1)}."
        )
    tail_padding = length - x.size(-1)
    if tail_padding == 0:
        return x
    return F.pad(x, (0, tail_padding), mode=torch_padding_mode(padding_mode))


def unfold_complete_windows_1d(x: Tensor, *, window_size: int, step: int) -> Tensor:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}.")
    if x.size(-1) < window_size:
        raise ValueError(
            f"input length must be at least window_size: got {x.size(-1)} and {window_size}."
        )
    remainder = (x.size(-1) - window_size) % step
    if remainder != 0:
        raise ValueError(
            "input length must fit complete windows: "
            f"length={x.size(-1)}, window_size={window_size}, step={step}."
        )
    return x.unfold(dimension=-1, size=window_size, step=step)


def unfold_segments_1d(
    x: Tensor,
    *,
    segment_size: SizeLike,
    overlap: SizeLike | None,
    padding_mode: PaddingMode,
) -> Tensor:
    segment = scalar_1d(size_1d(segment_size, name="segment_size"), name="segment_size")
    if segment <= 0:
        raise ValueError(f"segment_size must be positive, got {segment}.")
    if overlap is None:
        window_size = segment
    else:
        overlap_value = scalar_1d(size_1d(overlap, name="overlap"), name="overlap")
        if overlap_value < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap_value}.")
        window_size = segment + overlap_value

    if x.size(-1) < window_size:
        tail_padding = window_size - x.size(-1)
    else:
        remainder = (x.size(-1) - window_size) % segment
        tail_padding = 0 if remainder == 0 else segment - remainder
    if tail_padding:
        x = F.pad(x, (0, tail_padding), mode=torch_padding_mode(padding_mode))
    return x.unfold(dimension=-1, size=window_size, step=segment)


def trim_1d(x: Tensor, length: int) -> Tensor:
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}.")
    return x[..., :length]


def fold_transposed_segments_1d(
    output: Tensor,
    *,
    num_segments: int,
    segment_size: int,
    stride: int,
    raw_segment_output_size: int,
    raw_full_output_size: int,
    padding: int,
) -> Tensor:
    output = rearrange(output, "(b k) c t -> b (c t) k", k=num_segments)
    folded = F.fold(
        output,
        output_size=(raw_full_output_size, 1),
        kernel_size=(raw_segment_output_size, 1),
        stride=(stride * segment_size, 1),
        padding=(padding, 0),
    )
    return folded.squeeze(-1)
