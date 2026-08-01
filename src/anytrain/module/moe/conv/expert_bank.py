"""Grouped convolution helpers for parameter-bank MoE execution."""

from __future__ import annotations

from einops import rearrange
from torch import Tensor
from torch.nn import functional as F


def grouped_conv1d(
    cells: Tensor,
    weight: Tensor,
    *,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tensor:
    x = rearrange(cells, "b c t -> 1 (b c) t")
    conv_weight = rearrange(weight, "b o c k -> (b o) c k")
    output = F.conv1d(
        x,
        conv_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=cells.size(0) * groups,
    )
    return rearrange(output, "1 (b o) t -> b o t", b=cells.size(0))


def grouped_conv_transpose1d(
    cells: Tensor,
    weight: Tensor,
    *,
    stride: int,
    dilation: int,
    groups: int,
) -> Tensor:
    x = rearrange(cells, "b c t -> 1 (b c) t")
    conv_weight = rearrange(weight, "b i o k -> (b i) o k")
    output = F.conv_transpose1d(
        x,
        conv_weight,
        bias=None,
        stride=stride,
        padding=0,
        output_padding=0,
        dilation=dilation,
        groups=cells.size(0) * groups,
    )
    return rearrange(output, "1 (b o) t -> b o t", b=cells.size(0))


def grouped_conv2d(
    cells: Tensor,
    weight: Tensor,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> Tensor:
    x = rearrange(cells, "b c h w -> 1 (b c) h w")
    conv_weight = rearrange(weight, "b o c kh kw -> (b o) c kh kw")
    output = F.conv2d(
        x,
        conv_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=cells.size(0) * groups,
    )
    return rearrange(output, "1 (b o) h w -> b o h w", b=cells.size(0))


__all__ = ["grouped_conv1d", "grouped_conv2d", "grouped_conv_transpose1d"]
