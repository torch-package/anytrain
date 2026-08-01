from __future__ import annotations

from collections.abc import Sequence
from typing import Union, overload

from torch import Size

from anytrain._compat import strict_zip

SizeLike = Union[int, Sequence[int], Size]


@overload
def size_1d(value: None, *, name: str) -> None: ...


@overload
def size_1d(value: SizeLike, *, name: str) -> Size: ...


def size_1d(value: SizeLike | None, *, name: str) -> Size | None:
    if value is None:
        return None
    if isinstance(value, int):
        return Size([value])
    if len(value) != 1:
        raise ValueError(f"{name} must have length 1, got {len(value)}.")
    return Size([int(value[0])])


@overload
def size_2d(value: None, *, name: str) -> None: ...


@overload
def size_2d(value: SizeLike, *, name: str) -> Size: ...


def size_2d(value: SizeLike | None, *, name: str) -> Size | None:
    if value is None:
        return None
    if isinstance(value, int):
        return Size([value, value])
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {len(value)}.")
    return Size([int(value[0]), int(value[1])])


def scalar_1d(value: Size, *, name: str) -> int:
    if len(value) != 1:
        raise ValueError(f"{name} must have length 1, got {len(value)}.")
    return int(value[0])


def pair_2d(value: Size, *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {len(value)}.")
    return int(value[0]), int(value[1])


def effective_kernel_size_1d(kernel_size: Size, dilation: Size) -> Size:
    kernel = scalar_1d(kernel_size, name="kernel_size")
    dilation_value = scalar_1d(dilation, name="dilation")
    return Size([(kernel - 1) * dilation_value + 1])


def effective_kernel_size_2d(kernel_size: Size, dilation: Size) -> Size:
    kernel_h, kernel_w = pair_2d(kernel_size, name="kernel_size")
    dilation_h, dilation_w = pair_2d(dilation, name="dilation")
    return Size(
        [
            (kernel_h - 1) * dilation_h + 1,
            (kernel_w - 1) * dilation_w + 1,
        ]
    )


def infer_padding_1d(effective_kernel_size: Size, stride: Size) -> Size:
    kernel = scalar_1d(effective_kernel_size, name="effective_kernel_size")
    stride_value = scalar_1d(stride, name="stride")
    if stride_value == 1:
        return Size([(kernel - 1) // 2])
    if stride_value % 2 == 0:
        return Size([stride_value // 2])
    raise ValueError(
        "automatic padding only supports stride=1 or an even stride: "
        f"got effective_kernel_size={kernel}, stride={stride_value}."
    )


def infer_padding_2d(effective_kernel_size: Size, stride: Size) -> Size:
    kernels = pair_2d(effective_kernel_size, name="effective_kernel_size")
    strides = pair_2d(stride, name="stride")
    return Size(
        [
            _infer_padding_axis(kernel, stride_value)
            for kernel, stride_value in strict_zip(kernels, strides)
        ]
    )


def validate_conv1d_args(
    *,
    kernel_size: Size,
    stride: Size,
    padding: Size,
    dilation: Size,
) -> None:
    kernel = scalar_1d(kernel_size, name="kernel_size")
    stride_value = scalar_1d(stride, name="stride")
    padding_value = scalar_1d(padding, name="padding")
    dilation_value = scalar_1d(dilation, name="dilation")
    if kernel <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel}.")
    if stride_value <= 0:
        raise ValueError(f"stride must be positive, got {stride_value}.")
    if padding_value < 0:
        raise ValueError(f"padding must be non-negative, got {padding_value}.")
    if dilation_value <= 0:
        raise ValueError(f"dilation must be positive, got {dilation_value}.")


def validate_conv2d_args(
    *,
    kernel_size: Size,
    stride: Size,
    padding: Size,
    dilation: Size,
) -> None:
    kernel_h, kernel_w = pair_2d(kernel_size, name="kernel_size")
    stride_h, stride_w = pair_2d(stride, name="stride")
    padding_h, padding_w = pair_2d(padding, name="padding")
    dilation_h, dilation_w = pair_2d(dilation, name="dilation")
    for name, value in (("kernel_size", kernel_h), ("kernel_size", kernel_w)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    for name, value in (("stride", stride_h), ("stride", stride_w)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    for name, value in (("padding", padding_h), ("padding", padding_w)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    for name, value in (("dilation", dilation_h), ("dilation", dilation_w)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")


def validate_dynamic_conv1d_args(
    *,
    kernel_size: Size,
    stride: Size,
    padding: Size,
    dilation: Size,
    segment_size: Size | None,
) -> None:
    validate_conv1d_args(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if segment_size is None:
        return

    segment = scalar_1d(segment_size, name="segment_size")
    if segment <= 0:
        raise ValueError(f"segment_size must be positive when provided, got {segment}.")

    stride_value = scalar_1d(stride, name="stride")
    padding_value = scalar_1d(padding, name="padding")
    effective_kernel = scalar_1d(
        effective_kernel_size_1d(kernel_size, dilation),
        name="effective_kernel_size",
    )

    if stride_value == 1:
        if effective_kernel % 2 == 0:
            raise ValueError(
                "DynamicConv1d with segment_size requires an odd effective kernel size "
                f"when stride=1, got {effective_kernel}."
            )
        expected_padding = (effective_kernel - 1) // 2
        if padding_value != expected_padding:
            raise ValueError(
                "DynamicConv1d with segment_size requires same padding when stride=1: "
                f"got padding={padding_value}, expected={expected_padding}."
            )
        return

    if stride_value % 2 == 0:
        expected_kernel = 2 * stride_value
        if effective_kernel != expected_kernel:
            raise ValueError(
                "DynamicConv1d with segment_size requires effective_kernel_size == "
                f"2 * stride for even strides: got {effective_kernel}, "
                f"expected={expected_kernel}."
            )
        expected_padding = stride_value // 2
        if padding_value != expected_padding:
            raise ValueError(
                "DynamicConv1d with segment_size requires padding == stride // 2: "
                f"got padding={padding_value}, expected={expected_padding}."
            )
        if segment % stride_value != 0:
            raise ValueError(
                "DynamicConv1d with segment_size requires segment_size to be a multiple "
                f"of stride: got segment_size={segment}, stride={stride_value}."
            )
        return

    raise ValueError(
        "DynamicConv1d with segment_size only supports stride=1 or even stride: "
        f"got stride={stride_value}."
    )


def _infer_padding_axis(kernel: int, stride_value: int) -> int:
    if stride_value == 1:
        return (kernel - 1) // 2
    if stride_value % 2 == 0:
        return stride_value // 2
    raise ValueError(
        "automatic padding only supports stride=1 or an even stride: "
        f"got effective_kernel_size={kernel}, stride={stride_value}."
    )
