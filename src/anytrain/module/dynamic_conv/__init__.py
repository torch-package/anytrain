"""Backward-compatible imports for convolutional MoE layers.

The canonical home for these modules is :mod:`anytrain.module.moe.conv`.
"""

from ..moe.conv import (
    ADTRouter1d,
    ADTRouter2d,
    DynamicConv1d,
    DynamicConv2d,
    DynamicConvTranspose1d,
    MultiScalePool1d,
    eca_kernel_size,
)

__all__ = [
    "ADTRouter1d",
    "ADTRouter2d",
    "DynamicConv1d",
    "DynamicConv2d",
    "DynamicConvTranspose1d",
    "MultiScalePool1d",
    "eca_kernel_size",
]
