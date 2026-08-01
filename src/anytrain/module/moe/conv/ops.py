"""Legacy ``nn.Conv1d`` subclass; prefer :class:`~anytrain.module.moe.Conv1dExpert`."""

from __future__ import annotations

from torch import nn


class Conv1d(nn.Conv1d):
    """``nn.Conv1d`` with an optional causal pad flag.

    New MoE code should use ``Conv1dExpert(causal=...)`` plus explicit
    ``weight`` / ``bias`` Parameters instead of this module template.
    """

    causal: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        *,
        causal: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.causal = causal


__all__ = ["Conv1d"]
