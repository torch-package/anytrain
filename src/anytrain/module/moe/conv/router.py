from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from ...dirichlet_tempering import ADT
from ..routing import validate_num_experts

ActivationName = Literal["gelu", "identity", "relu", "silu"]
NormName = Literal["batch", "group", "identity"]


def eca_kernel_size(channels: int, beta: int = 1, gamma: int = 2) -> int:
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}.")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}.")
    kernel_size = (int(math.log2(channels)) + beta) // gamma
    if kernel_size % 2 == 0:
        kernel_size += 1
    return max(1, kernel_size)


class MultiScalePool1d(nn.Module):
    layers: nn.ModuleList

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels is None:
            out_channels = in_channels
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(in_channels, out_channels, 1)),
                nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Conv1d(in_channels, out_channels, 1)),
                nn.Sequential(nn.AdaptiveAvgPool1d(3), nn.Conv1d(in_channels, out_channels, 1)),
                nn.Sequential(nn.AdaptiveMaxPool1d(3), nn.Conv1d(in_channels, out_channels, 1)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.cat([layer(x) for layer in self.layers], dim=-1)
        return pooled.mean(dim=-1, keepdim=True)


class ADTRouter1d(nn.Module):
    dropout: nn.Module
    layer: nn.Sequential
    gate: ADT

    def __init__(
        self,
        channels: int,
        num_experts: int,
        *,
        hidden_size: int | None = None,
        multi_scale: bool | None = None,
        norm: NormName = "group",
        activation: ActivationName = "silu",
        norm_groups: int | None = None,
        weight_norm: bool = False,
        dropout: float | None = None,
        **adt_kwargs,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        validate_num_experts(num_experts)
        if dropout is not None and not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.channels = channels
        self.num_experts = num_experts
        self.dropout = nn.Identity() if dropout is None else nn.Dropout1d(dropout)

        if multi_scale is None:
            multi_scale = num_experts >= 10
        if hidden_size is None:
            hidden_size = max(1, channels // 4) if multi_scale else channels
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")

        kernel_size = eca_kernel_size(channels)
        conv_in: nn.Module = nn.Conv1d(
            channels,
            hidden_size,
            kernel_size,
            padding=kernel_size // 2,
        )
        conv_out: nn.Module = nn.Conv1d(hidden_size, num_experts, kernel_size=1)

        if weight_norm:
            conv_in = nn.utils.parametrizations.weight_norm(conv_in)
            conv_out = nn.utils.parametrizations.weight_norm(conv_out)

        pooling: nn.Module = (
            MultiScalePool1d(hidden_size) if multi_scale else nn.AdaptiveAvgPool1d(1)
        )

        self.layer = nn.Sequential(
            conv_in,
            build_norm_1d(norm, hidden_size, norm_groups=norm_groups),
            build_activation(activation),
            pooling,
            conv_out,
        )
        self.gate = ADT.from_kwargs(num_experts=num_experts, **adt_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"ADTRouter1d expects input shape (B, C, T), got {tuple(x.shape)}.")
        if x.size(1) != self.channels:
            raise ValueError(
                "ADTRouter1d channel mismatch: "
                f"got {x.size(1)}, expected {self.channels}."
            )
        logits = self.layer(self.dropout(x)).squeeze(-1)
        return self.gate(logits)


class ADTRouter2d(nn.Module):
    dropout: nn.Module
    layer: nn.Sequential
    gate: ADT

    def __init__(
        self,
        channels: int,
        num_experts: int,
        *,
        hidden_size: int | None = None,
        norm: NormName = "group",
        activation: ActivationName = "silu",
        norm_groups: int | None = None,
        weight_norm: bool = False,
        dropout: float | None = None,
        **adt_kwargs,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        validate_num_experts(num_experts)
        if dropout is not None and not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.channels = channels
        self.num_experts = num_experts
        self.dropout = nn.Identity() if dropout is None else nn.Dropout2d(dropout)

        if hidden_size is None:
            hidden_size = channels
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")

        conv_in: nn.Module = nn.Conv2d(channels, hidden_size, kernel_size=1)
        conv_out: nn.Module = nn.Conv2d(hidden_size, num_experts, kernel_size=1)

        if weight_norm:
            conv_in = nn.utils.parametrizations.weight_norm(conv_in)
            conv_out = nn.utils.parametrizations.weight_norm(conv_out)

        self.layer = nn.Sequential(
            conv_in,
            build_norm_2d(norm, hidden_size, norm_groups=norm_groups),
            build_activation(activation),
            nn.AdaptiveAvgPool2d(1),
            conv_out,
        )
        self.gate = ADT.from_kwargs(num_experts=num_experts, **adt_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"ADTRouter2d expects input shape (B, C, H, W), got {tuple(x.shape)}.")
        if x.size(1) != self.channels:
            raise ValueError(
                "ADTRouter2d channel mismatch: "
                f"got {x.size(1)}, expected {self.channels}."
            )
        logits = self.layer(self.dropout(x)).flatten(1)
        return self.gate(logits)


def build_activation(name: ActivationName) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation {name!r}.")


def build_norm_1d(name: NormName, channels: int, *, norm_groups: int | None) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm1d(channels)
    if name == "group":
        groups = _group_count(channels, norm_groups)
        return nn.GroupNorm(groups, channels)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported norm {name!r}.")


def build_norm_2d(name: NormName, channels: int, *, norm_groups: int | None) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "group":
        groups = _group_count(channels, norm_groups)
        return nn.GroupNorm(groups, channels)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported norm {name!r}.")


def _group_count(channels: int, requested_groups: int | None) -> int:
    if requested_groups is not None:
        if requested_groups <= 0:
            raise ValueError(f"norm_groups must be positive, got {requested_groups}.")
        if channels % requested_groups != 0:
            raise ValueError(
                "norm_groups must divide channels: "
                f"got channels={channels}, norm_groups={requested_groups}."
            )
        return requested_groups

    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return groups
