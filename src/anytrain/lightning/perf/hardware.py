from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ComputeDType = Literal["float32", "tf32", "float16", "bfloat16", "float8"]


@dataclass(frozen=True)
class PeakFlops:
    flops: float
    source: Literal["auto", "override"]
    device_name: str
    dtype: ComputeDType


@dataclass(frozen=True)
class _HardwareSpec:
    match: tuple[str, ...]
    flops: dict[ComputeDType, float]


_SPECS = [
    _HardwareSpec(
        match=("nvidia h100", "pcie"),
        flops={
            "float32": 51.0e12,
            "tf32": 378.0e12,
            "float16": 756.0e12,
            "bfloat16": 756.0e12,
            "float8": 1_513.0e12,
        },
    ),
    _HardwareSpec(
        match=("nvidia h100",),
        flops={
            "float32": 67.0e12,
            "tf32": 495.0e12,
            "float16": 989.0e12,
            "bfloat16": 989.0e12,
            "float8": 1_979.0e12,
        },
    ),
    _HardwareSpec(
        match=("nvidia a100",),
        flops={
            "float32": 19.5e12,
            "tf32": 156.0e12,
            "float16": 312.0e12,
            "bfloat16": 312.0e12,
        },
    ),
    _HardwareSpec(
        match=("nvidia geforce rtx 4090 d",),
        flops={
            "float32": 73.5e12,
            "tf32": 73.5e12,
            "float16": 147.0e12,
            "bfloat16": 147.0e12,
        },
    ),
    _HardwareSpec(
        match=("nvidia geforce rtx 4090",),
        flops={
            "float32": 82.6e12,
            "tf32": 82.6e12,
            "float16": 165.2e12,
            "bfloat16": 165.2e12,
        },
    ),
    _HardwareSpec(
        match=("nvidia geforce rtx 3090",),
        flops={
            "float32": 35.6e12,
            "tf32": 35.6e12,
            "float16": 71.0e12,
            "bfloat16": 71.0e12,
        },
    ),
]


def dtype_key(dtype: torch.dtype | str) -> ComputeDType:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if hasattr(torch, "float8_e4m3fn") and dtype is torch.float8_e4m3fn:
        return "float8"
    if hasattr(torch, "float8_e5m2") and dtype is torch.float8_e5m2:
        return "float8"

    if isinstance(dtype, str):
        value = dtype.lower().replace("-", "").replace("_", "")
        if value in {"fp32", "float32"}:
            return "float32"
        if value == "tf32":
            return "tf32"
        if value in {"fp16", "float16", "half"}:
            return "float16"
        if value in {"bf16", "bfloat16"}:
            return "bfloat16"
        if value in {"fp8", "float8"}:
            return "float8"

    raise ValueError("dtype must be one of float32, tf32, float16, bfloat16 or float8.")


def infer_peak_flops(
    *,
    dtype: torch.dtype | str,
    device: torch.device | int | str | None = None,
    device_name: str | None = None,
    hardware_peak_flops: float | None = None,
) -> PeakFlops | None:
    key = dtype_key(dtype)
    name = _device_name(device=device, device_name=device_name)
    if hardware_peak_flops is not None:
        _require_positive(hardware_peak_flops, "hardware_peak_flops")
        return PeakFlops(
            flops=float(hardware_peak_flops),
            source="override",
            device_name=name,
            dtype=key,
        )

    normalized_name = _normalize(name)
    for spec in _SPECS:
        if all(part in normalized_name for part in spec.match):
            flops = spec.flops.get(key)
            if flops is None:
                return None
            return PeakFlops(flops=flops, source="auto", device_name=name, dtype=key)

    return None


def _device_name(
    *,
    device: torch.device | int | str | None,
    device_name: str | None,
) -> str:
    if device_name is not None:
        if not device_name.strip():
            raise ValueError("device_name must not be empty.")
        return device_name

    if not torch.cuda.is_available():
        return "unknown"

    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            return str(torch_device)
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()

    return torch.cuda.get_device_name(index)


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
