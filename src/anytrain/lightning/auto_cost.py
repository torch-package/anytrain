"""Model-agnostic CUDA memory calibration for cost-aware batches."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn

BatchT = TypeVar("BatchT")
ModuleT = TypeVar("ModuleT", bound=nn.Module)
PLModuleT = TypeVar("PLModuleT")

_GIB = 1024**3
_MIB = 1024**2


def _clear_module_gradients(pl_module: object, model: nn.Module) -> None:
    del pl_module
    model.zero_grad(set_to_none=True)


def _no_autocast(device: torch.device) -> AbstractContextManager[object]:
    del device
    return nullcontext()


@dataclass(frozen=True)
class CostProbe(Generic[BatchT]):
    """One already-collated probe batch and its requested cost."""

    batch: BatchT
    target_cost: int
    actual_cost: int | None = None
    padded_cost: int | None = None
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.target_cost, bool) or not isinstance(self.target_cost, int):
            raise TypeError("probe target_cost must be an integer.")
        if self.target_cost <= 0:
            raise ValueError("probe target_cost must be positive.")
        for name in ("actual_cost", "padded_cost", "sample_count"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"probe {name} must be a positive integer or None.")
            if value <= 0:
                raise ValueError(f"probe {name} must be a positive integer or None.")


@dataclass(frozen=True)
class AutoCostHooks(Generic[BatchT, ModuleT, PLModuleT]):
    """Project adapter used by calibrate_auto_cost."""

    prepare: Callable[[PLModuleT, BatchT], BatchT]
    move_to_device: Callable[[PLModuleT, BatchT, torch.device], BatchT]
    run_forward_backward: Callable[[PLModuleT, ModuleT, BatchT], None]
    actual_cost: Callable[[PLModuleT, BatchT], int] | None = None
    padded_cost: Callable[[PLModuleT, BatchT], int | None] | None = None
    clear_gradients: Callable[[PLModuleT, ModuleT], None] = _clear_module_gradients
    autocast: Callable[[torch.device], AbstractContextManager[object]] = _no_autocast

    def __post_init__(self) -> None:
        for name in (
            "prepare",
            "move_to_device",
            "run_forward_backward",
            "clear_gradients",
            "autocast",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"automatic cost hook {name} must be callable.")
        for name in ("actual_cost", "padded_cost"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"automatic cost hook {name} must be callable or None.")


@dataclass(frozen=True)
class AutoCostMeasurement:
    """One successful forward/backward memory measurement."""

    target_cost: int
    actual_cost: int
    peak_bytes: int
    padded_cost: int | None
    sample_count: int | None


@dataclass(frozen=True)
class AutoCostCalibration:
    """Successful measurements and probe targets skipped after OOM."""

    measurements: tuple[AutoCostMeasurement, ...]
    oom_targets: tuple[int, ...]

    @property
    def points(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (measurement.actual_cost, measurement.peak_bytes)
            for measurement in self.measurements
        )


@dataclass(frozen=True)
class AutoCostMemoryCalibration:
    """Linear fit of temporary CUDA bytes against an arbitrary batch cost."""

    mib_per_cost: float
    intercept_mib: float
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AutoCostBudget:
    """Conservative cost budget resolved from free CUDA memory and a fit."""

    cost: int
    raw_cost: int
    free_gib: float
    usable_gib: float
    memory_fraction: float
    baseline_gib: float
    headroom_gib: float
    mib_per_cost: float
    intercept_mib: float


def calibrate_auto_cost(
    pl_module: PLModuleT,
    model: ModuleT,
    probes: Sequence[CostProbe[BatchT]],
    hooks: AutoCostHooks[BatchT, ModuleT, PLModuleT],
    *,
    device: torch.device | str | int,
    steps: int = 1,
    warmup_steps: int = 0,
    sync_distributed: bool = True,
) -> AutoCostCalibration:
    """Measure real forward/backward memory for arbitrary project batches."""
    cuda_device = _cuda_device(device)
    _positive_int(steps, "steps")
    _non_negative_int(warmup_steps, "warmup_steps")
    if not isinstance(sync_distributed, bool):
        raise TypeError("sync_distributed must be a boolean.")
    normalized = tuple(probes)
    if not normalized:
        raise ValueError("automatic cost calibration requires at least one probe.")
    if any(not isinstance(probe, CostProbe) for probe in normalized):
        raise TypeError("automatic cost calibration probes must contain CostProbe values.")

    measurements: list[AutoCostMeasurement] = []
    oom_targets: list[int] = []
    for probe in normalized:
        failed = False
        for _ in range(warmup_steps):
            if _synchronized_measurement(
                pl_module,
                model,
                probe,
                hooks,
                device=cuda_device,
                measure=False,
                sync_distributed=sync_distributed,
            ) is None:
                failed = True
                break
        if not failed:
            for _ in range(steps):
                measurement = _synchronized_measurement(
                    pl_module,
                    model,
                    probe,
                    hooks,
                    device=cuda_device,
                    measure=True,
                    sync_distributed=sync_distributed,
                )
                if measurement is None:
                    failed = True
                    break
                measurements.append(measurement)
        if failed:
            oom_targets.append(probe.target_cost)
        hooks.clear_gradients(pl_module, model)
        torch.cuda.empty_cache()
    return AutoCostCalibration(tuple(measurements), tuple(oom_targets))


def fit_memory_calibration(
    points: Sequence[tuple[int, int]],
) -> AutoCostMemoryCalibration:
    """Fit peak_bytes = intercept + slope * cost for arbitrary cost units."""
    normalized_points: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        try:
            cost, peak = point
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"memory calibration point {index} must contain exactly two integers."
            ) from exc
        _exact_int(cost, f"memory calibration point {index} cost")
        _exact_int(peak, f"memory calibration point {index} peak")
        normalized_points.append((cost, peak))
    normalized = tuple(normalized_points)
    if len(normalized) < 2:
        raise ValueError("memory calibration requires at least two points.")
    if any(cost <= 0 or peak < 0 for cost, peak in normalized):
        raise ValueError(
            "memory calibration points must have positive costs and non-negative bytes."
        )
    mean_cost = math.fsum(cost for cost, _ in normalized) / len(normalized)
    mean_peak = math.fsum(peak for _, peak in normalized) / len(normalized)
    denominator = math.fsum((cost - mean_cost) ** 2 for cost, _ in normalized)
    if denominator <= 0.0:
        raise ValueError("memory calibration points must contain distinct costs.")
    slope_bytes = math.fsum(
        (cost - mean_cost) * (peak - mean_peak)
        for cost, peak in normalized
    ) / denominator
    intercept_bytes = mean_peak - slope_bytes * mean_cost
    if not math.isfinite(slope_bytes) or slope_bytes <= 0.0:
        raise ValueError("memory calibration produced a non-positive memory slope.")
    if intercept_bytes < 0.0:
        # Slightly superlinear peak memory (common on 2-point CUDA probes) makes
        # OLS intercept negative. resolve_auto_cost rejects that, and feeding a
        # negative intercept would inflate usable memory. Use a through-origin
        # envelope instead: intercept=0 and slope=max(peak/cost).
        slope_bytes = max(peak / cost for cost, peak in normalized)
        intercept_bytes = 0.0
        if not math.isfinite(slope_bytes) or slope_bytes <= 0.0:
            raise ValueError("memory calibration produced a non-positive memory slope.")
    return AutoCostMemoryCalibration(
        mib_per_cost=slope_bytes / _MIB,
        intercept_mib=intercept_bytes / _MIB,
        points=normalized,
    )


def resolve_auto_cost(
    *,
    device: torch.device | str | int,
    memory_fraction: float = 0.70,
    baseline_gib: float = 0.0,
    headroom_gib: float = 2.0,
    mib_per_cost: float,
    intercept_mib: float = 0.0,
    min_cost: int = 1000,
    concurrent_loaders: int = 1,
) -> AutoCostBudget:
    """Resolve a conservative arbitrary-unit cost budget from CUDA memory."""
    _finite_fraction(memory_fraction, "memory_fraction")
    _non_negative_real(baseline_gib, "baseline_gib")
    _non_negative_real(headroom_gib, "headroom_gib")
    _positive_real(mib_per_cost, "mib_per_cost")
    _non_negative_real(intercept_mib, "intercept_mib")
    _positive_int(min_cost, "min_cost")
    _positive_int(concurrent_loaders, "concurrent_loaders")
    cuda_device = _cuda_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(cuda_device)
    free_gib = free_bytes / _GIB
    effective_free_bytes = max(0.0, float(free_bytes) - float(baseline_gib) * _GIB)
    usable_bytes = (
        effective_free_bytes * float(memory_fraction)
        - float(headroom_gib) * _GIB
        - float(intercept_mib) * _MIB
    ) / concurrent_loaders
    raw_cost = math.floor(usable_bytes / (float(mib_per_cost) * _MIB))
    if raw_cost < min_cost:
        raise RuntimeError(
            "automatic cost budget is below the configured minimum: "
            f"free={free_gib:.2f} GiB, usable={max(usable_bytes, 0.0) / _GIB:.2f} GiB, "
            f"min_cost={min_cost}."
        )
    cost = max(min_cost, (raw_cost // 256) * 256)
    return AutoCostBudget(
        cost=cost,
        raw_cost=raw_cost,
        free_gib=free_gib,
        usable_gib=max(usable_bytes, 0.0) / _GIB,
        memory_fraction=float(memory_fraction),
        baseline_gib=float(baseline_gib),
        headroom_gib=float(headroom_gib),
        mib_per_cost=float(mib_per_cost),
        intercept_mib=float(intercept_mib),
    )


def _synchronized_measurement(
    pl_module: PLModuleT,
    model: ModuleT,
    probe: CostProbe[BatchT],
    hooks: AutoCostHooks[BatchT, ModuleT, PLModuleT],
    *,
    device: torch.device,
    measure: bool,
    sync_distributed: bool,
) -> AutoCostMeasurement | None:
    local_oom = torch.zeros((), dtype=torch.int32, device=device)
    result: AutoCostMeasurement | None = None
    try:
        result = _measurement(
            pl_module,
            model,
            probe,
            hooks,
            device=device,
            measure=measure,
        )
    except torch.cuda.OutOfMemoryError:
        local_oom.fill_(1)
        torch.cuda.empty_cache()
    if sync_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        from .audit import audit_collective

        audit_collective(
            "torch.distributed.all_reduce",
            lambda: torch.distributed.all_reduce(
                local_oom, op=torch.distributed.ReduceOp.MAX
            ),
            reduce_op="max",
            source="auto_cost.local_oom",
            tensor_shape=tuple(local_oom.shape),
            tensor_dtype=str(local_oom.dtype),
            tensor_device=str(local_oom.device),
        )
    if int(local_oom.item()):
        hooks.clear_gradients(pl_module, model)
        torch.cuda.empty_cache()
        return None
    if result is None:
        raise RuntimeError("automatic cost measurement failed without an OOM status.")
    if sync_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        from .audit import audit_collective

        peak = torch.tensor(result.peak_bytes, dtype=torch.int64, device=device)
        audit_collective(
            "torch.distributed.all_reduce",
            lambda: torch.distributed.all_reduce(
                peak, op=torch.distributed.ReduceOp.MAX
            ),
            reduce_op="max",
            source="auto_cost.peak_bytes",
            tensor_shape=tuple(peak.shape),
            tensor_dtype=str(peak.dtype),
            tensor_device=str(peak.device),
        )
        result = AutoCostMeasurement(
            target_cost=result.target_cost,
            actual_cost=result.actual_cost,
            peak_bytes=int(peak.item()),
            padded_cost=result.padded_cost,
            sample_count=result.sample_count,
        )
    return result


def _measurement(
    pl_module: PLModuleT,
    model: ModuleT,
    probe: CostProbe[BatchT],
    hooks: AutoCostHooks[BatchT, ModuleT, PLModuleT],
    *,
    device: torch.device,
    measure: bool,
) -> AutoCostMeasurement:
    hooks.clear_gradients(pl_module, model)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    prepared = probe.batch
    try:
        prepared = hooks.prepare(pl_module, probe.batch)
        actual_cost = probe.actual_cost
        if hooks.actual_cost is not None:
            actual_cost = hooks.actual_cost(pl_module, prepared)
        if actual_cost is None:
            raise ValueError("each cost probe needs actual_cost or AutoCostHooks.actual_cost.")
        _positive_int(actual_cost, "probe actual_cost")
        padded_cost = probe.padded_cost
        if hooks.padded_cost is not None:
            padded_cost = hooks.padded_cost(pl_module, prepared)
        if padded_cost is not None:
            _positive_int(padded_cost, "probe padded_cost")
        prepared = hooks.move_to_device(pl_module, prepared, device)
        with hooks.autocast(device):
            hooks.run_forward_backward(pl_module, model, prepared)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    finally:
        hooks.clear_gradients(pl_module, model)
        del prepared
    return AutoCostMeasurement(
        target_cost=probe.target_cost,
        actual_cost=actual_cost,
        peak_bytes=max(0, int(peak - baseline)) if measure else 0,
        padded_cost=padded_cost,
        sample_count=probe.sample_count,
    )


def _cuda_device(value: torch.device | str | int) -> torch.device:
    if isinstance(value, bool):
        raise TypeError("automatic cost device must be a CUDA device, string, or integer index.")
    device = torch.device("cuda", value) if isinstance(value, int) else torch.device(value)
    if device.type != "cuda":
        raise ValueError("automatic cost calibration requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("automatic cost calibration requires a CUDA device.")
    return device


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _exact_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _positive_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a positive real number.")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive real number.")


def _non_negative_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a non-negative real number.")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a non-negative real number.")


def _finite_fraction(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be in (0, 1).")
    if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")


__all__ = [
    "AutoCostBudget",
    "AutoCostCalibration",
    "AutoCostHooks",
    "AutoCostMeasurement",
    "AutoCostMemoryCalibration",
    "CostProbe",
    "calibrate_auto_cost",
    "fit_memory_calibration",
    "resolve_auto_cost",
]
