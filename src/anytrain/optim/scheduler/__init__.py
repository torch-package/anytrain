"""Named step-level learning-rate schedulers."""

from __future__ import annotations

from math import cos, pi

import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str = "constant",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    stable_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    _validate_non_negative_int(warmup_steps, name="warmup_steps")
    _validate_ratio(min_lr_ratio, name="min_lr_ratio")
    _validate_schedule(schedule)

    if schedule == "constant":
        _reject_constant_steps(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    if schedule == "warmup_cosine":
        if stable_steps is not None or decay_steps is not None:
            raise ValueError(
                "stable_steps and decay_steps are only valid for wsd schedules."
            )
        resolved_total_steps = _require_positive_int(total_steps, name="total_steps")
        if resolved_total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps.")

        def lr_ratio(step: int) -> float:
            step = max(step, 0)
            if warmup_steps > 0 and step <= warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (resolved_total_steps - warmup_steps)
            return _cosine_ratio(progress, min_lr_ratio=min_lr_ratio)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_ratio)

    resolved_stable_steps = _require_non_negative_int(stable_steps, name="stable_steps")
    resolved_decay_steps = _require_positive_int(decay_steps, name="decay_steps")
    if total_steps is not None:
        resolved_total_steps = _require_positive_int(total_steps, name="total_steps")
        expected_total_steps = warmup_steps + resolved_stable_steps + resolved_decay_steps
        if resolved_total_steps != expected_total_steps:
            raise ValueError(
                "total_steps must equal warmup_steps + stable_steps + decay_steps for wsd."
            )
    decay_start_step = warmup_steps + resolved_stable_steps

    def lr_ratio(step: int) -> float:
        step = max(step, 0)
        if warmup_steps > 0 and step <= warmup_steps:
            return step / warmup_steps
        if resolved_stable_steps > 0 and step <= decay_start_step:
            return 1.0
        progress = (step - decay_start_step) / resolved_decay_steps
        return _cosine_ratio(progress, min_lr_ratio=min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_ratio)


def _cosine_ratio(progress: float, *, min_lr_ratio: float) -> float:
    clamped_progress = min(max(progress, 0.0), 1.0)
    cosine_progress = 0.5 * (1.0 - cos(pi * clamped_progress))
    return 1.0 + (min_lr_ratio - 1.0) * cosine_progress


def _validate_schedule(schedule: str) -> None:
    if not isinstance(schedule, str):
        raise TypeError("schedule must be a string.")
    if schedule not in {"constant", "warmup_cosine", "wsd"}:
        raise ValueError("schedule must be constant, warmup_cosine, or wsd.")


def _reject_constant_steps(
    *,
    warmup_steps: int,
    total_steps: int | None,
    stable_steps: int | None,
    decay_steps: int | None,
) -> None:
    if warmup_steps != 0:
        raise ValueError("warmup_steps is only valid for warmup_cosine or wsd schedules.")
    if total_steps is not None:
        raise ValueError("total_steps is only valid for warmup_cosine or wsd schedules.")
    if stable_steps is not None:
        raise ValueError("stable_steps is only valid for wsd schedules.")
    if decay_steps is not None:
        raise ValueError("decay_steps is only valid for wsd schedules.")


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _validate_non_negative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_non_negative_int(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    _validate_non_negative_int(value, name=name)
    return value


def _require_positive_int(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


__all__ = ["create_scheduler"]
