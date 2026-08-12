from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor, nn

SCHEMA_VERSION = 1
BLOCKED_LOAD_KEY = "__anytrain_blocked_load__"
BLOCKED_LOAD = object()


def extra_state(
    loss_names: tuple[str, ...],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "loss_names": loss_names,
        **config,
    }


def validate_extra_state(
    state: object,
    *,
    loss_names: tuple[str, ...],
    config: Mapping[str, Any],
) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeError("loss balancer extra state must be a mapping.")
    expected_keys = ("version", "loss_names", *config)
    missing_keys = [key for key in expected_keys if key not in state]
    unknown_keys = [key for key in state if key not in expected_keys]
    if missing_keys or unknown_keys:
        raise RuntimeError(
            "loss balancer checkpoint extra-state keys do not match: "
            f"missing={missing_keys}, unknown={unknown_keys}."
        )

    version = state["version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise RuntimeError("loss balancer checkpoint has an unsupported extra-state version.")

    checkpoint_names = state["loss_names"]
    if not isinstance(checkpoint_names, (list, tuple)) or not all(
        isinstance(name, str) for name in checkpoint_names
    ):
        raise RuntimeError("loss balancer checkpoint loss_names must be strings.")
    checkpoint_names = tuple(checkpoint_names)
    if checkpoint_names != loss_names:
        raise RuntimeError(
            "loss balancer checkpoint loss_names do not match the configured order: "
            f"checkpoint={checkpoint_names!r}, configured={loss_names!r}."
        )

    for field, expected_value in config.items():
        checkpoint_value = state[field]
        if (
            type(checkpoint_value) is not type(expected_value)
            or checkpoint_value != expected_value
        ):
            raise RuntimeError(
                f"loss balancer checkpoint {field}={checkpoint_value!r} does not "
                f"match configured {field}={expected_value!r}."
            )


def validate_checkpoint_state(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    loss_names: tuple[str, ...],
    config: Mapping[str, Any],
) -> None:
    extra_state_key = f"{prefix}_extra_state"
    if extra_state_key not in state_dict:
        raise RuntimeError(
            f"{type(module).__qualname__} checkpoint is missing loss-name metadata; "
            "it cannot be safely aligned with configured losses."
        )
    validate_extra_state(
        state_dict[extra_state_key],
        loss_names=loss_names,
        config=config,
    )

    expected_state = module.state_dict()
    expected_keys = {f"{prefix}{key}" for key in expected_state}
    actual_keys = {key for key in state_dict if key.startswith(prefix)}
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "loss balancer checkpoint keys do not match: "
            f"missing={missing_keys}, unexpected={unexpected_keys}."
        )

    for key, expected_value in expected_state.items():
        if not isinstance(expected_value, Tensor):
            continue
        checkpoint_value = state_dict[f"{prefix}{key}"]
        if not isinstance(checkpoint_value, Tensor):
            raise RuntimeError(
                f"loss balancer checkpoint value {prefix}{key!r} must be a tensor."
            )
        if checkpoint_value.shape != expected_value.shape:
            raise RuntimeError(
                f"loss balancer checkpoint tensor {prefix}{key!r} has shape "
                f"{tuple(checkpoint_value.shape)}, expected {tuple(expected_value.shape)}."
            )


def validate_blend(value: object, *, parse: Callable[[float], float]) -> None:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 0
        or not value.dtype.is_floating_point
    ):
        raise RuntimeError(
            "interpolated loss balancer checkpoint blend must be a scalar floating tensor."
        )
    try:
        parse(float(value.detach().cpu().item()))
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "interpolated loss balancer checkpoint blend must be finite and between 0 and 1."
        ) from error
