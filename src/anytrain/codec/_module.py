from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from anytrain._buffer import register_buffer


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def integer_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError(f"{name} must contain integer ids.")
    return value


def validate_frame_codes(value: Any, codebook_sizes: Sequence[int]) -> Tensor:
    codes = integer_tensor(value, "codes")
    if codes.dim() != 3:
        raise ValueError("codes must have shape [batch, time, codebook].")
    if codes.shape[-1] != len(codebook_sizes):
        raise ValueError(
            f"codes must contain {len(codebook_sizes)} aligned codebooks."
        )
    return codes


class DeviceModule(nn.Module):
    _device: torch.Tensor

    def _init_device(self, device: torch.device) -> None:
        register_buffer(
            self,
            "_device",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self._device.device

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if assign:
            self._raise_assign_error()
        return super().load_state_dict(state_dict, strict=strict, assign=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        if local_metadata.get("assign_to_params_buffers", False):
            self._raise_assign_error()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _raise_assign_error() -> None:
        raise ValueError(
            "Codec wrappers do not support load_state_dict(assign=True) because it can "
            "separate backend tensors from the configured device."
        )
