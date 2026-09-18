from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ..abc import LossABC
from ..observation import (
    emit_diagnostics,
    loss_with_diagnostics,
)
from ..observation import (
    loss as register_loss,
)


def _magnitude(spectrum: Tensor) -> Tensor:
    return spectrum.abs() if torch.is_complex(spectrum) else spectrum


def _require_complex(estimate: Tensor, reference: Tensor, *, loss_name: str) -> None:
    _require_matching_tensors(estimate, reference, loss_name=loss_name)
    if not torch.is_complex(estimate) or not torch.is_complex(reference):
        raise ValueError(f"{loss_name} requires complex estimate and reference tensors.")


def _require_matching_tensors(
    estimate: Tensor,
    reference: Tensor,
    *,
    loss_name: str,
) -> None:
    if not isinstance(estimate, Tensor) or not isinstance(reference, Tensor):
        raise TypeError(f"{loss_name} requires Tensor estimate and reference inputs.")
    if estimate.shape != reference.shape:
        raise ValueError(f"{loss_name} estimate and reference shapes must match.")


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _require_callable(value: object, *, name: str) -> Callable[[Tensor, Tensor], Tensor]:
    if not callable(value):
        raise TypeError(f"{name} must be callable.")
    return cast(Callable[[Tensor, Tensor], Tensor], value)


@register_loss
class LogMagnitudeLoss(LossABC):
    def __init__(
        self,
        *,
        eps: float = 1e-5,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.l1_loss,
    ) -> None:
        super().__init__()
        self.eps = _finite_positive(eps, name="eps")
        self.loss_fn = _require_callable(loss_fn, name="loss_fn")

    def compute_loss(self, estimate: Tensor, reference: Tensor) -> Tensor:
        _require_matching_tensors(estimate, reference, loss_name=type(self).__name__)
        estimate_log_magnitude = _magnitude(estimate).clamp_min(self.eps).log10()
        reference_log_magnitude = _magnitude(reference).clamp_min(self.eps).log10()
        return self.loss_fn(estimate_log_magnitude, reference_log_magnitude)


@loss_with_diagnostics
class CompressedSpectrogramLoss(LossABC):
    def __init__(
        self,
        *,
        compression: float = 0.3,
        eps: float = 1e-8,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.l1_loss,
    ) -> None:
        super().__init__()
        self.compression = _finite_positive(compression, name="compression")
        self.eps = _finite_positive(eps, name="eps")
        self.loss_fn = _require_callable(loss_fn, name="loss_fn")

    def compute_loss(self, estimate: Tensor, reference: Tensor) -> Tensor:
        _require_complex(estimate, reference, loss_name=type(self).__name__)

        estimate_magnitude, estimate_cos, estimate_sin = self._complex_to_magnitude_and_phase(
            estimate
        )
        reference_magnitude, reference_cos, reference_sin = self._complex_to_magnitude_and_phase(
            reference
        )

        estimate_compressed = estimate_magnitude.pow(self.compression)
        reference_compressed = reference_magnitude.pow(self.compression)

        magnitude_loss = self.loss_fn(estimate_compressed, reference_compressed)
        real_loss = self.loss_fn(
            estimate_cos * estimate_compressed,
            reference_cos * reference_compressed,
        )
        imag_loss = self.loss_fn(
            estimate_sin * estimate_compressed,
            reference_sin * reference_compressed,
        )
        emit_diagnostics(
            self,
            {
                "magnitude": magnitude_loss,
                "real": real_loss,
                "imag": imag_loss,
            },
        )
        return magnitude_loss + real_loss + imag_loss

    def _complex_to_magnitude_and_phase(self, spectrum: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        magnitude = spectrum.abs().clamp_min(self.eps)
        cos_phase = spectrum.real / magnitude
        sin_phase = spectrum.imag / magnitude
        return magnitude, cos_phase, sin_phase


@register_loss
class SpectralRMSELoss(LossABC):
    def __init__(
        self,
        *,
        eps: float = 1e-8,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
    ) -> None:
        super().__init__()
        self.eps = _finite_positive(eps, name="eps")
        self.loss_fn = _require_callable(loss_fn, name="loss_fn")

    def compute_loss(self, estimate: Tensor, reference: Tensor) -> Tensor:
        _require_complex(estimate, reference, loss_name=type(self).__name__)
        estimate_real_imag = torch.view_as_real(estimate)
        reference_real_imag = torch.view_as_real(reference)
        return torch.sqrt(self.loss_fn(estimate_real_imag, reference_real_imag) + self.eps)
