from __future__ import annotations

from collections.abc import Callable

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
    if not torch.is_complex(estimate) or not torch.is_complex(reference):
        raise ValueError(f"{loss_name} requires complex estimate and reference tensors.")


@register_loss
class LogMagnitudeLoss(LossABC):
    def __init__(
        self,
        *,
        eps: float = 1e-5,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.l1_loss,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.loss_fn = loss_fn

    def compute_loss(self, estimate: Tensor, reference: Tensor) -> Tensor:
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
        if compression <= 0:
            raise ValueError("compression must be positive.")
        self.compression = compression
        self.eps = eps
        self.loss_fn = loss_fn

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
        self.eps = eps
        self.loss_fn = loss_fn

    def compute_loss(self, estimate: Tensor, reference: Tensor) -> Tensor:
        _require_complex(estimate, reference, loss_name=type(self).__name__)
        estimate_real_imag = torch.view_as_real(estimate)
        reference_real_imag = torch.view_as_real(reference)
        return torch.sqrt(self.loss_fn(estimate_real_imag, reference_real_imag) + self.eps)
