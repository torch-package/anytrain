from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from anytrain._compat import strict_zip

from ..balancer import LossBalancerABC
from ._group import SpectralLossGroup
from .single import LogMagnitudeLoss
from .transform import MelSpectrogramTransform, SpectralTransformCache, STFTTransform


def _default_log_magnitude_losses() -> dict[str, nn.Module]:
    return {"log_magnitude": LogMagnitudeLoss()}


class _TransformedLoss(SpectralLossGroup):
    transform: STFTTransform | MelSpectrogramTransform

    def forward(self, input: Tensor, target: Tensor):
        return super().forward(self.transform(input), self.transform(target))

    def _forward_with_spectral_cache(
        self,
        input: Tensor,
        target: Tensor,
        *,
        spectral_cache: SpectralTransformCache,
    ):
        return super().forward(
            self.transform(input, cache=spectral_cache),
            self.transform(target, cache=spectral_cache),
        )


class _MultiScaleLoss(SpectralLossGroup):
    @staticmethod
    def _validate_scales(values: Sequence[int], label: str) -> tuple[int, ...]:
        if not values:
            raise ValueError(f"{label} must contain at least one scale.")
        return tuple(int(value) for value in values)


class STFTLoss(_TransformedLoss):
    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int | None = None,
        losses: Mapping[str, nn.Module] | None = None,
        balancer: LossBalancerABC | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__(losses or _default_log_magnitude_losses(), balancer=balancer)
        self.transform = STFTTransform(n_fft=n_fft, hop_length=hop_length, backend=backend)


class MelLoss(_TransformedLoss):
    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        n_mels: int = 128,
        hop_length: int | None = None,
        losses: Mapping[str, nn.Module] | None = None,
        balancer: LossBalancerABC | None = None,
        mel_scale: str = "htk",
        backend: str = "auto",
    ) -> None:
        super().__init__(losses or _default_log_magnitude_losses(), balancer=balancer)
        self.transform = MelSpectrogramTransform(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            mel_scale=mel_scale,
            backend=backend,
        )


class MultiScaleSTFTLoss(_MultiScaleLoss):
    def __init__(
        self,
        *,
        n_fft_list: Sequence[int] = (2048, 1024, 512, 256),
        loss_kwargs: Mapping[str, Any] | None = None,
        balancer: LossBalancerABC | None = None,
        backend: str = "auto",
    ) -> None:
        loss_kwargs = dict(loss_kwargs or {})
        losses = {
            f"stft_{n_fft}": STFTLoss(n_fft=n_fft, backend=backend, **loss_kwargs)
            for n_fft in self._validate_scales(n_fft_list, "n_fft_list")
        }
        super().__init__(losses, balancer=balancer)


class MultiScaleMelLoss(_MultiScaleLoss):
    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        n_fft_list: Sequence[int] = (2048, 1024, 512, 256, 128, 64, 32),
        n_mels_list: Sequence[int] = (320, 160, 80, 40, 20, 10, 5),
        loss_kwargs: Mapping[str, Any] | None = None,
        balancer: LossBalancerABC | None = None,
        mel_scale: str = "htk",
        backend: str = "auto",
    ) -> None:
        fft_sizes = self._validate_scales(n_fft_list, "n_fft_list")
        mel_bins = self._validate_scales(n_mels_list, "n_mels_list")
        if len(fft_sizes) != len(mel_bins):
            raise ValueError("n_fft_list and n_mels_list must have the same length.")

        loss_kwargs = dict(loss_kwargs or {})
        losses = {
            f"mel_{n_fft}_{n_mels}": MelLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                mel_scale=mel_scale,
                backend=backend,
                **loss_kwargs,
            )
            for n_fft, n_mels in strict_zip(fft_sizes, mel_bins)
        }
        super().__init__(losses, balancer=balancer)
