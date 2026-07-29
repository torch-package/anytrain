from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from anytrain._compat import strict_zip

from ..balancer import LossBalancerABC
from ..group import LossGroup
from .single import LogMagnitudeLoss
from .transform import MelSpectrogramTransform, SpectralTransformCache, STFTTransform


def _default_log_magnitude_losses() -> dict[str, nn.Module]:
    return {"log_magnitude": LogMagnitudeLoss()}


class STFTLoss(LossGroup):
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


class MelLoss(LossGroup):
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


class MultiScaleSTFTLoss(LossGroup):
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

    def _validate_scales(self, values: Sequence[int], label: str) -> tuple[int, ...]:
        if not values:
            raise ValueError(f"{label} must contain at least one scale.")
        return tuple(int(value) for value in values)

    def _forward_with_spectral_cache(
        self,
        *args: Any,
        spectral_cache: SpectralTransformCache,
        **kwargs: Any,
    ):
        return self(*args, _spectral_cache=spectral_cache, **kwargs)


class MultiScaleMelLoss(LossGroup):
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

    def _validate_scales(self, values: Sequence[int], label: str) -> tuple[int, ...]:
        if not values:
            raise ValueError(f"{label} must contain at least one scale.")
        return tuple(int(value) for value in values)

    def _forward_with_spectral_cache(
        self,
        *args: Any,
        spectral_cache: SpectralTransformCache,
        **kwargs: Any,
    ):
        return self(*args, _spectral_cache=spectral_cache, **kwargs)
