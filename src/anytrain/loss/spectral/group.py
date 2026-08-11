from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from anytrain._compat import strict_zip

from .._group import balancer as loss_balancer
from .._group import compute, module_dict
from ..abc import LossABC
from ..balancer import LossBalancerABC
from ..group import LossGroup
from ..observation import loss as register_loss
from ._transform import STFT, MelSpectrogram
from .single import LogMagnitudeLoss


def _default_log_magnitude_losses() -> dict[str, nn.Module]:
    return {"log_magnitude": LogMagnitudeLoss()}


class _TransformedLoss(LossABC):
    transform: STFT | MelSpectrogram

    def __init__(
        self,
        transform: STFT | MelSpectrogram,
        losses: Mapping[str, nn.Module],
        *,
        balancer: LossBalancerABC | None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.losses = module_dict(losses)
        self.balancer = loss_balancer(balancer)

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return compute(
            self.losses,
            self.balancer,
            self.transform(input),
            self.transform(target),
        )


class _MultiScaleLoss(LossGroup):
    @staticmethod
    def _validate_scales(values: Sequence[int], label: str) -> tuple[int, ...]:
        if not values:
            raise ValueError(f"{label} must contain at least one scale.")
        scales: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} values must be integers.")
            if value <= 0:
                raise ValueError(f"{label} values must be positive.")
            scales.append(value)
        return tuple(scales)


@register_loss
class STFTLoss(_TransformedLoss):
    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int | None = None,
        losses: Mapping[str, nn.Module] | None = None,
        balancer: LossBalancerABC | None = None,
    ) -> None:
        super().__init__(
            STFT(n_fft=n_fft, hop_length=hop_length),
            losses if losses is not None else _default_log_magnitude_losses(),
            balancer=balancer,
        )


@register_loss
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
    ) -> None:
        super().__init__(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=hop_length,
                mel_scale=mel_scale,
            ),
            losses if losses is not None else _default_log_magnitude_losses(),
            balancer=balancer,
        )


@register_loss
class MultiScaleSTFTLoss(_MultiScaleLoss):
    def __init__(
        self,
        *,
        n_fft_list: Sequence[int] = (2048, 1024, 512, 256),
        loss_kwargs: Mapping[str, Any] | None = None,
        balancer: LossBalancerABC | None = None,
    ) -> None:
        loss_kwargs = dict(loss_kwargs) if loss_kwargs is not None else {}
        losses = {
            f"stft_{n_fft}": STFTLoss(n_fft=n_fft, **loss_kwargs)
            for n_fft in self._validate_scales(n_fft_list, "n_fft_list")
        }
        super().__init__(losses, balancer=balancer)


@register_loss
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
    ) -> None:
        fft_sizes = self._validate_scales(n_fft_list, "n_fft_list")
        mel_bins = self._validate_scales(n_mels_list, "n_mels_list")
        if len(fft_sizes) != len(mel_bins):
            raise ValueError("n_fft_list and n_mels_list must have the same length.")

        loss_kwargs = dict(loss_kwargs) if loss_kwargs is not None else {}
        losses = {
            f"mel_{n_fft}_{n_mels}": MelLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                mel_scale=mel_scale,
                **loss_kwargs,
            )
            for n_fft, n_mels in strict_zip(fft_sizes, mel_bins)
        }
        super().__init__(losses, balancer=balancer)
