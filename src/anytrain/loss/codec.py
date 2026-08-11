from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from anytrain._compat import StrEnum

from .balancer import FixedWeightLossBalancer, LossBalancerABC
from .group import LossGroup
from .observation import loss as register_loss
from .spectral import MultiScaleMelLoss, MultiScaleSTFTLoss
from .temporal import SDRLoss

__all__ = ["CodecLoss", "CodecLossPreset"]

DAC_MEL_N_FFT_LIST = (2048, 1024, 512, 256, 128, 64, 32)
DAC_MEL_N_MELS_LIST = (320, 160, 80, 40, 20, 10, 5)
DYNACODEC_STFT_N_FFT_LIST = (2048, 1024, 512, 256)
DYNACODEC_LOSS_WEIGHTS = {
    "si_sdr": 1.0,
    "multi_mel": 10.0,
    "multi_stft": 10.0,
}


class CodecLossPreset(StrEnum):
    DAC = "dac"
    DYNACODEC = "dynacodec"


@register_loss
class CodecLoss(LossGroup):
    @classmethod
    def from_preset(
        cls,
        preset: CodecLossPreset | str,
        *,
        sample_rate: int = 44100,
        mel_n_fft_list: Sequence[int] = DAC_MEL_N_FFT_LIST,
        mel_n_mels_list: Sequence[int] = DAC_MEL_N_MELS_LIST,
        stft_n_fft_list: Sequence[int] = DYNACODEC_STFT_N_FFT_LIST,
        mel_loss_kwargs: Mapping[str, Any] | None = None,
        stft_loss_kwargs: Mapping[str, Any] | None = None,
        balancer: LossBalancerABC | None = None,
        mel_scale: str = "htk",
    ) -> CodecLoss:
        preset = cls._preset(preset)
        if preset == CodecLossPreset.DAC:
            return cls(
                {
                    "multi_mel": MultiScaleMelLoss(
                        sample_rate=sample_rate,
                        n_fft_list=mel_n_fft_list,
                        n_mels_list=mel_n_mels_list,
                        loss_kwargs=mel_loss_kwargs,
                        mel_scale=mel_scale,
                    )
                },
                balancer=balancer,
            )
        if preset == CodecLossPreset.DYNACODEC:
            return cls(
                {
                    "si_sdr": SDRLoss(),
                    "multi_mel": MultiScaleMelLoss(
                        sample_rate=sample_rate,
                        n_fft_list=mel_n_fft_list,
                        n_mels_list=mel_n_mels_list,
                        loss_kwargs=mel_loss_kwargs,
                        mel_scale=mel_scale,
                    ),
                    "multi_stft": MultiScaleSTFTLoss(
                        n_fft_list=stft_n_fft_list,
                        loss_kwargs=stft_loss_kwargs,
                    ),
                },
                balancer=(
                    balancer
                    if balancer is not None
                    else FixedWeightLossBalancer(DYNACODEC_LOSS_WEIGHTS)
                ),
            )
        raise ValueError(f"Unsupported codec loss preset {preset!r}.")

    @staticmethod
    def _preset(preset: CodecLossPreset | str) -> CodecLossPreset:
        if isinstance(preset, CodecLossPreset):
            return preset
        if not isinstance(preset, str):
            raise TypeError("preset must be a CodecLossPreset or string.")
        try:
            return CodecLossPreset(preset)
        except ValueError as exc:
            supported = ", ".join(item.value for item in CodecLossPreset)
            raise ValueError(
                f"Unknown codec loss preset {preset!r}. Supported presets: {supported}."
            ) from exc
