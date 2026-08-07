from __future__ import annotations

import torch
from torch import Tensor, nn

from anytrain._buffer import register_buffer


class STFT(nn.Module):
    window: Tensor

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
    ) -> None:
        super().__init__()
        self.n_fft = positive_int(n_fft, name="n_fft")
        self.win_length = (
            self.n_fft
            if win_length is None
            else positive_int(win_length, name="win_length")
        )
        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft.")
        self.hop_length = (
            self.win_length // 4
            if hop_length is None
            else positive_int(hop_length, name="hop_length")
        )
        if self.hop_length == 0:
            raise ValueError("default hop_length is zero; increase win_length or set hop_length.")
        register_buffer(self, "window", torch.hann_window(self.win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim < 2:
            raise ValueError("waveform must include batch and time dimensions.")

        original_shape = waveform.shape
        flattened = waveform.reshape(-1, original_shape[-1])
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            normalized=False,
            return_complex=True,
        )
        return spectrum.reshape(*original_shape[:-1], *spectrum.shape[-2:])


class MelSpectrogram(nn.Module):
    mel_filter: Tensor

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        hop_length: int | None = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = mel_scale
        register_buffer(
            self,
            "mel_filter",
            mel_filter(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                mel_scale=mel_scale,
            ),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        magnitude = self.stft(waveform).abs()
        return torch.einsum(
            "mf,...ft->...mt",
            self.mel_filter.to(device=waveform.device, dtype=waveform.dtype),
            magnitude,
        )


def mel_filter(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    mel_scale: str,
) -> Tensor:
    sample_rate = positive_int(sample_rate, name="sample_rate")
    n_fft = positive_int(n_fft, name="n_fft")
    n_mels = positive_int(n_mels, name="n_mels")
    if mel_scale not in {"htk", "slaney"}:
        raise ValueError("mel_scale must be 'htk' or 'slaney'.")

    num_bins = n_fft // 2 + 1
    if n_mels > num_bins:
        raise ValueError(
            f"n_mels={n_mels} exceeds available FFT bins ({num_bins}) for n_fft={n_fft}; "
            "reduce n_mels or increase n_fft."
        )

    max_frequency = sample_rate / 2
    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(0.0), mel_scale=mel_scale),
        hz_to_mel(torch.tensor(float(max_frequency)), mel_scale=mel_scale),
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points, mel_scale=mel_scale)
    fft_frequencies = torch.linspace(0, max_frequency, num_bins)

    filter_bank = torch.zeros(n_mels, num_bins)
    for mel_index in range(n_mels):
        lower = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        upper = hz_points[mel_index + 2]
        lower_slope = (fft_frequencies - lower) / (center - lower).clamp_min(1e-12)
        upper_slope = (upper - fft_frequencies) / (upper - center).clamp_min(1e-12)
        filter_bank[mel_index] = torch.minimum(lower_slope, upper_slope).clamp_min(0)
    return filter_bank


def hz_to_mel(frequency: Tensor, *, mel_scale: str) -> Tensor:
    if mel_scale == "htk":
        return 2595 * torch.log10(1 + frequency / 700)

    linear_scale = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_scale
    log_step = torch.log(
        torch.tensor(6.4, device=frequency.device, dtype=frequency.dtype)
    ) / 27.0
    linear_mel = frequency / linear_scale
    log_mel = min_log_mel + torch.log(frequency / min_log_hz) / log_step
    return torch.where(frequency >= min_log_hz, log_mel, linear_mel)


def mel_to_hz(mel: Tensor, *, mel_scale: str) -> Tensor:
    if mel_scale == "htk":
        return 700 * (10 ** (mel / 2595) - 1)

    linear_scale = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_scale
    log_step = torch.log(torch.tensor(6.4, device=mel.device, dtype=mel.dtype)) / 27.0
    linear_hz = mel * linear_scale
    log_hz = min_log_hz * torch.exp(log_step * (mel - min_log_mel))
    return torch.where(mel >= min_log_mel, log_hz, linear_hz)


def positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
