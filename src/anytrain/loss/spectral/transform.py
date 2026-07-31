from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anytrain._buffer import register_buffer


class _SpectrogramFactory(Protocol):
    def __call__(
        self,
        *,
        n_fft: int,
        hop_length: int | None,
        win_length: int | None,
        window_fn: Callable[..., Tensor],
        power: float | None,
        center: bool,
        normalized: bool,
    ) -> nn.Module: ...


class _MelSpectrogramFactory(Protocol):
    def __call__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        hop_length: int | None,
        win_length: int | None,
        f_min: float,
        f_max: float,
        power: float,
        mel_scale: str,
    ) -> nn.Module: ...


class _TorchAudioSpectrogramModule(Protocol):
    n_fft: int
    hop_length: int
    win_length: int
    window: Tensor
    normalized: bool | str
    center: bool
    pad_mode: str
    onesided: bool
    pad: int


TorchAudioMelSpectrogram: _MelSpectrogramFactory | None
TorchAudioSpectrogram: _SpectrogramFactory | None
_TORCHAUDIO_IMPORT_ERROR: Exception | None = None
try:
    from torchaudio.transforms import MelSpectrogram as _TorchAudioMelSpectrogram
    from torchaudio.transforms import Spectrogram as _TorchAudioSpectrogram
except (ImportError, OSError) as exc:  # pragma: no cover - depends on optional local environment
    TorchAudioMelSpectrogram = None
    TorchAudioSpectrogram = None
    _TORCHAUDIO_IMPORT_ERROR = exc
else:
    TorchAudioMelSpectrogram = _TorchAudioMelSpectrogram
    TorchAudioSpectrogram = _TorchAudioSpectrogram


def _torchaudio_required_error() -> ImportError:
    return ImportError("torchaudio is required for backend='torchaudio'.")


def _backend(backend: str, torchaudio_cls: object | None) -> str:
    if backend not in {"auto", "torchaudio", "torch"}:
        raise ValueError("backend must be 'auto', 'torchaudio', or 'torch'.")
    if backend == "torchaudio":
        if torchaudio_cls is None:
            raise _torchaudio_required_error() from _TORCHAUDIO_IMPORT_ERROR
        return "torchaudio"
    if backend == "auto":
        return "torchaudio" if torchaudio_cls is not None else "torch"
    return "torch"


def _require_torchaudio_spectrogram() -> _SpectrogramFactory:
    if TorchAudioSpectrogram is None:
        raise _torchaudio_required_error() from _TORCHAUDIO_IMPORT_ERROR
    return TorchAudioSpectrogram


def _require_torchaudio_mel_spectrogram() -> _MelSpectrogramFactory:
    if TorchAudioMelSpectrogram is None:
        raise _torchaudio_required_error() from _TORCHAUDIO_IMPORT_ERROR
    return TorchAudioMelSpectrogram


def _forward_torchaudio_spectrogram(
    transform: _TorchAudioSpectrogramModule,
    waveform: Tensor,
) -> Tensor:
    if transform.pad > 0:
        waveform = F.pad(waveform, (transform.pad, transform.pad), "constant")

    normalized = transform.normalized
    if normalized is True or normalized == "window":
        normalized_for_stft = False
        normalize_window = True
    elif normalized == "frame_length":
        normalized_for_stft = True
        normalize_window = False
    else:
        normalized_for_stft = bool(normalized)
        normalize_window = False

    original_shape = waveform.shape
    flattened = waveform.reshape(-1, original_shape[-1])
    spectrum = torch.stft(
        input=flattened,
        n_fft=transform.n_fft,
        hop_length=transform.hop_length,
        win_length=transform.win_length,
        window=transform.window,
        center=transform.center,
        pad_mode=transform.pad_mode,
        normalized=normalized_for_stft,
        onesided=transform.onesided,
        return_complex=True,
    )
    spectrum = spectrum.reshape(*original_shape[:-1], *spectrum.shape[-2:])
    if normalize_window:
        spectrum = spectrum / transform.window.pow(2.0).sum().sqrt()
    return spectrum


class SpectralTransformCache:
    """Cache raw STFTs for one loss forward pass.

    Entries are keyed by the transform configuration and tensor identity.  Keeping
    the source tensor in each entry makes the identity check robust against object
    id reuse while preserving autograd history for the cached result.  The cache is
    intentionally not a module or a process-wide singleton: callers own its
    lifetime and should create one per forward pass.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[object, ...], list[tuple[Tensor, Tensor]]] = {}

    def get(self, transform: STFTTransform, waveform: Tensor) -> Tensor:
        return self.get_with_key(
            transform.cache_key,
            waveform,
            lambda: transform._forward_uncached(waveform),
        )

    def get_with_key(
        self,
        key: tuple[object, ...],
        waveform: Tensor,
        compute: Callable[[], Tensor],
    ) -> Tensor:
        entries = self._entries.setdefault(key, [])
        for source, spectrum in entries:
            if source is waveform:
                return spectrum

        spectrum = compute()
        entries.append((waveform, spectrum))
        return spectrum


def _default_hop_length(*, n_fft: int, win_length: int) -> int:
    # Keep torch and torchaudio backends numerically aligned when hop is omitted.
    del n_fft
    return win_length // 4


class STFTTransform(nn.Module):
    window_tensor: Tensor

    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = (
            hop_length
            if hop_length is not None
            else _default_hop_length(n_fft=n_fft, win_length=self.win_length)
        )
        self.window = window
        self.center = center
        self.normalized = normalized
        self.backend = _backend(backend, TorchAudioSpectrogram)
        self.transform: nn.Module | None = None
        if self.backend == "torchaudio":
            spectrogram_cls = _require_torchaudio_spectrogram()
            self.transform = spectrogram_cls(
                n_fft=n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window_fn=self._get_window_fn(window),
                power=None,
                center=center,
                normalized=normalized,
            )
        else:
            register_buffer(
                self,
                "window_tensor",
                self._create_window(window, self.win_length),
            )

    @property
    def cache_key(self) -> tuple[object, ...]:
        if self.backend == "torch":
            window_device = self.window_tensor.device
            window_dtype = self.window_tensor.dtype
        else:
            if self.transform is None:
                raise RuntimeError("torchaudio spectrogram transform is not initialized.")
            window = getattr(self.transform, "window", None)
            if not isinstance(window, Tensor):
                raise RuntimeError("torchaudio spectrogram has no window tensor.")
            window_tensor = cast(Tensor, window)
            window_device = window_tensor.device
            window_dtype = window_tensor.dtype
        return (
            self.backend,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.normalized,
            0,
            "reflect",
            True,
            window_device,
            window_dtype,
        )

    def forward(
        self,
        waveform: Tensor,
        *,
        cache: SpectralTransformCache | None = None,
    ) -> Tensor:
        if cache is not None:
            return cache.get(self, waveform)
        return self._forward_uncached(waveform)

    def _forward_uncached(self, waveform: Tensor) -> Tensor:
        if self.backend == "torchaudio":
            if self.transform is None:
                raise RuntimeError("torchaudio spectrogram transform is not initialized.")
            return self.transform(waveform)

        original_shape = waveform.shape
        if waveform.ndim < 2:
            raise ValueError("waveform must include batch and time dimensions.")

        flattened = waveform.reshape(-1, original_shape[-1])
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_tensor.to(device=waveform.device, dtype=waveform.dtype),
            center=self.center,
            normalized=self.normalized,
            return_complex=True,
        )
        return spectrum.reshape(*original_shape[:-1], *spectrum.shape[-2:])

    def _create_window(self, window: str, win_length: int) -> Tensor:
        return self._get_window_fn(window)(win_length)

    def _get_window_fn(self, window: str):
        if window == "hann":
            return torch.hann_window
        if window == "none":
            return torch.ones
        raise ValueError(f"Unsupported window type {window!r}.")


class MelSpectrogramTransform(nn.Module):
    mel_filter: Tensor

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        n_mels: int = 128,
        hop_length: int | None = None,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 1.0,
        mel_scale: str = "htk",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if n_mels <= 0:
            raise ValueError("n_mels must be positive.")
        if power <= 0:
            raise ValueError("power must be positive.")
        if mel_scale not in {"htk", "slaney"}:
            raise ValueError("mel_scale must be 'htk' or 'slaney'.")

        self.backend = _backend(backend, TorchAudioMelSpectrogram)
        self.power = power
        self.mel_scale = mel_scale
        self.transform: nn.Module | None = None
        self.stft: STFTTransform | None = None
        resolved_win_length = win_length or n_fft
        resolved_hop_length = (
            hop_length
            if hop_length is not None
            else _default_hop_length(n_fft=n_fft, win_length=resolved_win_length)
        )
        f_max = f_max or sample_rate / 2
        if self.backend == "torchaudio":
            mel_spectrogram_cls = _require_torchaudio_mel_spectrogram()
            self.transform = mel_spectrogram_cls(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=resolved_hop_length,
                win_length=resolved_win_length,
                f_min=f_min,
                f_max=f_max,
                power=power,
                mel_scale=mel_scale,
            )
        else:
            self.stft = STFTTransform(
                n_fft=n_fft,
                hop_length=resolved_hop_length,
                win_length=resolved_win_length,
                backend="torch",
            )
            mel_filter = self._create_mel_filter(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                mel_scale=mel_scale,
            )
            if n_mels > n_fft // 2 + 1:
                raise ValueError(
                    f"n_mels={n_mels} exceeds available FFT bins ({n_fft // 2 + 1}) "
                    f"for n_fft={n_fft}; reduce n_mels or increase n_fft."
                )
            register_buffer(
                self,
                "mel_filter",
                mel_filter,
            )

    def forward(
        self,
        waveform: Tensor,
        *,
        cache: SpectralTransformCache | None = None,
    ) -> Tensor:
        if self.backend == "torchaudio":
            if self.transform is None:
                raise RuntimeError("torchaudio mel transform is not initialized.")
            if cache is not None:
                return self._forward_torchaudio_cached(waveform, cache=cache)
            return self.transform(waveform)

        if self.stft is None:
            raise RuntimeError("torch STFT fallback is not initialized.")
        magnitude = self.stft(waveform, cache=cache).abs().pow(self.power)
        return torch.einsum(
            "mf,...ft->...mt",
            self.mel_filter.to(device=waveform.device, dtype=waveform.dtype),
            magnitude,
        )

    def _forward_torchaudio_cached(
        self,
        waveform: Tensor,
        *,
        cache: SpectralTransformCache,
    ) -> Tensor:
        if self.transform is None:
            raise RuntimeError("torchaudio mel transform is not initialized.")
        spectrogram = getattr(self.transform, "spectrogram", None)
        mel_scale = getattr(self.transform, "mel_scale", None)
        if not isinstance(spectrogram, nn.Module) or not isinstance(mel_scale, nn.Module):
            raise RuntimeError("torchaudio mel transform has an unsupported structure.")

        module = cast(_TorchAudioSpectrogramModule, spectrogram)
        key = (
            "torchaudio",
            module.n_fft,
            module.hop_length,
            module.win_length,
            "hann",
            module.center,
            module.normalized,
            module.pad,
            module.pad_mode,
            module.onesided,
            module.window.device,
            module.window.dtype,
        )
        spectrum = cache.get_with_key(
            key,
            waveform,
            lambda: _forward_torchaudio_spectrogram(module, waveform),
        )
        mel_scale_module = cast(nn.Module, mel_scale)
        return mel_scale_module(spectrum.abs().pow(self.power))

    def _create_mel_filter(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        mel_scale: str,
    ) -> Tensor:
        if not 0 <= f_min < f_max <= sample_rate / 2:
            raise ValueError("f_min and f_max must satisfy 0 <= f_min < f_max <= sample_rate / 2.")

        mel_points = torch.linspace(
            self._hz_to_mel(torch.tensor(float(f_min)), mel_scale=mel_scale),
            self._hz_to_mel(torch.tensor(float(f_max)), mel_scale=mel_scale),
            n_mels + 2,
        )
        hz_points = self._mel_to_hz(mel_points, mel_scale=mel_scale)
        fft_frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

        filter_bank = torch.zeros(n_mels, n_fft // 2 + 1)
        for mel_index in range(n_mels):
            lower = hz_points[mel_index]
            center = hz_points[mel_index + 1]
            upper = hz_points[mel_index + 2]

            lower_slope = (fft_frequencies - lower) / (center - lower).clamp_min(1e-12)
            upper_slope = (upper - fft_frequencies) / (upper - center).clamp_min(1e-12)
            filter_bank[mel_index] = torch.minimum(lower_slope, upper_slope).clamp_min(0)
        return filter_bank

    @staticmethod
    def _hz_to_mel(frequency: Tensor, *, mel_scale: str) -> Tensor:
        if mel_scale == "htk":
            return 2595 * torch.log10(1 + frequency / 700)

        linear_scale = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / linear_scale
        log_step = (
            torch.log(torch.tensor(6.4, device=frequency.device, dtype=frequency.dtype)) / 27.0
        )
        linear_mel = frequency / linear_scale
        log_mel = min_log_mel + torch.log(frequency / min_log_hz) / log_step
        return torch.where(frequency >= min_log_hz, log_mel, linear_mel)

    @staticmethod
    def _mel_to_hz(mel: Tensor, *, mel_scale: str) -> Tensor:
        if mel_scale == "htk":
            return 700 * (10 ** (mel / 2595) - 1)

        linear_scale = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / linear_scale
        log_step = torch.log(torch.tensor(6.4, device=mel.device, dtype=mel.dtype)) / 27.0
        linear_hz = mel * linear_scale
        log_hz = min_log_hz * torch.exp(log_step * (mel - min_log_mel))
        return torch.where(mel >= min_log_mel, log_hz, linear_hz)
