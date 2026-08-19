from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

from anytrain._buffer import register_buffer
from anytrain._compat import strict_zip
from anytrain.framework._validation import finite_real, integer_at_least

_BANDS = (0.1, 0.25, 0.5, 0.75)


class DACDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        msd_dim: int = 16,
        mpd_dim: int = 32,
        mrd_dim: int = 32,
        sample_rates: Sequence[int] = (),
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        n_ffts: Sequence[int] = (2048, 1024, 512),
        sample_rate: int = 44_100,
        bands: Sequence[float] = _BANDS,
    ) -> None:
        super().__init__()
        in_channels = integer_at_least(in_channels, "in_channels", 1)
        msd_dim = integer_at_least(msd_dim, "msd_dim", 1)
        mpd_dim = integer_at_least(mpd_dim, "mpd_dim", 1)
        mrd_dim = integer_at_least(mrd_dim, "mrd_dim", 1)
        sample_rate = integer_at_least(sample_rate, "sample_rate", 1)
        sample_rates = tuple(
            integer_at_least(rate, "sample_rates item", 1) for rate in sample_rates
        )
        periods = tuple(integer_at_least(period, "periods item", 1) for period in periods)
        n_ffts = tuple(integer_at_least(n_fft, "n_ffts item", 1) for n_fft in n_ffts)
        bands = tuple(finite_real(band, "bands item") for band in bands)

        layers: list[nn.Module] = []
        layers.extend(
            _MSD(
                in_channels=in_channels,
                sample_rate=rate,
                source_sample_rate=sample_rate,
                dim=msd_dim,
            )
            for rate in sample_rates
        )
        layers.extend(_MPD(in_channels=in_channels, period=period, dim=mpd_dim) for period in periods)
        layers.extend(
            _MRD(in_channels=in_channels, n_fft=n_fft, bands=bands, dim=mrd_dim)
            for n_fft in n_ffts
        )
        if not layers:
            raise ValueError("DACDiscriminator requires at least one discriminator branch.")
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> list[list[Tensor]]:
        x = _preprocess(x)
        return [layer(x) for layer in self.layers]


class _MPD(nn.Module):
    def __init__(self, *, in_channels: int, period: int, dim: int) -> None:
        super().__init__()
        period = integer_at_least(period, "period", 1)
        dim = integer_at_least(dim, "dim", 1)
        self.period = period
        self.convs = nn.ModuleList(
            [
                _conv2d(in_channels, dim, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                _conv2d(dim, 4 * dim, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                _conv2d(4 * dim, 16 * dim, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                _conv2d(16 * dim, 32 * dim, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                _conv2d(32 * dim, 32 * dim, kernel_size=(5, 1), padding=(2, 0)),
            ]
        )
        self.post = _conv2d(
            32 * dim,
            in_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            activation=False,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.shape[-1] < self.period:
            raise ValueError(
                f"MPD period {self.period} requires at least {self.period} time samples; "
                f"got {x.shape[-1]}."
            )
        x = F.pad(x, (0, -x.shape[-1] % self.period), mode="reflect")
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)
        return _forward_stack(x, self.convs, self.post)


class _MSD(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        sample_rate: int,
        source_sample_rate: int,
        dim: int,
    ) -> None:
        super().__init__()
        sample_rate = integer_at_least(sample_rate, "sample rate", 1)
        source_sample_rate = integer_at_least(source_sample_rate, "source sample rate", 1)
        dim = integer_at_least(dim, "dim", 1)
        self.sample_rate = sample_rate
        self.source_sample_rate = source_sample_rate
        if dim % 4 != 0:
            raise ValueError(
                f"msd_dim must be divisible by 4 for DAC MSD group schedule, got {dim}."
            )
        self.convs = nn.ModuleList(
            [
                _conv1d(in_channels, dim, kernel_size=15, padding=7),
                _conv1d(dim, 4 * dim, kernel_size=41, stride=4, padding=20, groups=4),
                _conv1d(4 * dim, 16 * dim, kernel_size=41, stride=4, padding=20, groups=16),
                _conv1d(16 * dim, 64 * dim, kernel_size=41, stride=4, padding=20, groups=64),
                _conv1d(64 * dim, 64 * dim, kernel_size=41, stride=4, padding=20, groups=256),
                _conv1d(64 * dim, 64 * dim, kernel_size=5, padding=2),
            ]
        )
        self.post = _conv1d(
            64 * dim,
            in_channels,
            kernel_size=3,
            padding=1,
            activation=False,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.sample_rate != self.source_sample_rate:
            try:
                from torchaudio.functional import resample
            except ImportError as exc:
                raise ImportError(
                    "DAC MSD resampling requires torchaudio. "
                    "Install audio dependencies with `python -m pip install 'anytrain[audio]'`."
                ) from exc

            x = resample(x, self.source_sample_rate, self.sample_rate)
        return _forward_stack(x, self.convs, self.post)


class _MRD(nn.Module):
    window: Tensor

    def __init__(
        self,
        *,
        in_channels: int,
        n_fft: int,
        bands: Sequence[float],
        dim: int,
        hop_factor: float = 0.25,
    ) -> None:
        super().__init__()
        n_fft = integer_at_least(n_fft, "n_fft", 1)
        hop_factor = finite_real(hop_factor, "hop_factor")
        if hop_factor <= 0:
            raise ValueError("hop_factor must be positive.")

        self.n_fft = n_fft
        self.hop_length = int(hop_factor * n_fft)
        if self.hop_length < 1:
            raise ValueError("hop_factor * n_fft must produce a hop_length of at least 1.")
        register_buffer(self, "window", torch.hann_window(n_fft), persistent=False)
        bins = n_fft // 2 + 1
        self.bands = _split_indices(bands, bins=bins)
        branch_count = len(self.bands) + 1
        self.stacks = nn.ModuleList(
            [
                _Stack(
                    [
                        _conv2d(
                            2 * in_channels,
                            dim,
                            kernel_size=(3, 9),
                            padding=(1, 4),
                        ),
                        _conv2d(dim, dim, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                        _conv2d(dim, dim, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                        _conv2d(dim, dim, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                        _conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1)),
                    ]
                )
                for _ in range(branch_count)
            ]
        )
        self.post = _conv2d(
            dim,
            in_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            activation=False,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.shape[-1] <= self.n_fft // 2:
            raise ValueError(
                f"MRD n_fft={self.n_fft} requires more than {self.n_fft // 2} time samples; "
                f"got {x.shape[-1]}."
            )
        spec = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = rearrange(spec, "(b c) f t p -> b (c p) t f", b=x.shape[0], c=x.shape[1])

        features: list[Tensor] = []
        bands: list[Tensor] = []
        for band, stack in strict_zip(torch.tensor_split(spec, self.bands, dim=-1), self.stacks):
            band_features = stack(band)
            features.extend(band_features)
            band = band_features[-1]
            bands.append(band)

        logits = self.post(torch.cat(bands, dim=-1))
        features.append(logits)
        return features


def _forward_stack(x: Tensor, layers: nn.ModuleList, post: nn.Module) -> list[Tensor]:
    features: list[Tensor] = []
    for layer in layers:
        x = layer(x)
        features.append(x)
    x = post(x)
    features.append(x)
    return features


class _Stack(nn.Module):
    def __init__(self, layers: Sequence[nn.Module]) -> None:
        super().__init__()
        if not layers:
            raise ValueError("stack must contain at least one layer.")
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


def _preprocess(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError("DACDiscriminator expects a torch.Tensor input.")
    if x.ndim != 3:
        raise ValueError("DACDiscriminator expects input shape (batch, channels, time).")
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.abs().amax(dim=-1, keepdim=True) + 1e-8)
    return 0.8 * x


def _split_indices(bands: Sequence[float], *, bins: int) -> list[int]:
    indices: list[int] = []
    previous = 0
    for band in bands:
        band = finite_real(band, "band")
        if not 0 < band < 1:
            raise ValueError("bands must contain ratios between 0 and 1.")
        index = int(band * bins)
        if index <= previous or index >= bins:
            raise ValueError(
                "bands must produce strictly increasing non-empty frequency ranges; "
                f"got ratio {band!r} for {bins} bins."
            )
        indices.append(index)
        previous = index
    return indices


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
    *,
    groups: int = 1,
    activation: bool = True,
) -> nn.Module:
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
    )
    return _block(conv, activation=activation)


def _conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    *,
    groups: int = 1,
    activation: bool = True,
) -> nn.Module:
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
    )
    return _block(conv, activation=activation)


def _block(conv: nn.Conv1d | nn.Conv2d, *, activation: bool) -> nn.Module:
    layers: list[nn.Module] = [parametrizations.weight_norm(conv)]
    if activation:
        layers.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*layers)
