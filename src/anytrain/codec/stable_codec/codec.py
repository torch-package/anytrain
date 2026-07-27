from __future__ import annotations

import os
from collections.abc import Sequence
from math import prod
from typing import Any, Literal, Union

import torch
from torch import Tensor, nn
from typing_extensions import TypeAlias

from .._audio import model_frame_rate, resample
from .._module import DeviceModule

SupportedVersion = Literal["speech-16k", "speech-16k-base"]
PosthocBottleneckPreset = Literal[
    "1x46656_400bps",
    "2x15625_700bps",
    "4x729_1000bps",
]
PosthocBottleneck: TypeAlias = Union[
    PosthocBottleneckPreset,
    Sequence[tuple[Sequence[int], float]],
]

DEFAULT_VERSION: SupportedVersion = "speech-16k"
DEFAULT_PRETRAINED_MODEL = f"stabilityai/stable-codec-{DEFAULT_VERSION}"
DEFAULT_POSTHOC_BOTTLENECK: PosthocBottleneckPreset = "1x46656_400bps"
SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
DEFAULT_CODEBOOK_SIZE = 17**6
POSTHOC_CODEBOOK_SIZES: dict[PosthocBottleneckPreset, tuple[int, ...]] = {
    "1x46656_400bps": (46_656,),
    "2x15625_700bps": (15_625, 15_625),
    "4x729_1000bps": (729, 729, 729, 729),
}


class StableCodec(DeviceModule):
    name = "stable_codec"
    num_channels: int = NUM_CHANNELS
    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        posthoc_bottleneck: PosthocBottleneck | None = DEFAULT_POSTHOC_BOTTLENECK,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        if posthoc_bottleneck is not None:
            self.model.set_posthoc_bottleneck(posthoc_bottleneck)
        self.posthoc_bottleneck = posthoc_bottleneck is not None
        self.normalize = normalize
        self.sample_rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
        self.codebook_sizes = _codebook_sizes(model, posthoc_bottleneck)
        self._init_device(device)

    @property
    def frame_rate(self) -> float:
        return model_frame_rate(self.model, self.sample_rate)

    @classmethod
    def from_pretrained(
        cls,
        version: SupportedVersion = DEFAULT_VERSION,
        *,
        pretrained_model: str | None = None,
        device: str | torch.device | None = None,
        posthoc_bottleneck: PosthocBottleneck | None = DEFAULT_POSTHOC_BOTTLENECK,
        normalize: bool = True,
    ) -> StableCodec:
        model_cls = _load_stable_codec_model()
        resolved_device = _device(device)
        model = model_cls(
            pretrained_model=pretrained_model or f"stabilityai/stable-codec-{version}",
            device=resolved_device,
        )
        return cls(
            model=model,
            device=resolved_device,
            posthoc_bottleneck=posthoc_bottleneck,
            normalize=normalize,
        )

    @classmethod
    def from_config(
        cls,
        model_config_path: str | os.PathLike[str],
        *,
        ckpt_path: str | os.PathLike[str] | None = None,
        device: str | torch.device | None = None,
        posthoc_bottleneck: PosthocBottleneck | None = DEFAULT_POSTHOC_BOTTLENECK,
        normalize: bool = True,
    ) -> StableCodec:
        model_cls = _load_stable_codec_model()
        resolved_device = _device(device)
        model = model_cls(
            model_config_path=str(model_config_path),
            ckpt_path=None if ckpt_path is None else str(ckpt_path),
            device=resolved_device,
        )
        return cls(
            model=model,
            device=resolved_device,
            posthoc_bottleneck=posthoc_bottleneck,
            normalize=normalize,
        )

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        _, tokens = self.encode_latents(
            audio,
            sample_rate,
        )
        return tokens

    @torch.no_grad()
    def encode_latents(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> tuple[Tensor, Tensor]:
        if audio.dim() != 3:
            raise ValueError("StableCodec encode expects audio shape [batch, channels, time].")
        if audio.size(1) != NUM_CHANNELS:
            raise ValueError("StableCodec speech-16k expects mono audio with shape [batch, 1, time].")

        audio = resample(audio, sample_rate, self.sample_rate)
        latents, tokens = self.model.encode(
            audio.to(self.device),
            posthoc_bottleneck=self.posthoc_bottleneck,
            normalize=self.normalize,
        )
        tokens = _codes(tokens, posthoc_bottleneck=self.posthoc_bottleneck)
        self._validate_codes(tokens)
        return latents, tokens

    @torch.no_grad()
    def decode(
        self,
        codes: Tensor,
    ) -> Tensor:
        self._validate_codes(codes)
        backend_tokens: Tensor | list[Tensor]
        if self.posthoc_bottleneck:
            backend_tokens = list(codes.to(self.device).split(1, dim=-1))
        else:
            backend_tokens = codes.to(self.device)
        return self.model.decode(
            backend_tokens,
            posthoc_bottleneck=self.posthoc_bottleneck,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        return self.decode(self.encode(audio, sample_rate))

    def set_posthoc_bottleneck(self, stages: PosthocBottleneck) -> None:
        self.model.set_posthoc_bottleneck(stages)
        self.posthoc_bottleneck = True
        self.codebook_sizes = _posthoc_codebook_sizes(stages)

    def _validate_codes(self, codes: Tensor) -> None:
        if codes.dim() != 3:
            raise ValueError("codes must have shape [batch, time, codebook].")
        if codes.shape[-1] != len(self.codebook_sizes):
            raise ValueError(
                f"codes must contain {len(self.codebook_sizes)} aligned codebooks."
            )
        if codes.dtype == torch.bool or torch.is_floating_point(codes) or torch.is_complex(codes):
            raise TypeError("codes must contain integer ids.")


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_stable_codec_model() -> Any:
    try:
        from stable_codec import StableCodec as UpstreamStableCodec
    except ImportError as exc:
        raise ImportError(
            "StableCodec requires the `stable-codec` package. Install "
            "`stable-codec` in a compatible environment before using this wrapper."
        ) from exc
    return UpstreamStableCodec


def _codebook_sizes(
    model: Any,
    posthoc_bottleneck: PosthocBottleneck | None,
) -> tuple[int, ...]:
    if posthoc_bottleneck is not None:
        return _posthoc_codebook_sizes(posthoc_bottleneck)

    quantizer = model.model.bottleneck.quantizer
    return (int(quantizer.codebook_size),) * int(quantizer.num_codebooks)


def _codes(
    tokens: Tensor | list[Tensor],
    *,
    posthoc_bottleneck: bool,
) -> Tensor:
    if not posthoc_bottleneck:
        if not isinstance(tokens, Tensor):
            raise TypeError("Stable Codec native encode must return a Tensor.")
        return tokens

    if not isinstance(tokens, list):
        raise TypeError("Stable Codec posthoc encode must return a list of Tensors.")
    if not tokens:
        raise ValueError("Stable Codec posthoc encode returned no codebooks.")
    if any(not isinstance(token, Tensor) for token in tokens):
        raise TypeError("Stable Codec posthoc encode must return a list of Tensors.")
    if any(token.dim() != 3 or token.shape[-1] != 1 for token in tokens):
        raise ValueError(
            "Each Stable Codec posthoc codebook must have shape [batch, time, 1]."
        )
    if any(token.shape[:2] != tokens[0].shape[:2] for token in tokens[1:]):
        raise ValueError("Stable Codec posthoc codebooks must align on batch and time.")
    return torch.cat(tokens, dim=-1)


def _posthoc_codebook_sizes(stages: PosthocBottleneck) -> tuple[int, ...]:
    if isinstance(stages, str):
        return POSTHOC_CODEBOOK_SIZES[stages]
    return tuple(prod(levels) for levels, _ in stages)
