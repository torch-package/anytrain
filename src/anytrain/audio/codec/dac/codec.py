from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .._audio import model_frame_rate, resample
from .._module import DeviceModule, resolve_device, validate_frame_codes
from .assets import (
    DEFAULT_MODEL_BITRATE,
    DEFAULT_MODEL_TYPE,
    DACAssets,
    ModelBitrate,
    ModelType,
    ensure_dac_assets,
)

NUM_CHANNELS = 1


class _DACCodebook(Protocol):
    def decode_code(self, codes: Tensor) -> Tensor: ...

    @property
    def out_proj(self) -> object: ...


class _DACQuantizer(Protocol):
    def from_codes(self, codes: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

    @property
    def quantizers(self) -> list[_DACCodebook]: ...


class _DACModel(Protocol):
    sample_rate: int
    n_codebooks: int
    codebook_size: int
    quantizer: _DACQuantizer

    def preprocess(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def encode(self, audio: Tensor, *, n_quantizers: int) -> tuple[Tensor, ...]: ...

    def decode(self, features: Tensor) -> Tensor: ...

    def to(self, device: torch.device) -> _DACModel: ...

    def eval(self) -> _DACModel: ...


class DAC(DeviceModule):
    name = "dac"

    def __init__(
        self,
        model: _DACModel,
        device: torch.device,
        *,
        checkpoint: Path,
        assets: DACAssets | None = None,
        n_quantizers: int | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.checkpoint = checkpoint
        self.assets = assets
        self.sample_rate = int(model.sample_rate)

        total = int(model.n_codebooks)
        self.n_quantizers = total if n_quantizers is None else n_quantizers
        if not 1 <= self.n_quantizers <= total:
            raise ValueError(f"n_quantizers must be between 1 and {total}.")
        self.codebook_sizes = (int(model.codebook_size),) * self.n_quantizers
        self._init_device(device)

    @property
    def frame_rate(self) -> float:
        return model_frame_rate(self.model, self.sample_rate)

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        model_type: ModelType = DEFAULT_MODEL_TYPE,
        model_bitrate: ModelBitrate = DEFAULT_MODEL_BITRATE,
        tag: str = "latest",
        device: str | torch.device | None = None,
        n_quantizers: int | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        allow_unsafe_download: bool = False,
    ) -> DAC:
        assets = ensure_dac_assets(
            cache_dir,
            model_type=model_type,
            model_bitrate=model_bitrate,
            tag=tag,
            local_files_only=local_files_only,
            force_download=force_download,
            allow_unsafe_download=allow_unsafe_download,
        )
        resolved_device = resolve_device(device)
        model = _load_checkpoint(assets["checkpoint"], resolved_device)
        return cls(
            model=model,
            device=resolved_device,
            checkpoint=assets["checkpoint"],
            assets=assets,
            n_quantizers=n_quantizers,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | os.PathLike[str],
        *,
        device: str | torch.device | None = None,
        n_quantizers: int | None = None,
    ) -> DAC:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DAC checkpoint does not exist: {path}.")

        resolved_device = resolve_device(device)
        model = _load_checkpoint(path, resolved_device)
        return cls(
            model=model,
            device=resolved_device,
            checkpoint=path,
            n_quantizers=n_quantizers,
        )

    @torch.no_grad()
    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        _, codes = self.encode_features(audio, sample_rate)
        return codes

    @torch.no_grad()
    def encode_features(self, audio: Tensor, sample_rate: int) -> tuple[Tensor, Tensor]:
        if audio.dim() != 3 or audio.size(1) != NUM_CHANNELS:
            raise ValueError("DAC encode expects mono audio shape [batch, 1, time].")

        audio = resample(audio, sample_rate, self.sample_rate).to(self.device)
        audio = self.model.preprocess(audio, self.sample_rate)
        features, codes, *_ = self.model.encode(audio, n_quantizers=self.n_quantizers)
        codes = codes.transpose(1, 2).contiguous()
        validate_frame_codes(codes, self.codebook_sizes)
        return features.transpose(1, 2), codes

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        return self.decode_features(self.codes_to_features(codes))

    @torch.no_grad()
    def codes_to_features(self, codes: Tensor) -> Tensor:
        validate_frame_codes(codes, self.codebook_sizes)
        backend_codes = codes.transpose(1, 2).contiguous().to(self.device)
        features, _, _ = self.model.quantizer.from_codes(backend_codes)
        return features.transpose(1, 2)

    @torch.no_grad()
    def codebook_codes_to_features(self, index: int, codes: Tensor) -> Tensor:
        if not 0 <= index < self.n_quantizers:
            raise ValueError(f"DAC codebook index must be in [0, {self.n_quantizers}).")
        if codes.dim() != 2:
            raise ValueError("DAC codebook codes must have shape [batch, time].")
        quantizers = getattr(self.model.quantizer, "quantizers", None)
        if quantizers is None or len(quantizers) <= index:
            raise TypeError(f"DAC quantizer must expose codebook {index}.")
        quantizer = quantizers[index]
        decode_code = getattr(quantizer, "decode_code", None)
        out_proj = getattr(quantizer, "out_proj", None)
        if not callable(decode_code) or not callable(out_proj):
            raise TypeError("DAC codebooks must provide decode_code() and out_proj().")
        features = out_proj(decode_code(codes.to(self.device)))
        if not isinstance(features, Tensor):
            raise TypeError("DAC codebook projection must return a Tensor.")
        return features.transpose(1, 2)

    @torch.no_grad()
    def decode_codebook_features(self, features: tuple[Tensor, ...]) -> Tensor:
        if len(features) != self.n_quantizers:
            raise ValueError("DAC features must contain every configured codebook stream.")
        combined = features[0]
        for value in features[1:]:
            if value.shape != combined.shape:
                raise ValueError("DAC codebook features must share one shape.")
            combined = combined + value
        return self.decode_features(combined)

    @torch.no_grad()
    def decode_features(self, features: Tensor) -> Tensor:
        if features.dim() != 3:
            raise ValueError("DAC features must have shape [batch, time, dim].")
        if not torch.is_floating_point(features) or torch.is_complex(features):
            raise TypeError("DAC features must be floating point tensors.")
        return self.model.decode(features.transpose(1, 2).to(self.device))

def _load_checkpoint(checkpoint: Path, device: torch.device) -> _DACModel:
    model_cls = _load_dac_model()
    model = model_cls.load(checkpoint)
    model = model.to(device)
    model.eval()
    return model
def _load_dac_model() -> Any:
    try:
        import dac
    except ImportError as exc:
        raise ImportError(
            'DAC requires the `descript-audio-codec` package. Install it with `python -m pip install "descript-audio-codec>=1.0"`.'
        ) from exc
    return dac.DAC


__all__ = ["DAC", "NUM_CHANNELS"]
