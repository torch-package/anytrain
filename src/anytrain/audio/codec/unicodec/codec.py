from __future__ import annotations

import os
from typing import Any, Protocol

import torch
from torch import Tensor

from .._audio import model_frame_rate, resample
from .._module import DeviceModule, resolve_device, validate_frame_codes
from .assets import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CONFIG_NAME,
    DEFAULT_HF_REPO_ID,
    UniCodecAssets,
    ensure_unicodec_assets,
)

type Domain = str | int
SAMPLE_RATE = 24_000
NUM_CHANNELS = 1
DEFAULT_CODEBOOK_SIZE = 16_384


class _UniCodecModel(Protocol):
    feature_extractor: object

    def encode_infer(
        self,
        audio: Tensor,
        domains: tuple[str, ...],
        *,
        bandwidth_id: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def decode(self, features: Tensor, *, bandwidth_id: Tensor) -> Tensor: ...

    def codes_to_features(self, codes: Tensor) -> Tensor: ...


class UniCodec(DeviceModule):
    name = "unicodec"

    def __init__(
        self,
        model: _UniCodecModel,
        device: torch.device,
        assets: UniCodecAssets,
        *,
        domain: Domain = "0",
        bandwidth_id: int = 0,
    ) -> None:
        super().__init__()

        self.model = model
        self.assets = assets
        self.domain = _domain(domain)
        self.bandwidth_id = bandwidth_id
        self.sample_rate = SAMPLE_RATE
        self.codebook_sizes = _codebook_sizes(model)
        self._init_device(device)

    @property
    def frame_rate(self) -> float:
        return model_frame_rate(self.model.feature_extractor, self.sample_rate)

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        repo_id: str | None = None,
        config_name: str | None = None,
        checkpoint_filename: str | None = None,
        revision: str | None = None,
        device: str | torch.device | None = None,
        domain: Domain = "0",
        bandwidth_id: int = 0,
        local_files_only: bool = False,
        force_download: bool = False,
        allow_unpinned_revision: bool = False,
    ) -> UniCodec:
        assets = ensure_unicodec_assets(
            cache_dir=cache_dir,
            repo_id=repo_id if repo_id is not None else DEFAULT_HF_REPO_ID,
            config_name=config_name if config_name is not None else DEFAULT_CONFIG_NAME,
            checkpoint_filename=(
                checkpoint_filename
                if checkpoint_filename is not None
                else DEFAULT_CHECKPOINT_FILENAME
            ),
            revision=revision,
            local_files_only=local_files_only,
            force_download=force_download,
            allow_unpinned_revision=allow_unpinned_revision,
        )
        model_cls = _load_unicodec_model()
        resolved_device = resolve_device(device)
        model = model_cls.from_pretrained0802(
            str(assets["config"]),
            str(assets["checkpoint"]),
        )
        model = model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            device=resolved_device,
            assets=assets,
            domain=domain,
            bandwidth_id=bandwidth_id,
        )

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        _, codes = self.encode_features(audio, sample_rate)
        return codes

    @torch.no_grad()
    def encode_features(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> tuple[Tensor, Tensor]:
        if audio.dim() != 3 or audio.size(1) != NUM_CHANNELS:
            raise ValueError("UniCodec encode expects mono audio shape [batch, 1, time].")

        audio = resample(audio, sample_rate, self.sample_rate).squeeze(1)
        domains = (self.domain,) * audio.size(0)
        cond = _bandwidth_id(self.bandwidth_id, self.device)
        features, codes = self.model.encode_infer(
            audio.to(self.device),
            domains,
            bandwidth_id=cond,
        )
        codes = codes.permute(1, 2, 0).contiguous()
        validate_frame_codes(codes, self.codebook_sizes)
        return features.transpose(1, 2), codes

    @torch.no_grad()
    def decode(
        self,
        codes: Tensor,
    ) -> Tensor:
        return self.decode_features(self.codes_to_features(codes))

    @torch.no_grad()
    def decode_features(
        self,
        features: Tensor,
    ) -> Tensor:
        if features.dim() != 3:
            raise ValueError("UniCodec features must have shape [batch, time, dim].")
        if not torch.is_floating_point(features) or torch.is_complex(features):
            raise TypeError("UniCodec features must be floating point tensors.")
        cond = _bandwidth_id(self.bandwidth_id, self.device)
        audio = self.model.decode(
            features.transpose(1, 2).to(self.device),
            bandwidth_id=cond,
        )
        return audio.unsqueeze(1)

    @torch.no_grad()
    def codes_to_features(self, codes: Tensor) -> Tensor:
        validate_frame_codes(codes, self.codebook_sizes)
        backend_codes = codes.permute(2, 0, 1).contiguous().to(self.device)
        return self.model.codes_to_features(backend_codes).transpose(1, 2)

    @torch.no_grad()
    def codebook_codes_to_features(self, index: int, codes: Tensor) -> Tensor:
        if index != 0 or len(self.codebook_sizes) != 1:
            raise ValueError("UniCodec detokenizer preset requires exactly one codebook.")
        if codes.dim() != 2:
            raise ValueError("UniCodec codebook codes must have shape [batch, time].")
        return self.codes_to_features(codes[..., None])

    @torch.no_grad()
    def decode_codebook_features(self, features: tuple[Tensor, ...]) -> Tensor:
        if len(features) != 1:
            raise ValueError("UniCodec features must contain exactly one codebook stream.")
        return self.decode_features(features[0])

def _domain(domain: Domain) -> str:
    value = str(domain)
    if value not in {"0", "1", "2"}:
        raise ValueError("UniCodec domain must be one of '0', '1', or '2'.")
    return value


def _bandwidth_id(bandwidth_id: int, device: torch.device) -> Tensor:
    return torch.tensor([bandwidth_id], device=device)


def _codebook_sizes(model: object) -> tuple[int, ...]:
    feature_extractor = getattr(model, "feature_extractor", None)
    encodec = getattr(feature_extractor, "encodec", None)
    quantizer = getattr(encodec, "quantizer", None)
    if quantizer is None:
        raise TypeError("UniCodec model is missing feature_extractor.encodec.quantizer.")
    if not hasattr(quantizer, "n_q") or not hasattr(quantizer, "bins"):
        raise TypeError("UniCodec quantizer must expose n_q and bins.")
    num_codebooks = int(quantizer.n_q)
    codebook_size = int(quantizer.bins)
    if num_codebooks <= 0 or codebook_size <= 0:
        raise ValueError("UniCodec quantizer n_q and bins must be positive.")
    return (codebook_size,) * num_codebooks


def _load_unicodec_model() -> Any:
    try:
        from unicodec import Unicodec  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "UniCodec requires the installable UniCodec fork. Install it with "
            '`python -m pip install "unicodec @ git+https://github.com/rookiejune/UniCodec.git"`.'
        ) from exc
    return Unicodec
