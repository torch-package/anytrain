from __future__ import annotations

import os
from typing import Any, Union

import torch
from torch import Tensor, nn
from typing_extensions import TypeAlias

from .._audio import model_frame_rate, resample
from .._module import DeviceModule
from .assets import UniCodecAssets, ensure_unicodec_assets

Domain: TypeAlias = Union[str, int]
SAMPLE_RATE = 24_000
NUM_CHANNELS = 1
DEFAULT_CODEBOOK_SIZE = 16_384


class UniCodec(DeviceModule):
    name = "unicodec"

    def __init__(
        self,
        model: nn.Module,
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
        device: str | torch.device | None = None,
        domain: Domain = "0",
        bandwidth_id: int = 0,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> UniCodec:
        kwargs: dict[str, object] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }
        if repo_id is not None:
            kwargs["repo_id"] = repo_id
        if config_name is not None:
            kwargs["config_name"] = config_name
        if checkpoint_filename is not None:
            kwargs["checkpoint_filename"] = checkpoint_filename

        assets = ensure_unicodec_assets(**kwargs)
        model_cls = _load_unicodec_model()
        resolved_device = _device(device)
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
        self._validate_codes(codes)
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
        cond = _bandwidth_id(self.bandwidth_id, self.device)
        audio = self.model.decode(
            features.transpose(1, 2).to(self.device),
            bandwidth_id=cond,
        )
        return audio.unsqueeze(1)

    @torch.no_grad()
    def codes_to_features(self, codes: Tensor) -> Tensor:
        self._validate_codes(codes)
        backend_codes = codes.permute(2, 0, 1).contiguous().to(self.device)
        return self.model.codes_to_features(backend_codes).transpose(1, 2)

    @torch.no_grad()
    def reconstruct(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        return self.decode(self.encode(audio, sample_rate))

    def _validate_codes(self, codes: Tensor) -> None:
        if codes.dim() != 3:
            raise ValueError("codes must have shape [batch, time, codebook].")
        if codes.shape[-1] != len(self.codebook_sizes):
            raise ValueError(
                f"codes must contain {len(self.codebook_sizes)} aligned codebooks."
            )
        if codes.dtype == torch.bool or torch.is_floating_point(codes) or torch.is_complex(codes):
            raise TypeError("codes must contain integer ids.")

def _domain(domain: Domain) -> str:
    value = str(domain)
    if value not in {"0", "1", "2"}:
        raise ValueError("UniCodec domain must be one of '0', '1', or '2'.")
    return value


def _bandwidth_id(bandwidth_id: int, device: torch.device) -> Tensor:
    return torch.tensor([bandwidth_id], device=device)


def _codebook_sizes(model: nn.Module) -> tuple[int, ...]:
    feature_extractor = getattr(model, "feature_extractor", None)
    encodec = getattr(feature_extractor, "encodec", None)
    quantizer = getattr(encodec, "quantizer", None)
    num_codebooks = int(getattr(quantizer, "n_q", 1))
    codebook_size = int(getattr(quantizer, "bins", DEFAULT_CODEBOOK_SIZE))
    return (codebook_size,) * num_codebooks


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_unicodec_model() -> Any:
    try:
        from unicodec import Unicodec
    except ImportError as exc:
        raise ImportError(
            "UniCodec requires the installable UniCodec fork. Install it with "
            '`python -m pip install "unicodec @ git+https://github.com/rookiejune/UniCodec.git"`.'
        ) from exc
    return Unicodec
