from __future__ import annotations

import os
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .._module import DeviceModule, integer_tensor, resolve_device, validate_frame_codes
from ..protocol import AcousticLayout, SemanticAcousticCodes
from .assets import (
    DECODER_SPECS,
    LongCatAssets,
    LongCatDecoderName,
    ensure_longcat_assets,
)

DEFAULT_DECODER: LongCatDecoderName = "16k_4codebooks"
SEMANTIC_CODEBOOK_SIZE = 8192
ACOUSTIC_FACTOR_CODEBOOK_SIZE = 90
NUM_CHANNELS = 1


class LongCat(DeviceModule):
    name = "longcat"
    acoustic_layout = AcousticLayout.FRAME_ALIGNED

    def __init__(
        self,
        encoder: Any,
        decoders: dict[str, nn.Module],
        device: torch.device,
        assets: LongCatAssets,
        decoder: LongCatDecoderName = DEFAULT_DECODER,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        self.assets = assets
        self.decoder = decoder
        model = self._decoder()
        spec = DECODER_SPECS[decoder]
        num_codebooks = int(getattr(model, "n_codebooks", spec.codebooks - 1)) + 1
        acoustic_factor_size = int(
            getattr(
                model,
                "acoustic_codebook_size",
                ACOUSTIC_FACTOR_CODEBOOK_SIZE,
            )
        )
        acoustic_size = acoustic_factor_size**2
        self.sample_rate = spec.sample_rate
        self.codebook_sizes = (SEMANTIC_CODEBOOK_SIZE,) + (acoustic_size,) * (num_codebooks - 1)
        self.semantic_codebook_sizes = self.codebook_sizes[:1]
        self.acoustic_codebook_sizes = self.codebook_sizes[1:]
        self._init_device(device)

    @property
    def acoustic_unit_length(self) -> int | None:
        """Frame-aligned LongCat units have no independent fixed acoustic length."""
        return None

    @property
    def semantic_frame_rate(self) -> float:
        encoder = self.encoder
        sample_rate = getattr(encoder, "input_sample_rate", None)
        hop_length = getattr(encoder, "hop_length", None)
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, Integral)
            or isinstance(hop_length, bool)
            or not isinstance(hop_length, Integral)
        ):
            raise TypeError("LongCat encoder must expose positive input_sample_rate and hop_length.")
        sample_rate = int(sample_rate)
        hop_length = int(hop_length)
        if sample_rate <= 0 or hop_length <= 0:
            raise TypeError("LongCat encoder must expose positive input_sample_rate and hop_length.")
        return float(sample_rate / hop_length)

    @property
    def frame_rate(self) -> float:
        return self.semantic_frame_rate

    @property
    def acoustic_feature_dim(self) -> int:
        value = getattr(self._decoder(), "latent_dim", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError("LongCat decoder must expose a positive integer latent_dim.")
        return value

    @property
    def semantic_codebook(self) -> Tensor:
        codebook = self.encoder.get_semantic_codebook()
        if not isinstance(codebook, Tensor):
            raise TypeError("LongCat semantic codebook must be a Tensor.")
        if codebook.dim() != 2:
            raise ValueError("LongCat semantic codebook must have shape [dim, vocab].")
        return codebook.transpose(0, 1).contiguous().to(self.device)

    @property
    def semantic_feature_dim(self) -> int:
        return int(self.semantic_codebook.size(-1))

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        decoder: LongCatDecoderName = DEFAULT_DECODER,
        revision: str | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        allow_unpinned_revision: bool = False,
    ) -> LongCat:
        load_encoder, load_decoder = _load_longcat_loaders()
        decoder_names = (decoder,)
        assets = ensure_longcat_assets(
            cache_dir=cache_dir,
            decoders=decoder_names,
            revision=revision,
            local_files_only=local_files_only,
            force_download=force_download,
            allow_unpinned_revision=allow_unpinned_revision,
        )
        resolved_device = resolve_device(device)

        with _longcat_checkpoint_env(assets.ckpt_dir):
            encoder = load_encoder(str(assets.configs.encoder), resolved_device)
            loaded_decoders: dict[str, nn.Module] = {
                name: load_decoder(str(assets.configs.decoder(name)), resolved_device)
                for name in decoder_names
            }

        return cls(
            encoder=encoder,
            decoders=loaded_decoders,
            device=resolved_device,
            assets=assets,
            decoder=decoder,
        )

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        codes = self.tokenize(audio, sample_rate)
        return torch.cat((codes.semantic, codes.acoustic), dim=-1)

    @torch.no_grad()
    def tokenize(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> SemanticAcousticCodes:
        if audio.dim() != 3 or audio.size(1) != NUM_CHANNELS:
            raise ValueError("LongCat tokenize expects mono audio shape [batch, 1, time].")
        backend_codes = self.encoder(
            audio.to(self.device),
            sample_rate,
            n_acoustic_codebooks=len(self.acoustic_codebook_sizes),
        )
        semantic = integer_tensor(backend_codes[0], "LongCat semantic codes")
        acoustic = integer_tensor(backend_codes[1], "LongCat acoustic codes")
        if semantic.dim() != 2:
            raise ValueError("LongCat semantic encoder output must have shape [batch, time].")
        if acoustic.dim() != 3:
            raise ValueError(
                "LongCat acoustic encoder output must have shape [batch, codebook, time]."
            )

        codes = SemanticAcousticCodes(
            semantic=semantic.unsqueeze(-1),
            acoustic=acoustic.transpose(1, 2).contiguous(),
        )
        self._validate_tokens(codes)
        return codes

    @torch.no_grad()
    def decode(
        self,
        codes: Tensor,
    ) -> Tensor:
        return self.detokenize(self._split_codes(codes))

    @torch.no_grad()
    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        self._validate_tokens(codes)
        model = self._decoder()
        return model(
            codes.semantic[..., 0].to(self.device),
            codes.acoustic.transpose(1, 2).contiguous().to(self.device),
        )

    @torch.no_grad()
    def acoustic_codes_to_features(
        self,
        acoustic_codes: Tensor,
    ) -> Tensor:
        acoustic_codes = integer_tensor(acoustic_codes, "acoustic_codes")
        if acoustic_codes.dim() != 3:
            raise ValueError("acoustic_codes must have shape [batch, time, codebook].")
        num_codebooks = len(self.codebook_sizes) - 1
        if acoustic_codes.shape[-1] != num_codebooks:
            raise ValueError(f"acoustic_codes must contain {num_codebooks} aligned codebooks.")

        model = self._decoder()
        convert = getattr(model, "acoustic_codes_to_latents", None)
        if not callable(convert):
            raise TypeError("LongCat decoder must provide acoustic_codes_to_latents().")
        backend_codes = acoustic_codes.transpose(1, 2).contiguous().to(self.device)
        features = convert(backend_codes)
        if not isinstance(features, Tensor):
            raise TypeError("LongCat acoustic_codes_to_latents() must return a Tensor.")
        if features.dim() != 3:
            raise ValueError("LongCat acoustic features must have shape [batch, dim, time].")
        if not torch.is_floating_point(features) or torch.is_complex(features):
            raise TypeError("LongCat acoustic features must be floating point tensors.")
        return features.transpose(1, 2)

    @torch.no_grad()
    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        semantic_codes = integer_tensor(semantic_codes, "semantic_codes")
        if semantic_codes.dim() != 3:
            raise ValueError("semantic_codes must have shape [batch, time, codebook].")
        if semantic_codes.shape[-1] != 1:
            raise ValueError("semantic_codes must contain 1 aligned codebook.")
        if acoustic_features.dim() != 3:
            raise ValueError("acoustic_features must have shape [batch, time, dim].")
        if acoustic_features.shape[:2] != semantic_codes.shape[:2]:
            raise ValueError("semantic_codes and acoustic_features must align on batch and time.")
        if not torch.is_floating_point(acoustic_features) or torch.is_complex(acoustic_features):
            raise TypeError("acoustic_features must be floating point tensors.")

        model = self._decoder()
        return model(
            semantic_codes[..., 0].to(self.device),
            acoustic_features.to(self.device).transpose(1, 2),
        )

    def _decoder(self) -> Any:
        try:
            return self.decoders[self.decoder]
        except KeyError as exc:
            available = ", ".join(sorted(self.decoders))
            raise ValueError(
                f"Decoder {self.decoder!r} is not loaded. Available decoders: {available}."
            ) from exc

    def _split_codes(self, codes: Tensor) -> SemanticAcousticCodes:
        validate_frame_codes(codes, self.codebook_sizes)
        return SemanticAcousticCodes(
            semantic=codes[..., :1],
            acoustic=codes[..., 1:],
        )

    def _validate_tokens(self, codes: SemanticAcousticCodes) -> None:
        semantic = integer_tensor(codes.semantic, "semantic codes")
        acoustic = integer_tensor(codes.acoustic, "acoustic codes")
        if semantic.dim() != 3:
            raise ValueError("semantic codes must have shape [batch, time, codebook].")
        if semantic.shape[-1] != len(self.semantic_codebook_sizes):
            raise ValueError(
                f"semantic codes must contain {len(self.semantic_codebook_sizes)} codebooks."
            )
        if acoustic.dim() != 3:
            raise ValueError("acoustic codes must have shape [batch, time, codebook].")
        if acoustic.shape[-1] != len(self.acoustic_codebook_sizes):
            raise ValueError(
                f"acoustic codes must contain {len(self.acoustic_codebook_sizes)} codebooks."
            )
        if semantic.shape[:2] != acoustic.shape[:2]:
            raise ValueError("semantic and acoustic codes must align on batch and time.")


def _load_longcat_loaders():
    try:
        from longcat_audio_codec import load_decoder, load_encoder
    except ImportError as exc:
        raise ImportError(
            "LongCat requires the installable LongCat fork. Install it with "
            '`python -m pip install "longcat-audio-codec @ git+https://github.com/rookiejune/LongCat-Audio-Codec.git"`.'
        ) from exc
    return load_encoder, load_decoder


@contextmanager
def _longcat_checkpoint_env(ckpt_dir: Path):
    name = "LONGCAT_AUDIO_CODEC_CKPT_DIR"
    old = os.environ.get(name)
    os.environ[name] = str(ckpt_dir)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old
