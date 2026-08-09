from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .._module import DeviceModule, integer_tensor, resolve_device, validate_frame_codes
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

    def semantic_codes_to_features(self, semantic_codes: Tensor) -> Tensor:
        semantic = integer_tensor(semantic_codes, "semantic_codes")
        if semantic.dim() != 2:
            raise ValueError("semantic_codes must have shape [batch, time].")
        if bool((semantic < 0).any()) or bool((semantic >= self.codebook_sizes[0]).any()):
            raise ValueError("semantic_codes contain an ID outside the semantic codebook.")
        convert = getattr(self._decoder(), "semantic_dequantizer", None)
        if not callable(convert):
            raise TypeError("LongCat decoder must provide semantic_dequantizer().")
        features = convert(semantic.to(self.device))
        return _feature_tensor(features, name="semantic features", shape=semantic.shape)

    @torch.no_grad()
    def acoustic_codebook_codes_to_features(self, index: int, codes: Tensor) -> Tensor:
        if not 0 <= index < len(self.acoustic_codebook_sizes):
            raise ValueError("LongCat acoustic codebook index is outside the configured range.")
        values = integer_tensor(codes, "acoustic codebook codes")
        if values.dim() != 2:
            raise ValueError("acoustic codebook codes must have shape [batch, time].")
        if bool((values < 0).any()) or bool(
            (values >= self.acoustic_codebook_sizes[index]).any()
        ):
            raise ValueError("acoustic codebook codes contain an ID outside the codebook.")
        quantizer = self._acoustic_quantizer(index)
        size_b = _positive_int(quantizer.codebook_size_b, "acoustic codebook_size_b")
        code_a = torch.div(values, size_b, rounding_mode="floor").to(self.device)
        code_b = values.remainder(size_b).to(self.device)
        decode_code = getattr(quantizer, "decode_code", None)
        if not callable(decode_code):
            raise TypeError("LongCat acoustic quantizers must provide decode_code().")
        feature_a = quantizer.out_proj_a(decode_code(code_a, quantizer.codebook_a))
        feature_b = quantizer.out_proj_b(decode_code(code_b, quantizer.codebook_b))
        features = torch.cat((feature_a, feature_b), dim=1).transpose(1, 2)
        return _feature_tensor(features, name="acoustic features", shape=values.shape)

    @torch.no_grad()
    def decode_feature_streams(
        self,
        semantic_features: Tensor,
        acoustic_features: tuple[Tensor, ...],
    ) -> Tensor:
        semantic = _feature_tensor(semantic_features, name="semantic features")
        if not acoustic_features:
            raise ValueError("LongCat decode requires at least one acoustic feature stream.")
        acoustic = _feature_tensor(
            acoustic_features[0],
            name="acoustic features",
            shape=semantic.shape[:2],
        )
        for value in acoustic_features[1:]:
            prepared = _feature_tensor(
                value,
                name="acoustic features",
                shape=semantic.shape[:2],
            )
            if prepared.size(-1) != acoustic.size(-1):
                raise ValueError("LongCat acoustic feature streams must share a feature dim.")
            acoustic = acoustic + prepared
        merged = torch.cat((semantic, acoustic), dim=-1)
        decoder = getattr(self._decoder(), "decoder", None)
        if not callable(decoder):
            raise TypeError("LongCat decoder must provide its waveform decoder module.")
        audio = decoder(merged.transpose(1, 2).to(self.device))
        if not isinstance(audio, Tensor):
            raise TypeError("LongCat waveform decoder must return a Tensor.")
        return audio

    @torch.no_grad()
    def decode_acoustic_features(
        self,
        semantic_features: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        semantic = _feature_tensor(semantic_features, name="semantic features")
        acoustic = _feature_tensor(
            acoustic_features,
            name="acoustic features",
            shape=semantic.shape[:2],
        )
        if acoustic.size(-1) == self.acoustic_feature_dim:
            return self.decode_feature_streams(semantic, (acoustic,))

        projected: list[Tensor] = []
        offset = 0
        for index in range(len(self.acoustic_codebook_sizes)):
            quantizer = self._acoustic_quantizer(index)
            dim_a = _embedding_dim(quantizer.codebook_a, "acoustic codebook_a")
            dim_b = _embedding_dim(quantizer.codebook_b, "acoustic codebook_b")
            end = offset + dim_a + dim_b
            if end > acoustic.size(-1):
                break
            feature_a = acoustic[..., offset : offset + dim_a].transpose(1, 2)
            feature_b = acoustic[..., offset + dim_a : end].transpose(1, 2)
            projected.append(
                torch.cat(
                    (quantizer.out_proj_a(feature_a), quantizer.out_proj_b(feature_b)),
                    dim=1,
                ).transpose(1, 2)
            )
            offset = end
            if offset == acoustic.size(-1):
                return self.decode_feature_streams(semantic, tuple(projected))
        raise ValueError(
            "LongCat acoustic features must match the native feature dim or a complete "
            "prefix of factor codebook feature dims."
        )

    def _acoustic_quantizer(self, index: int) -> Any:
        acoustic = getattr(self._decoder(), "acoustic_quantizer", None)
        quantizers = getattr(acoustic, "quantizers", None)
        if quantizers is None or not 0 <= index < len(quantizers):
            raise TypeError(f"LongCat decoder is missing acoustic quantizer {index}.")
        return quantizers[index]

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        if audio.dim() != 3 or audio.size(1) != NUM_CHANNELS:
            raise ValueError("LongCat encode expects mono audio shape [batch, 1, time].")
        trace = _trace_start(audio, sample_rate, self.device)
        try:
            backend_codes = self.encoder(
                audio.to(self.device),
                sample_rate,
                n_acoustic_codebooks=len(self.acoustic_codebook_sizes),
            )
        except BaseException as exc:
            _trace_finish(trace, error=exc)
            raise
        _trace_finish(trace, extra={"encoder_output_shapes": _shape(backend_codes)})
        semantic = integer_tensor(backend_codes[0], "LongCat semantic codes")
        acoustic = integer_tensor(backend_codes[1], "LongCat acoustic codes")
        if semantic.dim() != 2:
            raise ValueError("LongCat semantic encoder output must have shape [batch, time].")
        if acoustic.dim() != 3:
            raise ValueError(
                "LongCat acoustic encoder output must have shape [batch, codebook, time]."
            )
        codes = torch.cat(
            (semantic[..., None], acoustic.transpose(1, 2).contiguous()),
            dim=-1,
        )
        return validate_frame_codes(codes, self.codebook_sizes)

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        values = validate_frame_codes(codes, self.codebook_sizes)
        return self._decoder()(
            values[..., 0].to(self.device),
            values[..., 1:].transpose(1, 2).contiguous().to(self.device),
        )

    def _decoder(self) -> Any:
        try:
            return self.decoders[self.decoder]
        except KeyError as exc:
            available = ", ".join(sorted(self.decoders))
            raise ValueError(
                f"Decoder {self.decoder!r} is not loaded. Available decoders: {available}."
            ) from exc



def _feature_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | torch.Size | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"LongCat {name} must have shape [batch, time, feature].")
    if shape is not None and value.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"LongCat {name} must align on batch and time.")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"LongCat {name} must be floating point tensors.")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"LongCat {name} must be a positive integer.")
    return value


def _embedding_dim(value: object, name: str) -> int:
    dimension = getattr(value, "embedding_dim", None)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise TypeError(f"LongCat {name} must expose a positive embedding_dim.")
    return dimension


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


def _trace_start(
    audio: Tensor,
    sample_rate: int,
    device: torch.device,
) -> tuple[Path, float, torch.device | None, dict[str, object]] | None:
    raw_path = os.environ.get("ANYTRAIN_LONGCAT_DEBUG_LOG")
    if not raw_path:
        return None
    cuda_device = device if device.type == "cuda" else None
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    fields: dict[str, object] = {
        "audio_batch": int(audio.size(0)),
        "audio_dtype": str(audio.dtype),
        "audio_numel": int(audio.numel()),
        "audio_sample_rate": sample_rate,
        "audio_shape": tuple(int(value) for value in audio.shape),
        "audio_seconds": float(audio.size(-1) * audio.size(0) / sample_rate),
        "device": str(device),
        "event": "longcat_encode_boundary",
    }
    return Path(raw_path).expanduser(), time.perf_counter(), cuda_device, fields


def _trace_finish(
    trace: tuple[Path, float, torch.device | None, dict[str, object]] | None,
    *,
    extra: dict[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    if trace is None:
        return
    path, started, cuda_device, fields = trace
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        fields.update(
            {
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(cuda_device)),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
                "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(cuda_device)),
            }
        )
    fields["elapsed_seconds"] = time.perf_counter() - started
    if error is None:
        fields["status"] = "complete"
        if extra:
            fields.update(extra)
    else:
        fields.update(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "error",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(fields, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _shape(value: object) -> object:
    if isinstance(value, Tensor):
        return tuple(int(size) for size in value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape(item) for item in value]
    return type(value).__name__
