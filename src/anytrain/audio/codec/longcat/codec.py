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
    def acoustic_factor_codebooks(self) -> tuple[Tensor, ...]:
        """Return the two factor embedding tables for each acoustic codebook."""
        codebooks: list[Tensor] = []
        for index in range(len(self.acoustic_codebook_sizes)):
            quantizer = self._acoustic_quantizer(index)
            codebooks.extend(
                (
                    _embedding_weight(quantizer.codebook_a, f"acoustic factor {index}a"),
                    _embedding_weight(quantizer.codebook_b, f"acoustic factor {index}b"),
                )
            )
        return tuple(codebooks)

    @property
    def acoustic_factor_sizes(self) -> tuple[tuple[int, int], ...]:
        """Return the factor vocabulary sizes behind each composite acoustic codebook."""
        sizes: list[tuple[int, int]] = []
        for index in range(len(self.acoustic_codebook_sizes)):
            quantizer = self._acoustic_quantizer(index)
            size_a = _positive_int(quantizer.codebook_size_a, f"acoustic factor {index}a size")
            size_b = _positive_int(quantizer.codebook_size_b, f"acoustic factor {index}b size")
            if size_a * size_b != self.acoustic_codebook_sizes[index]:
                raise ValueError(
                    f"LongCat acoustic factor sizes for codebook {index} do not match its vocabulary."
                )
            sizes.append((size_a, size_b))
        return tuple(sizes)

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

        feature_dim = 0
        codebooks = self.acoustic_factor_codebooks
        for index in range(len(self.acoustic_codebook_sizes)):
            feature_dim += codebooks[index * 2].size(1) + codebooks[index * 2 + 1].size(1)
            if feature_dim == acoustic.size(-1):
                projected = self.project_acoustic_factor_features(
                    acoustic,
                    active_codebooks=index + 1,
                )
                return self.decode_feature_streams(semantic, (projected,))
            if feature_dim > acoustic.size(-1):
                break
        raise ValueError(
            "LongCat acoustic features must match the native feature dim or a complete "
            "prefix of factor codebook feature dims."
        )

    @torch.no_grad()
    def project_acoustic_factor_features(
        self,
        factor_features: Tensor,
        *,
        active_codebooks: int,
    ) -> Tensor:
        """Project concatenated factor embeddings into native decoder features."""
        count = _active_codebooks(active_codebooks, len(self.acoustic_codebook_sizes))
        features = _feature_tensor(factor_features, name="factor features")
        codebooks = self.acoustic_factor_codebooks[: count * 2]
        dimensions = tuple(codebook.size(1) for codebook in codebooks)
        if features.size(-1) != sum(dimensions):
            raise ValueError(
                "LongCat factor features must contain both embeddings for every active codebook."
            )
        reference = codebooks[0]
        factors = features.to(device=reference.device, dtype=reference.dtype).split(
            dimensions,
            dim=-1,
        )
        projected: Tensor | None = None
        for index in range(count):
            quantizer = self._acoustic_quantizer(index)
            value = torch.cat(
                (
                    quantizer.out_proj_a(factors[index * 2].transpose(1, 2).contiguous()),
                    quantizer.out_proj_b(factors[index * 2 + 1].transpose(1, 2).contiguous()),
                ),
                dim=1,
            )
            projected = value if projected is None else projected + value
        if projected is None:
            raise AssertionError("positive active_codebooks must project at least one codebook.")
        result = projected.transpose(1, 2).contiguous()
        result = _feature_tensor(
            result,
            name="projected factor features",
            shape=features.shape[:2],
        )
        if result.size(-1) != self.acoustic_feature_dim:
            raise ValueError("LongCat projected factor features must match acoustic_feature_dim.")
        return result

    @torch.no_grad()
    def quantize_acoustic_residual(self, residual: Tensor, *, codebook: int) -> Tensor:
        """Quantize one native residual into its two factor code IDs."""
        if not 0 <= codebook < len(self.acoustic_codebook_sizes):
            raise ValueError("LongCat residual codebook index is outside the configured range.")
        values = _feature_tensor(residual, name="acoustic residual")
        if values.size(-1) != self.acoustic_feature_dim:
            raise ValueError("LongCat acoustic residual must match acoustic_feature_dim.")
        quantizer = self._acoustic_quantizer(codebook)
        reference = self.acoustic_factor_codebooks[codebook * 2]
        prepared = values.to(device=reference.device, dtype=reference.dtype)
        output = quantizer(prepared.transpose(1, 2).contiguous())
        if not isinstance(output, tuple) or len(output) < 4 or not isinstance(output[3], Tensor):
            raise TypeError("LongCat acoustic quantizer must return code indices at position 3.")
        composite = integer_tensor(output[3], "LongCat acoustic residual codes")
        if composite.shape != values.shape[:2]:
            raise ValueError("LongCat acoustic residual codes must align on batch and time.")
        size_a, size_b = self.acoustic_factor_sizes[codebook]
        if bool((composite < 0).any()) or bool((composite >= size_a * size_b).any()):
            raise ValueError("LongCat acoustic residual codes contain an ID outside the codebook.")
        return torch.stack(
            (
                torch.div(composite, size_b, rounding_mode="floor"),
                composite.remainder(size_b),
            ),
            dim=-1,
        )

    @torch.no_grad()
    def native_acoustic_features(
        self,
        acoustic_codes: Tensor,
        *,
        active_codebooks: int,
    ) -> Tensor:
        """Convert a prefix of composite acoustic codes to native decoder latents."""
        count = _active_codebooks(active_codebooks, len(self.acoustic_codebook_sizes))
        values = integer_tensor(acoustic_codes, "LongCat acoustic codes")
        if values.dim() != 3 or values.size(-1) != count:
            raise ValueError(
                "LongCat acoustic codes must have shape [batch, time, active_codebooks]."
            )
        for index, size in enumerate(self.acoustic_codebook_sizes[:count]):
            codes = values[..., index]
            if bool((codes < 0).any()) or bool((codes >= size).any()):
                raise ValueError(f"LongCat acoustic codebook {index} contains an ID outside its range.")
        acoustic = getattr(self._decoder(), "acoustic_quantizer", None)
        convert = getattr(acoustic, "from_codes", None)
        if not callable(convert):
            raise TypeError("LongCat acoustic quantizer must provide from_codes().")
        output = convert(values.transpose(1, 2).contiguous().to(self.device))
        if not isinstance(output, tuple) or not output or not isinstance(output[0], Tensor):
            raise TypeError("LongCat acoustic quantizer from_codes() must return native features.")
        features = output[0]
        if features.dim() != 3:
            raise ValueError("LongCat native acoustic features must have shape [batch, dim, time].")
        return _feature_tensor(
            features.transpose(1, 2).contiguous(),
            name="native acoustic features",
            shape=values.shape[:2],
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
        return self._codes_from_backend(backend_codes)

    @torch.no_grad()
    def encode_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode one padded batch and return its valid frame lengths."""

        _validate_padded_inputs(audio, lengths)
        encoder_sample_rate = _encoder_sample_rate(self.encoder, self.sample_rate)
        if sample_rate != encoder_sample_rate:
            raise ValueError(
                "LongCat padded encoding requires audio at the encoder sample rate: "
                f"got {sample_rate}, expected {encoder_sample_rate}."
            )

        semantic_with_lengths = getattr(self.encoder, "get_semantic_codes_with_lengths", None)
        acoustic_with_lengths = getattr(self.encoder, "get_acoustic_codes_with_lengths", None)
        if callable(semantic_with_lengths) and callable(acoustic_with_lengths):
            device_audio = audio.to(self.device)
            device_lengths = lengths.to(device=self.device, dtype=torch.int32)
            semantic_result = semantic_with_lengths(
                device_audio,
                wav_lens=device_lengths,
            )
            acoustic_result = acoustic_with_lengths(
                device_audio,
                audio_lens=device_lengths,
                n_acoustic_codebooks=len(self.acoustic_codebook_sizes),
            )
            if (
                not isinstance(semantic_result, tuple)
                or len(semantic_result) != 2
                or not isinstance(acoustic_result, tuple)
                or len(acoustic_result) != 2
            ):
                raise TypeError(
                    "LongCat length-aware encoder methods must return codes and lengths."
                )
            semantic, semantic_lengths = semantic_result
            acoustic, acoustic_lengths = acoustic_result
            if not isinstance(semantic_lengths, Tensor) or not isinstance(acoustic_lengths, Tensor):
                raise TypeError("LongCat length-aware encoder methods must return Tensor lengths.")
            codes = self._codes_from_backend((semantic, acoustic))
            valid_lengths = torch.minimum(
                semantic_lengths.to(device=codes.device, dtype=torch.long),
                acoustic_lengths.to(device=codes.device, dtype=torch.long),
            )
        else:
            codes = self.encode(audio, sample_rate)
            valid_lengths = _fallback_frame_lengths(
                lengths.to(device=codes.device),
                sample_rate=sample_rate,
                encoder=self.encoder,
            )

        valid_lengths = valid_lengths.clamp(min=1, max=codes.size(1))
        return codes, valid_lengths

    def _codes_from_backend(self, backend_codes: object) -> Tensor:
        if not isinstance(backend_codes, (tuple, list)) or len(backend_codes) != 2:
            raise TypeError("LongCat encoder must return semantic and acoustic code tensors.")
        semantic = integer_tensor(backend_codes[0], "LongCat semantic codes")
        acoustic = integer_tensor(backend_codes[1], "LongCat acoustic codes")
        if semantic.dim() != 2:
            raise ValueError("LongCat semantic encoder output must have shape [batch, time].")
        if acoustic.dim() != 3:
            raise ValueError(
                "LongCat acoustic encoder output must have shape [batch, codebook, time]."
            )
        extent = min(semantic.size(1), acoustic.size(2))
        if extent <= 0:
            raise ValueError("LongCat encoder must return at least one frame.")
        if semantic.size(0) != acoustic.size(0):
            raise ValueError("LongCat semantic and acoustic codes must align on batch.")
        codes = torch.cat(
            (semantic[:, :extent, None], acoustic.transpose(1, 2)[:, :extent]),
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


def _embedding_weight(value: object, name: str) -> Tensor:
    weight = getattr(value, "weight", None)
    if not isinstance(weight, Tensor) or weight.dim() != 2:
        raise TypeError(f"LongCat {name} must expose a rank-2 weight Tensor.")
    return weight


def _active_codebooks(value: object, total: int) -> int:
    count = _positive_int(value, "active_codebooks")
    if count > total:
        raise ValueError("LongCat active_codebooks exceeds the configured acoustic codebooks.")
    return count


def _validate_padded_inputs(audio: Tensor, lengths: Tensor) -> None:
    if audio.dim() != 3 or audio.size(1) != NUM_CHANNELS:
        raise ValueError("LongCat padded encoding expects mono audio shape [batch, 1, time].")
    if not isinstance(lengths, Tensor) or lengths.dim() != 1:
        raise ValueError("LongCat padded waveform lengths must have shape [batch].")
    if lengths.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("LongCat padded waveform lengths must contain signed integers.")
    if lengths.numel() != audio.size(0):
        raise ValueError("LongCat padded waveform lengths must align with the batch.")
    if lengths.numel() == 0 or bool((lengths <= 0).any()):
        raise ValueError("LongCat padded waveform lengths must be positive.")
    if bool((lengths > audio.size(-1)).any()):
        raise ValueError("LongCat padded waveform lengths exceed the waveform extent.")


def _encoder_sample_rate(encoder: object, fallback: int) -> int:
    value = getattr(encoder, "input_sample_rate", fallback)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise TypeError("LongCat encoder must expose a positive input_sample_rate.")
    return int(value)


def _fallback_frame_lengths(lengths: Tensor, *, sample_rate: int, encoder: object) -> Tensor:
    input_sample_rate = _encoder_sample_rate(encoder, sample_rate)
    hop_length = getattr(encoder, "hop_length", None)
    if isinstance(hop_length, bool) or not isinstance(hop_length, Integral) or hop_length <= 0:
        raise TypeError("LongCat encoder must expose a positive hop_length.")
    numerator = lengths.to(dtype=torch.long) * input_sample_rate
    denominator = int(sample_rate) * int(hop_length)
    return torch.div(numerator, denominator, rounding_mode="floor")


def _load_longcat_loaders():
    try:
        from longcat_audio_codec import (  # pyright: ignore[reportMissingImports]
            load_decoder,
            load_encoder,
        )
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
