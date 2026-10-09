from __future__ import annotations

import os
from numbers import Integral
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn

from .._audio import resample
from .._module import DeviceModule, integer_tensor, resolve_device, validate_frame_codes
from . import _loading, _trace
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


class _LengthAwareEncoder(Protocol):
    input_sample_rate: int

    def preprocess(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def get_semantic_codes_with_lengths(
        self,
        audio: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def get_acoustic_codes_with_lengths(
        self,
        audio: Tensor,
        lengths: Tensor,
        *,
        n_acoustic_codebooks: int,
    ) -> tuple[Tensor, Tensor]: ...


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
            raise TypeError(
                "LongCat encoder must expose positive input_sample_rate and hop_length."
            )
        sample_rate = int(sample_rate)
        hop_length = int(hop_length)
        if sample_rate <= 0 or hop_length <= 0:
            raise TypeError(
                "LongCat encoder must expose positive input_sample_rate and hop_length."
            )
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
        load_encoder, load_decoder = _loading.load_loaders()
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

        with _loading.checkpoint_env(assets.ckpt_dir):
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
        if bool((values < 0).any()) or bool((values >= self.acoustic_codebook_sizes[index]).any()):
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
                raise ValueError(
                    f"LongCat acoustic codebook {index} contains an ID outside its range."
                )
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
        _validate_audio(audio)
        sample_rate = _sample_rate(sample_rate)
        trace = _trace.start(audio, sample_rate, self.device)
        try:
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
            codes = torch.cat(
                (semantic[..., None], acoustic.transpose(1, 2).contiguous()),
                dim=-1,
            )
            codes = validate_frame_codes(codes, self.codebook_sizes)
        except BaseException as exc:
            _trace.finish(trace, error=exc)
            raise
        _trace.finish(trace, extra={"encoder_output_shapes": _trace.shape(backend_codes)})
        return codes

    @torch.no_grad()
    def encode_padded(
        self,
        audio: Tensor,
        sample_rate: int,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode padded audio while retaining the usable frame count per sample."""

        _validate_audio(audio)
        sample_rate = _sample_rate(sample_rate)

        input_rate = getattr(self.encoder, "input_sample_rate", None)
        if isinstance(input_rate, bool) or not isinstance(input_rate, Integral):
            raise TypeError("LongCat encoder must expose a positive input_sample_rate.")
        input_rate = int(input_rate)
        if input_rate <= 0:
            raise TypeError("LongCat encoder must expose a positive input_sample_rate.")

        prepared_lengths = _waveform_lengths(
            lengths,
            batch_size=audio.size(0),
            maximum=audio.size(-1),
            device=audio.device,
        )
        if sample_rate != input_rate:
            audio, prepared_lengths = _resample_padded_audio(
                audio,
                prepared_lengths,
                sample_rate=sample_rate,
                target_rate=input_rate,
            )
            sample_rate = input_rate

        prepared_lengths = prepared_lengths.to(device=self.device)
        preprocess = getattr(self.encoder, "preprocess", None)
        semantic_call = getattr(self.encoder, "get_semantic_codes_with_lengths", None)
        acoustic_call = getattr(self.encoder, "get_acoustic_codes_with_lengths", None)
        if not callable(preprocess) or not callable(semantic_call) or not callable(acoustic_call):
            raise TypeError("LongCat encoder must provide length-aware encode methods.")
        encoder = cast(_LengthAwareEncoder, self.encoder)

        trace = _trace.start(audio, sample_rate, self.device)
        try:
            prepared = encoder.preprocess(audio.to(self.device), sample_rate)
            if not isinstance(prepared, Tensor):
                raise TypeError("LongCat encoder preprocess must return a Tensor.")
            semantic, semantic_lengths = encoder.get_semantic_codes_with_lengths(
                prepared,
                prepared_lengths,
            )
            acoustic, acoustic_lengths = encoder.get_acoustic_codes_with_lengths(
                prepared,
                prepared_lengths,
                n_acoustic_codebooks=len(self.acoustic_codebook_sizes),
            )
            # Rows shorter than one encoder frame emit zero valid code frames
            # (convolutions over the zero-masked tail still produce frames, so
            # the codes are deterministic silence encodings).  Encode such a
            # row as exactly one frame instead of failing the positive-length
            # validation below — the padded encode must not reject short
            # audio that the caller legitimately asked to tokenize.
            semantic_lengths = _positive_frame_lengths(semantic_lengths)
            acoustic_lengths = _positive_frame_lengths(acoustic_lengths)
            semantic = integer_tensor(semantic, "LongCat semantic codes").to(self.device)
            acoustic = integer_tensor(acoustic, "LongCat acoustic codes").to(self.device)
            if semantic.dim() != 2:
                raise ValueError("LongCat semantic encoder output must have shape [batch, time].")
            if semantic.size(0) != audio.size(0):
                raise ValueError("LongCat semantic encoder output must align on the audio batch.")
            if acoustic.dim() != 3:
                raise ValueError(
                    "LongCat acoustic encoder output must have shape [batch, codebook, time]."
                )
            if acoustic.size(0) != audio.size(0):
                raise ValueError("LongCat acoustic encoder output must align on the audio batch.")
            if acoustic.size(1) != len(self.acoustic_codebook_sizes):
                raise ValueError(
                    "LongCat acoustic encoder output must contain every configured codebook."
                )
            if semantic.size(1) != acoustic.size(2):
                raise ValueError("LongCat semantic and acoustic padded code extents must align.")

            codes = torch.cat(
                (semantic[..., None], acoustic.transpose(1, 2).contiguous()),
                dim=-1,
            )
            codes = validate_frame_codes(codes, self.codebook_sizes)
            semantic_valid_lengths = _code_lengths(
                semantic_lengths,
                batch_size=codes.size(0),
                maximum=codes.size(1),
                device=codes.device,
                name="semantic",
            )
            acoustic_valid_lengths = _code_lengths(
                acoustic_lengths,
                batch_size=codes.size(0),
                maximum=codes.size(1),
                device=codes.device,
                name="acoustic",
            )
            length_delta = semantic_valid_lengths - acoustic_valid_lengths
            if bool((length_delta < 0).any()) or bool((length_delta > 1).any()):
                raise ValueError(
                    "LongCat semantic valid lengths must equal acoustic lengths or exceed "
                    "them by one tail frame."
                )
            _trace.finish(
                trace,
                extra={
                    "encoder_output_shapes": _trace.shape((semantic, acoustic)),
                    "semantic_acoustic_length_delta": _trace.length_summary(length_delta),
                    "valid_code_lengths": _trace.length_summary(acoustic_valid_lengths),
                },
            )
            return codes, acoustic_valid_lengths
        except BaseException as error:
            _trace.finish(trace, error=error)
            raise

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


def _validate_audio(value: object) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("LongCat encode expects audio to be a Tensor.")
    if value.dim() != 3 or value.size(1) != NUM_CHANNELS:
        raise ValueError("LongCat encode expects mono audio shape [batch, 1, time].")
    if value.size(0) <= 0 or value.size(-1) <= 0:
        raise ValueError("LongCat audio must not be empty.")


def _sample_rate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("LongCat sample_rate must be an integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("LongCat sample_rate must be positive.")
    return resolved


def _resample_padded_audio(
    audio: Tensor,
    lengths: Tensor,
    *,
    sample_rate: int,
    target_rate: int,
) -> tuple[Tensor, Tensor]:
    """Resample each valid clip, then re-pad so length-aware encode stays aligned."""

    if sample_rate == target_rate:
        return audio, lengths

    rows: list[Tensor] = []
    resized: list[int] = []
    for index, length in enumerate(int(value) for value in lengths.tolist()):
        clip = audio[index, ..., :length].contiguous()
        resampled = resample(clip.unsqueeze(0), sample_rate, target_rate).squeeze(0)
        if resampled.dim() != 2 or resampled.size(0) != NUM_CHANNELS:
            raise ValueError(
                "LongCat resampled audio must keep mono shape [1, time]."
            )
        if resampled.size(-1) <= 0:
            raise ValueError("LongCat resampled audio must not be empty.")
        rows.append(resampled)
        resized.append(int(resampled.size(-1)))

    maximum = max(resized)
    padded = audio.new_zeros((audio.size(0), NUM_CHANNELS, maximum))
    for index, row in enumerate(rows):
        padded[index, ..., : row.size(-1)] = row
    return padded, torch.tensor(resized, dtype=torch.long, device=lengths.device)


def _waveform_lengths(
    value: Tensor,
    *,
    batch_size: int,
    maximum: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError("LongCat waveform lengths must have shape [batch].")
    if value.numel() != batch_size:
        raise ValueError("LongCat waveform lengths must align with the audio batch.")
    if value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("LongCat waveform lengths must contain signed integers.")
    lengths = value.to(device=device, dtype=torch.long)
    if lengths.numel() == 0 or bool((lengths <= 0).any()):
        raise ValueError("LongCat waveform lengths must be positive.")
    if bool((lengths > maximum).any()):
        raise ValueError("LongCat waveform lengths must be within the padded audio extent.")
    return lengths


def _code_lengths(
    value: Tensor,
    *,
    batch_size: int,
    maximum: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError(f"LongCat {name} code lengths must have shape [batch].")
    if value.numel() != batch_size:
        raise ValueError(f"LongCat {name} code lengths must align with the code batch.")
    if value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError(f"LongCat {name} code lengths must contain signed integers.")
    lengths = value.to(device=device, dtype=torch.long)
    if lengths.numel() == 0 or bool((lengths <= 0).any()):
        raise ValueError(f"LongCat {name} code lengths must be positive.")
    if bool((lengths > maximum).any()):
        raise ValueError(f"LongCat {name} code lengths exceed the padded code extent.")
    return lengths


def _positive_frame_lengths(value: Tensor) -> Tensor:
    """Clamp backend-reported code lengths to at least one frame.

    A waveform shorter than one encoder frame legitimately encodes to zero
    frames; the frames at padded positions are deterministic encodings of
    silence, so reporting one frame keeps the padded contract total instead
    of raising on sub-frame rows.  Non-tensor backend outputs pass through so
    ``_code_lengths`` still reports the shape/type problem.
    """

    if isinstance(value, Tensor):
        return value.clamp(min=1)
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"LongCat {name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"LongCat {name} must be a positive integer.")
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
