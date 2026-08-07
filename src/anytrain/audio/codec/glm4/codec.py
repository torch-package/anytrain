from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from anytrain.audio.codec._audio import resample
from anytrain.audio.codec._module import DeviceModule, resolve_device, validate_frame_codes

from ._registry import (
    DEFAULT_GLM4_ARTIFACT,
    DEFAULT_GLM4_REVISION,
    GLM4_RUNTIME_GUIDANCE,
)

GLM4_TRANSFORMERS_VERSION = "4.44.1"
SAMPLE_RATE = 16_000
FRAME_RATE = 12.5
CODEBOOK_SIZE = 16_384
FEATURE_DIM = 1_280


class GLM4Tokenizer(DeviceModule):
    """Official GLM-4-Voice waveform tokenizer on its native 12.5 Hz grid.

    The upstream tokenizer is not a decoder. This wrapper deliberately exposes
    only ``encode`` and is therefore a ``FrameTokenizerBackend``, not a
    bidirectional ``FrameCodec``.
    """

    name = "glm4"
    sample_rate = SAMPLE_RATE
    frame_rate = FRAME_RATE
    codebook_sizes = (CODEBOOK_SIZE,)

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Any,
        *,
        device: torch.device,
        strict_contract: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("GLM-4 tokenizer model must be an nn.Module.")

        model.float()
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.feature_extractor = feature_extractor

        sampling_rate = _positive_integer_attribute(feature_extractor, "sampling_rate")
        hop_length = _positive_integer_attribute(feature_extractor, "hop_length")
        n_fft = _positive_integer_attribute(feature_extractor, "n_fft")
        convolution_stride = _stride(_module(model, "conv1")) * _stride(_module(model, "conv2"))
        config = _config(model)
        pooling = _positive_integer_attribute(config, "pooling_kernel_size")
        codebook = _codebook(model)
        if strict_contract:
            _require_contract(
                sampling_rate=sampling_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                convolution_stride=convolution_stride,
                pooling=pooling,
                codebook=codebook,
            )

        self._convolution_stride = convolution_stride
        self._pooling = pooling
        self._pad_multiple = convolution_stride * pooling * hop_length
        self._stft_guard_samples = n_fft // 2
        self._init_device(device)

    @property
    def semantic_codebook(self) -> Tensor:
        """Return the frozen VQ table used by GLM audio-token embeddings."""

        return _codebook(self.model).weight.detach()

    @property
    def semantic_feature_dim(self) -> int:
        return int(self.semantic_codebook.shape[1])

    @classmethod
    def from_pretrained(
        cls,
        *,
        source_root: str | Path,
        model_name_or_path: str | Path = DEFAULT_GLM4_ARTIFACT,
        revision: str = DEFAULT_GLM4_REVISION,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
    ) -> GLM4Tokenizer:
        if revision != DEFAULT_GLM4_REVISION:
            raise ValueError(
                _with_guidance(
                    "GLM-4 weights revision is fixed to "
                    f"{DEFAULT_GLM4_REVISION}; observed {revision!r}."
                )
            )
        root = _validate_source_root(source_root)
        model_cls, feature_extractor_cls = _load_runtime(root)
        kwargs: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())

        model = model_cls.from_pretrained(str(model_name_or_path), **kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError(
                _with_guidance("WhisperVQEncoder.from_pretrained() must return an nn.Module.")
            )
        feature_extractor = feature_extractor_cls.from_pretrained(
            str(model_name_or_path),
            **kwargs,
        )
        resolved_device = resolve_device(device)
        return cls(
            model,
            feature_extractor,
            device=resolved_device,
        )

    @torch.inference_mode()
    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        if not isinstance(audio, Tensor):
            raise TypeError("GLM-4 encode expects audio to be a Tensor.")
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("GLM-4 encode expects mono audio shape [batch, 1, time].")
        if audio.shape[0] == 0 or audio.shape[-1] == 0:
            raise ValueError("GLM-4 encode expects a non-empty waveform batch.")
        if not audio.is_floating_point() or audio.is_complex():
            raise TypeError("GLM-4 encode expects floating-point waveform samples.")

        waveform = audio.detach().float().cpu()
        waveform = resample(waveform, sample_rate, self.sample_rate)
        rows = [self._encode_one(waveform[index, 0]) for index in range(waveform.shape[0])]
        lengths = {int(row.numel()) for row in rows}
        if len(lengths) != 1:
            raise RuntimeError(
                "GLM-4 encoded batch rows have different frame lengths; pass an "
                "unpadded equal-length batch or tokenize samples separately."
            )
        codes = torch.stack(rows, dim=0).unsqueeze(-1).to(dtype=torch.long)
        return validate_frame_codes(codes, self.codebook_sizes)

    def _encode_one(self, waveform: Tensor) -> Tensor:
        guarded_samples = int(waveform.numel()) + self._stft_guard_samples
        padded_samples = (
            (guarded_samples + self._pad_multiple - 1) // self._pad_multiple
        ) * self._pad_multiple

        with _tf32_disabled():
            features = self.feature_extractor(
                [waveform.contiguous().numpy()],
                sampling_rate=self.sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
                padding="max_length",
                max_length=padded_samples,
                pad_to_multiple_of=self._pad_multiple,
                device="cpu",
            )
            feature_mapping = _feature_mapping(features)
            attention_mask = feature_mapping.get("attention_mask")
            if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
                raise RuntimeError("GLM-4 feature extractor must return a 2D attention_mask.")
            if attention_mask.shape[0] != 1:
                raise RuntimeError("GLM-4 canonical extraction expects one feature row.")
            model_inputs = {
                key: value.to(device=self.device)
                for key, value in feature_mapping.items()
                if isinstance(value, Tensor)
            }
            outputs = self.model(**model_inputs)

        ids = getattr(outputs, "quantized_token_ids", None)
        if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise RuntimeError("GLM-4 tokenizer must return [1, time] quantized_token_ids.")
        if ids.dtype is torch.bool or ids.is_floating_point() or ids.is_complex():
            raise TypeError("GLM-4 quantized_token_ids must contain integer ids.")

        token_mask = attention_mask[:, :: self._convolution_stride]
        token_mask = token_mask[:, :: self._pooling]
        if token_mask.shape != ids.shape:
            raise RuntimeError(
                "GLM-4 token mask shape differs from quantized IDs: "
                f"mask={tuple(token_mask.shape)}, ids={tuple(ids.shape)}."
            )
        selected = ids[0, token_mask[0].to(device=ids.device, dtype=torch.bool)]
        if selected.numel() == 0:
            raise RuntimeError("GLM-4 tokenizer produced an empty valid sequence.")
        return selected


def _validate_source_root(source_root: str | Path) -> Path:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            _with_guidance(f"GLM-4-Voice source_root does not exist: {source_root!s}.")
        ) from exc
    required = (
        "speech_tokenizer/configuration_whisper.py",
        "speech_tokenizer/generation_whisper.py",
        "speech_tokenizer/modeling_whisper.py",
    )
    missing = tuple(path for path in required if not root.joinpath(path).is_file())
    if missing:
        raise FileNotFoundError(
            _with_guidance(
                f"GLM-4-Voice source_root {root} is missing required files: {', '.join(missing)}."
            )
        )
    return root


def _load_runtime(source_root: Path) -> tuple[Any, Any]:
    try:
        transformers = import_module("transformers")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            _with_guidance("GLM-4 online tokenization cannot import Transformers.")
        ) from exc
    version = getattr(transformers, "__version__", None)
    if version != GLM4_TRANSFORMERS_VERSION:
        raise RuntimeError(
            _with_guidance(
                "GLM-4 online tokenization has only been validated with Transformers "
                f"{GLM4_TRANSFORMERS_VERSION}; observed {version!r}."
            )
        )

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        modeling = import_module("speech_tokenizer.modeling_whisper")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            _with_guidance(
                "Failed to import speech_tokenizer.modeling_whisper from source_root "
                f"{source_root}."
            )
        ) from exc
    module_path = _module_path(modeling)
    try:
        module_path.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(
            _with_guidance(
                "speech_tokenizer was imported from outside the configured "
                f"source_root {source_root}: {module_path}."
            )
        ) from exc

    model_cls = getattr(modeling, "WhisperVQEncoder", None)
    feature_extractor_cls = getattr(transformers, "WhisperFeatureExtractor", None)
    if (
        not isinstance(model_cls, type)
        or not issubclass(model_cls, nn.Module)
        or not callable(getattr(model_cls, "from_pretrained", None))
        or not callable(getattr(feature_extractor_cls, "from_pretrained", None))
    ):
        raise RuntimeError(
            _with_guidance(
                "The configured GLM-4 runtime must expose an nn.Module "
                "WhisperVQEncoder and WhisperFeatureExtractor, both with "
                "from_pretrained()."
            )
        )
    return model_cls, feature_extractor_cls


def _config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("GLM-4 tokenizer model must expose config.")
    return config


def _codebook(model: nn.Module) -> nn.Embedding:
    codebook = getattr(model, "codebook", None)
    if not isinstance(codebook, nn.Embedding):
        raise TypeError("GLM-4 tokenizer model must expose an embedding codebook.")
    return codebook


def _module(model: nn.Module, name: str) -> nn.Module:
    value = getattr(model, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"GLM-4 tokenizer model must expose module {name!r}.")
    return value


def _positive_integer_attribute(value: object, name: str) -> int:
    resolved = getattr(value, name, None)
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise TypeError(f"GLM-4 runtime must expose positive integer {name!r}.")
    return resolved


def _stride(convolution: nn.Module) -> int:
    stride = getattr(convolution, "stride", None)
    if not isinstance(stride, tuple) or len(stride) != 1 or int(stride[0]) <= 0:
        raise TypeError("GLM-4 convolution must expose one positive stride.")
    return int(stride[0])


def _require_contract(
    *,
    sampling_rate: int,
    hop_length: int,
    n_fft: int,
    convolution_stride: int,
    pooling: int,
    codebook: nn.Embedding,
) -> None:
    observed = {
        "sampling_rate": sampling_rate,
        "hop_length": hop_length,
        "n_fft": n_fft,
        "convolution_stride": convolution_stride,
        "pooling_kernel_size": pooling,
        "codebook_size": codebook.num_embeddings,
        "feature_dim": codebook.embedding_dim,
    }
    expected = {
        "sampling_rate": SAMPLE_RATE,
        "hop_length": 160,
        "n_fft": 400,
        "convolution_stride": 2,
        "pooling_kernel_size": 4,
        "codebook_size": CODEBOOK_SIZE,
        "feature_dim": FEATURE_DIM,
    }
    if observed != expected:
        raise RuntimeError(
            _with_guidance(
                "GLM-4 tokenizer model/feature-extractor/preprocess contract drift: "
                f"expected={expected}, observed={observed}."
            )
        )


def _feature_mapping(features: object) -> Mapping[str, Tensor]:
    if isinstance(features, Mapping):
        return cast(Mapping[str, Tensor], features)
    data = getattr(features, "data", None)
    if isinstance(data, Mapping):
        return cast(Mapping[str, Tensor], data)
    raise TypeError("GLM-4 feature extractor output must be a mapping of tensors.")


def _module_path(value: object) -> Path:
    resolved = getattr(value, "__file__", None)
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(_with_guidance("GLM-4 source module does not expose __file__."))
    try:
        return Path(resolved).resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            _with_guidance(f"GLM-4 source module path does not exist: {resolved}.")
        ) from exc


def _with_guidance(message: str) -> str:
    return f"{message} {GLM4_RUNTIME_GUIDANCE}"


@contextmanager
def _tf32_disabled() -> Iterator[None]:
    cudnn = torch.backends.cudnn
    matmul = torch.backends.cuda.matmul
    previous_cudnn = cudnn.allow_tf32
    previous_matmul = matmul.allow_tf32
    cudnn.allow_tf32 = False
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        cudnn.allow_tf32 = previous_cudnn
        matmul.allow_tf32 = previous_matmul


__all__ = [
    "CODEBOOK_SIZE",
    "FEATURE_DIM",
    "FRAME_RATE",
    "GLM4Tokenizer",
    "GLM4_TRANSFORMERS_VERSION",
    "SAMPLE_RATE",
]
