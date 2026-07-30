from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .._audio import resample
from .._module import DeviceModule, integer_tensor, resolve_device
from ..protocol import AcousticLayout, SemanticAcousticCodes
from .assets import BiCodecAssets, ensure_bicodec_assets

SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
FEATURE_HIDDEN_STATE_INDEXES = (11, 14, 16)
BICODEC_INSTALL_COMMAND = (
    'python -m pip install "spark-tts @ '
    "git+https://github.com/rookiejune/Spark-TTS.git@"
    'cf5302b0da797aeffc18af172de4dda0bb6ab718"'
)


class BiCodec(DeviceModule):
    name = "bicodec"
    num_channels: int = NUM_CHANNELS
    acoustic_layout = AcousticLayout.FIXED_LENGTH

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        feature_extractor: nn.Module,
        device: torch.device,
        *,
        model_dir: Path | None = None,
        assets: BiCodecAssets | None = None,
        config: dict[str, Any] | None = None,
        feature_hidden_state_indexes: tuple[int, ...] = FEATURE_HIDDEN_STATE_INDEXES,
    ) -> None:
        super().__init__()

        if not feature_hidden_state_indexes:
            raise ValueError("feature_hidden_state_indexes must not be empty.")

        self.model = model
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.model_dir = model_dir
        self.assets = assets
        self.config = {} if config is None else config
        self.feature_hidden_state_indexes = feature_hidden_state_indexes
        self.sample_rate = int(self.config.get("sample_rate", SAMPLE_RATE))
        self.ref_segment_length = _ref_segment_length(self.config)
        self.semantic_codebook_sizes = _codebook_sizes(getattr(model, "quantizer", None))
        speaker_encoder = getattr(model, "speaker_encoder", None)
        self.acoustic_codebook_sizes = _codebook_sizes(getattr(speaker_encoder, "quantizer", None))
        if not self.semantic_codebook_sizes:
            raise TypeError("BiCodec semantic quantizer must expose codebook sizes.")
        if not self.acoustic_codebook_sizes:
            raise TypeError("BiCodec speaker quantizer must expose codebook sizes.")
        self._acoustic_unit_length = _unit_length(speaker_encoder)
        self._acoustic_feature_dim = _feature_dim(speaker_encoder)
        if self.acoustic_layout is AcousticLayout.FIXED_LENGTH and self._acoustic_unit_length is None:
            raise TypeError("BiCodec speaker encoder must expose its fixed acoustic unit length.")

        extractor_config = getattr(self.feature_extractor, "config", None)
        if extractor_config is not None:
            extractor_config.output_hidden_states = True
        self._init_device(device)

    @property
    def semantic_codebook(self) -> Tensor:
        return _quantizer_codebook(self.model.quantizer, device=self.device)

    @property
    def acoustic_feature_dim(self) -> int:
        value = self._acoustic_feature_dim
        if value is None:
            raise TypeError("BiCodec speaker encoder must expose its acoustic feature dimension.")
        return value

    @property
    def acoustic_unit_length(self) -> int | None:
        return self._acoustic_unit_length

    @property
    def semantic_frame_rate(self) -> float:
        hop_length = self.config.get("latent_hop_length")
        if hop_length is None or int(hop_length) <= 0:
            raise TypeError("BiCodec config must expose a positive latent_hop_length.")
        return float(self.sample_rate / int(hop_length))

    @property
    def frame_rate(self) -> float:
        return self.semantic_frame_rate

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        model_dir: str | os.PathLike[str] | None = None,
        repo_id: str = "SparkAudio/Spark-TTS-0.5B",
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> BiCodec:
        model_cls = _load_bicodec_model()
        processor_cls, feature_extractor_cls = _load_wav2vec2_classes()
        assets: BiCodecAssets | None = None
        if model_dir is None:
            assets = ensure_bicodec_assets(
                cache_dir=cache_dir,
                repo_id=repo_id,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            root = assets["model_dir"]
        else:
            root = Path(model_dir).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"BiCodec model directory does not exist: {root}.")

        resolved_device = resolve_device(device)
        config = _load_config(root / "config.yaml")

        model = model_cls.load_from_checkpoint(root / "BiCodec").to(resolved_device)
        model.eval()
        processor = processor_cls.from_pretrained(str(root / "wav2vec2-large-xlsr-53"))
        feature_extractor = feature_extractor_cls.from_pretrained(
            str(root / "wav2vec2-large-xlsr-53")
        ).to(resolved_device)
        feature_extractor.eval()

        return cls(
            model=model,
            processor=processor,
            feature_extractor=feature_extractor,
            device=resolved_device,
            model_dir=root,
            assets=assets,
            config=config,
        )

    @torch.no_grad()
    def tokenize(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        ref_audio: Tensor | None = None,
        ref_sample_rate: int | None = None,
    ) -> SemanticAcousticCodes:
        wav = self._audio(audio, sample_rate)
        ref_wav = (
            self._audio(ref_audio, ref_sample_rate or sample_rate)
            if ref_audio is not None
            else wav
        )
        ref_wav = self._reference(ref_wav)
        features = self.extract_features(wav)

        tokenize = getattr(self.model, "tokenize", None)
        if not callable(tokenize):
            raise TypeError("BiCodec model must provide tokenize().")
        result = tokenize(
            {
                "wav": wav.to(self.device),
                "ref_wav": ref_wav.to(self.device),
                "feat": features.to(self.device),
            }
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("BiCodec tokenize() must return semantic and acoustic tensors.")
        semantic, backend_acoustic = result
        codes = SemanticAcousticCodes(
            semantic=_semantic_tokens(semantic).unsqueeze(-1),
            acoustic=_acoustic_tokens(backend_acoustic).transpose(1, 2).contiguous(),
        )
        self._validate_codes(codes)
        return codes

    @torch.no_grad()
    def extract_features(self, wav: Tensor) -> Tensor:
        if wav.dim() != 2:
            raise ValueError("wav must have shape [batch, time].")

        waveforms = [item.detach().cpu().numpy() for item in wav]
        inputs = self.processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        output = self.feature_extractor(inputs.to(self.device))
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise ValueError("BiCodec wav2vec2 feature extractor did not return hidden states.")
        indexes = self.feature_hidden_state_indexes
        if len(hidden_states) <= max(indexes):
            raise ValueError(
                "BiCodec wav2vec2 feature extractor returned too few hidden states: "
                f"{len(hidden_states)}."
            )

        features = hidden_states[indexes[0]]
        for index in indexes[1:]:
            features = features + hidden_states[index]
        features = features / len(indexes)
        if not isinstance(features, Tensor):
            raise TypeError("BiCodec wav2vec2 mixed features must be a Tensor.")
        if features.dim() != 3:
            raise ValueError("BiCodec wav2vec2 mixed features must have shape [batch, time, dim].")
        return features

    @torch.no_grad()
    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        self._validate_codes(codes)
        return self._detokenize(
            codes.semantic[..., 0],
            codes.acoustic.transpose(1, 2).contiguous(),
        )

    @torch.no_grad()
    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        """Convert global token ids to the per-slot speaker latent representation."""
        self._validate_acoustic_codes(acoustic_codes)
        quantizer = getattr(self.model.speaker_encoder, "quantizer", None)
        convert = getattr(quantizer, "get_output_from_indices", None)
        if not callable(convert):
            raise TypeError("BiCodec speaker quantizer must provide get_output_from_indices().")
        features = convert(acoustic_codes.to(self.device))
        if not isinstance(features, Tensor):
            raise TypeError("BiCodec acoustic feature conversion must return a Tensor.")
        if features.dim() != 3 or features.shape[:2] != acoustic_codes.shape[:2]:
            raise ValueError(
                "BiCodec acoustic features must have shape [batch, acoustic_unit, feature]."
            )
        if not torch.is_floating_point(features) or torch.is_complex(features):
            raise TypeError("BiCodec acoustic features must be floating point tensors.")
        return features

    @torch.no_grad()
    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        """Decode semantic ids and continuous global speaker features."""
        semantic = _semantic_codes(semantic_codes)
        if bool((semantic[..., 0] >= self.semantic_codebook_sizes[0]).any()):
            raise ValueError("semantic codes contain an ID outside the semantic codebook.")
        self._validate_acoustic_features(acoustic_features, batch_size=semantic.size(0))
        quantizer = getattr(self.model, "quantizer", None)
        detokenize = getattr(quantizer, "detokenize", None)
        if not callable(detokenize):
            raise TypeError("BiCodec semantic quantizer must provide detokenize().")
        z_q = detokenize(semantic[..., 0].to(self.device))
        if not isinstance(z_q, Tensor) or z_q.dim() != 3:
            raise ValueError("BiCodec semantic quantizer must return [batch, feature, time].")

        speaker_encoder = getattr(self.model, "speaker_encoder", None)
        project = getattr(speaker_encoder, "project", None)
        if not callable(project):
            raise TypeError("BiCodec speaker encoder must provide project().")
        d_vector = project(
            acoustic_features.to(self.device)
            .transpose(1, 2)
            .reshape(semantic.size(0), -1)
        )
        if not isinstance(d_vector, Tensor) or d_vector.dim() != 2:
            raise ValueError("BiCodec speaker project() must return [batch, speaker_dim].")
        prenet = getattr(self.model, "prenet", None)
        decoder = getattr(self.model, "decoder", None)
        if not callable(prenet) or not callable(decoder):
            raise TypeError("BiCodec model must provide prenet and decoder modules.")
        x = prenet(z_q, d_vector)
        if not isinstance(x, Tensor) or x.dim() != 3:
            raise ValueError("BiCodec prenet must return [batch, feature, time].")
        audio = decoder(x + d_vector.unsqueeze(-1))
        return _audio_tensor(audio)

    def _detokenize(self, semantic: Tensor, acoustic_tokens: Tensor) -> Tensor:
        detokenize = getattr(self.model, "detokenize", None)
        if not callable(detokenize):
            raise TypeError("BiCodec model must provide detokenize().")
        return _audio_tensor(
            detokenize(
                semantic.to(self.device),
                acoustic_tokens.to(self.device),
            )
        )

    def _audio(self, audio: Tensor, sample_rate: int) -> Tensor:
        if audio.dim() != 3:
            raise ValueError("BiCodec tokenize expects audio shape [batch, channels, time].")
        if audio.size(1) != NUM_CHANNELS:
            raise ValueError("BiCodec expects mono audio with shape [batch, 1, time].")
        if audio.size(-1) <= 0:
            raise ValueError("BiCodec audio must contain at least one sample.")
        return resample(audio, sample_rate, self.sample_rate).squeeze(1)

    def _reference(self, wav: Tensor) -> Tensor:
        length = self.ref_segment_length
        if length is None:
            return wav
        if wav.size(-1) <= 0:
            raise ValueError("BiCodec reference audio must contain at least one sample.")
        if wav.size(-1) < length:
            repeats = math.ceil(length / wav.size(-1))
            wav = wav.repeat(1, repeats)
        return wav[:, :length]

    def _validate_codes(self, codes: SemanticAcousticCodes) -> None:
        semantic = _semantic_codes(codes.semantic)
        acoustic = self._validate_acoustic_codes(codes.acoustic)
        semantic_size = self.semantic_codebook_sizes[0]
        if bool((semantic[..., 0] >= semantic_size).any()):
            raise ValueError("semantic codes contain an ID outside the semantic codebook.")
        if semantic.size(0) != acoustic.size(0):
            raise ValueError("semantic and acoustic codes must align on batch.")

    def _validate_acoustic_codes(self, acoustic_codes: Tensor) -> Tensor:
        acoustic = integer_tensor(acoustic_codes, "acoustic codes")
        if acoustic.dim() != 3:
            raise ValueError("acoustic codes must have shape [batch, slot, codebook].")
        if acoustic.size(-1) != len(self.acoustic_codebook_sizes):
            raise ValueError("acoustic codes must match acoustic_codebook_sizes.")
        if acoustic.size(1) <= 0:
            raise ValueError("acoustic codes must not be empty.")
        if self.acoustic_unit_length is not None and acoustic.size(1) != self.acoustic_unit_length:
            raise ValueError(
                "BiCodec acoustic codes must use the configured fixed acoustic unit length "
                f"{self.acoustic_unit_length}, got {acoustic.size(1)}."
            )
        if bool((acoustic < 0).any()):
            raise ValueError("acoustic codes must not contain negative IDs.")
        for index, size in enumerate(self.acoustic_codebook_sizes):
            if bool((acoustic[..., index] >= size).any()):
                raise ValueError(f"acoustic codes contain an ID outside codebook {index}.")
        return acoustic

    def _validate_acoustic_features(self, features: Tensor, *, batch_size: int) -> None:
        if features.dim() != 3:
            raise ValueError(
                "BiCodec acoustic features must have shape [batch, acoustic_unit, feature]."
            )
        if features.size(0) != batch_size:
            raise ValueError("BiCodec semantic and acoustic features must align on batch.")
        if self.acoustic_unit_length is not None and features.size(1) != self.acoustic_unit_length:
            raise ValueError(
                "BiCodec acoustic features must use the configured fixed acoustic unit length "
                f"{self.acoustic_unit_length}, got {features.size(1)}."
            )
        if features.size(-1) != self.acoustic_feature_dim:
            raise ValueError(
                "BiCodec acoustic features must match acoustic_feature_dim "
                f"{self.acoustic_feature_dim}, got {features.size(-1)}."
            )
        if not torch.is_floating_point(features) or torch.is_complex(features):
            raise TypeError("BiCodec acoustic features must be floating point tensors.")


def _ref_segment_length(config: dict[str, Any]) -> int | None:
    duration = config.get("ref_segment_duration")
    hop_length = config.get("latent_hop_length")
    sample_rate = int(config.get("sample_rate", SAMPLE_RATE))
    if duration is None or hop_length is None:
        return None
    length = int(sample_rate * float(duration))
    hop = int(hop_length)
    if hop <= 0:
        raise ValueError("latent_hop_length must be positive.")
    return max(hop, length // hop * hop)


def _semantic_codes(value: Any) -> Tensor:
    codes = integer_tensor(value, "semantic codes")
    if codes.dim() != 3:
        raise ValueError("semantic codes must have shape [batch, time, codebook].")
    if codes.size(-1) != 1:
        raise ValueError("BiCodec semantic codes must contain one codebook.")
    if codes.size(1) <= 0:
        raise ValueError("semantic codes must not be empty.")
    if bool((codes < 0).any()):
        raise ValueError("semantic codes must not contain negative IDs.")
    return codes


def _audio_tensor(value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("BiCodec decoder must return a Tensor.")
    if value.dim() == 2:
        value = value.unsqueeze(1)
    if value.dim() != 3:
        raise ValueError("BiCodec decoder must return audio shape [batch, channels, time].")
    return value


def _semantic_tokens(value: Any) -> Tensor:
    tokens = integer_tensor(value, "semantic tokens")
    if tokens.dim() != 2:
        raise ValueError("semantic tokens must have shape [batch, time].")
    return tokens.contiguous()


def _acoustic_tokens(value: Any) -> Tensor:
    tokens = integer_tensor(value, "acoustic tokens")
    if tokens.dim() != 3:
        raise ValueError("acoustic tokens must have shape [batch, codebook, slot].")
    return tokens.contiguous()


def _codebook_sizes(module: Any) -> tuple[int, ...]:
    if module is None:
        return ()
    value = getattr(module, "codebook_sizes", None)
    if value is not None:
        return tuple(int(size) for size in value)
    value = getattr(module, "codebook_size", None)
    if value is not None:
        count = int(getattr(module, "num_codebooks", getattr(module, "num_quantizers", 1)))
        return (int(value),) * count
    for name in ("num_embeddings", "n_tokens", "num_tokens", "vocab_size"):
        value = getattr(module, name, None)
        if value is not None:
            return (int(value),)
    return ()


def _unit_length(speaker_encoder: Any) -> int | None:
    sampler = getattr(speaker_encoder, "perceiver_sampler", None)
    latents = getattr(sampler, "latents", None)
    if isinstance(latents, Tensor) and latents.dim() == 2:
        return int(latents.size(0))
    for name in ("num_latents", "token_num", "num_tokens"):
        value = getattr(sampler, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _feature_dim(speaker_encoder: Any) -> int | None:
    quantizer = getattr(speaker_encoder, "quantizer", None)
    projection = getattr(quantizer, "project_out", None)
    for name in ("out_features", "out_channels"):
        value = getattr(projection, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    value = getattr(quantizer, "codebook_dim", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


@torch.no_grad()
def _quantizer_codebook(quantizer: Any, *, device: torch.device) -> Tensor:
    size = getattr(quantizer, "codebook_size", None)
    detokenize = getattr(quantizer, "detokenize", None)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or not callable(detokenize):
        raise TypeError("BiCodec semantic quantizer must expose codebook_size and detokenize().")
    ids = torch.arange(size, device=device, dtype=torch.long)[None]
    values = detokenize(ids)
    if not isinstance(values, Tensor) or values.dim() != 3 or values.size(0) != 1:
        raise ValueError("BiCodec semantic quantizer detokenize() must return [batch, dim, time].")
    if not torch.is_floating_point(values) or torch.is_complex(values):
        raise TypeError("BiCodec semantic codebook must be floating point.")
    return values[0].transpose(0, 1).contiguous()


def _load_bicodec_model() -> Any:
    try:
        from sparktts.models.bicodec import BiCodec as UpstreamBiCodec
    except ImportError as exc:
        raise ImportError(
            "BiCodec requires the Spark-TTS source package and its runtime dependencies. "
            "Run this command in the shell that will launch Python:\n"
            f"{BICODEC_INSTALL_COMMAND}"
        ) from exc
    return UpstreamBiCodec


def _load_config(path: Path) -> dict[str, Any]:
    try:
        from sparktts.utils.file import load_config
    except ImportError as exc:
        raise ImportError(
            "BiCodec config loading requires the Spark-TTS source package. "
            "Run this command in the shell that will launch Python:\n"
            f"{BICODEC_INSTALL_COMMAND}"
        ) from exc
    config = load_config(str(path))
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        DictConfig = None
        OmegaConf = None
    if DictConfig is not None and isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"BiCodec config must be a dict: {path}.")
    return config


def _load_wav2vec2_classes() -> tuple[Any, Any]:
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    except ImportError as exc:
        raise ImportError(
            'BiCodec feature extraction requires transformers. Install it with `python -m pip install "transformers>=4.46"`.'
        ) from exc
    return Wav2Vec2FeatureExtractor, Wav2Vec2Model


__all__ = [
    "BiCodec",
    "FEATURE_HIDDEN_STATE_INDEXES",
    "NUM_CHANNELS",
    "SAMPLE_RATE",
]
