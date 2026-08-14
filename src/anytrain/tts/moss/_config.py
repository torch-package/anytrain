from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypedDict

from anytrain.tts import (
    TTSAcousticEngine,
    TTSEngineCapabilities,
    TTSInferenceConfig,
    TTSOptions,
    TTSTransformerEngine,
)
from anytrain.tts._common import (
    hashable_mapping,
    optional_non_empty_string,
    positive_int,
    validate_mapping,
)

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEFAULT_CODEC_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DEFAULT_SAMPLE_RATE = 24000
ENGINE_CAPABILITIES = TTSEngineCapabilities(
    transformer_engines=frozenset(
        (TTSTransformerEngine.TORCH, TTSTransformerEngine.VLLM_OMNI)
    ),
    acoustic_engines=frozenset((TTSAcousticEngine.TORCH,)),
)
_PLACEMENT_LOAD_KWARGS = frozenset(
    (
        "batch_timeout",
        "cache_dir",
        "deploy_config",
        "device_map",
        "devices",
        "gpu_memory_utilization",
        "init_timeout",
        "log_stats",
        "max_num_batched_tokens",
        "max_num_seqs",
        "stage_init_timeout",
        "local_files_only",
    )
)


class MossLoadKwargs(TypedDict, total=False):
    attn_implementation: object
    device_map: object
    dtype: object
    low_cpu_mem_usage: bool
    torch_dtype: object
    use_safetensors: bool


class MossRuntimeKwargs(TypedDict, total=False):
    do_sample: bool
    max_new_tokens: int
    temperature: float
    tokens: object
    top_p: float


@dataclass(frozen=True)
class MossTTSConfig:
    model: str = DEFAULT_MODEL
    codec_model: str = DEFAULT_CODEC_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    language: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    inference: TTSInferenceConfig = field(default_factory=TTSInferenceConfig)
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        optional_non_empty_string(self.model, "model")
        optional_non_empty_string(self.codec_model, "codec_model")
        positive_int(self.sample_rate, "sample_rate")
        optional_non_empty_string(self.language, "language")
        optional_non_empty_string(self.revision, "revision")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("trust_remote_code must be a bool.")
        if not isinstance(self.inference, TTSInferenceConfig):
            raise TypeError("inference must be a TTSInferenceConfig.")
        ENGINE_CAPABILITIES.validate(self.inference, "MOSS-TTS v1.5")
        validate_mapping(self.load_kwargs, "load_kwargs")
        validate_mapping(self.runtime_kwargs, "runtime_kwargs")

    def options(self) -> TTSOptions:
        return TTSOptions(
            language=self.language,
            sample_rate=self.sample_rate,
            extra=self.runtime_kwargs,
        )

    def hash_dict(self) -> dict[str, object]:
        semantic_load_kwargs = {
            key: value
            for key, value in self.load_kwargs.items()
            if key not in _PLACEMENT_LOAD_KWARGS
        }
        return {
            "model": self.model,
            "codec_model": self.codec_model,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "inference": self.inference.hash_dict(),
            "load_kwargs": hashable_mapping(semantic_load_kwargs),
            "runtime_kwargs": hashable_mapping(self.runtime_kwargs),
        }
