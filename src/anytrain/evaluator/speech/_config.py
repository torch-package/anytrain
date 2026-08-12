from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from anytrain._semantic import semantic_id, semantic_options

from ..text import TextNormalizationConfig

if TYPE_CHECKING:
    from .asr import WhisperASREvaluator
    from .evaluator import SpeechEvaluator
    from .utmos import UTMOSEvaluator

DEFAULT_WHISPER_MODEL = "large-v3"
OPENAI_WHISPER_VERSION = "20250625"
WHISPER_DECODER_PROTOCOL = "anytrain-openai-whisper-decoder-v1"
WHISPER_METRIC_PROTOCOL = "anytrain-speech-text-metrics-v1"

WHISPER_MODEL_SHA256 = {
    "tiny.en": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
    "tiny": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "base.en": "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
    "base": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    "small.en": "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872",
    "small": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "medium.en": "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
    "medium": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    "large-v1": "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
    "large-v2": "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    "large-v3": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    "large": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    "large-v3-turbo": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
    "turbo": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
}
SUPPORTED_WHISPER_MODELS = tuple(WHISPER_MODEL_SHA256)

DEFAULT_UTMOS_REPOSITORY = "tarepan/SpeechMOS"
DEFAULT_UTMOS_REVISION = "ed25eacbfa42b99156c36ebec67a733b5dbb9b79"
DEFAULT_UTMOS_MODEL = "utmos22_strong"
DEFAULT_UTMOS_WEIGHTS = "utmos22_strong_step7459_v1.pt"
DEFAULT_UTMOS_WEIGHTS_SHA256 = (
    "38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff"
)
UTMOS_PROTOCOL = "anytrain-torchhub-utmos-v1"
SPEECH_EVALUATOR_PROTOCOL = "anytrain-speech-evaluator-v1"


@dataclass(frozen=True)
class WhisperSemanticConfig:
    model_name: str = DEFAULT_WHISPER_MODEL
    decode_options: tuple[tuple[str, object], ...] = ()
    normalization: TextNormalizationConfig = field(
        default_factory=TextNormalizationConfig
    )
    bleu_smoothing: bool = True
    backend_version: str = OPENAI_WHISPER_VERSION
    decoder_protocol: str = WHISPER_DECODER_PROTOCOL
    metric_protocol: str = WHISPER_METRIC_PROTOCOL

    def __post_init__(self) -> None:
        if self.model_name not in WHISPER_MODEL_SHA256:
            raise ValueError(
                "model_name must be one of "
                f"{SUPPORTED_WHISPER_MODELS!r}; got {self.model_name!r}."
            )
        semantic_options(self.decode_options, name="decode_options")
        if not isinstance(self.normalization, TextNormalizationConfig):
            raise TypeError("normalization must be a TextNormalizationConfig.")
        _normalization(self.normalization)
        if not isinstance(self.bleu_smoothing, bool):
            raise TypeError("bleu_smoothing must be a boolean.")
        if self.backend_version != OPENAI_WHISPER_VERSION:
            raise ValueError(
                f"unsupported openai-whisper version {self.backend_version!r}."
            )
        if self.decoder_protocol != WHISPER_DECODER_PROTOCOL:
            raise ValueError(f"unsupported Whisper decoder {self.decoder_protocol!r}.")
        if self.metric_protocol != WHISPER_METRIC_PROTOCOL:
            raise ValueError(f"unsupported speech metric protocol {self.metric_protocol!r}.")

    @property
    def artifact_sha256(self) -> str:
        return WHISPER_MODEL_SHA256[self.model_name]

    def semantic_payload(self) -> dict[str, object]:
        normalization = self.normalization
        return {
            "artifact_sha256": self.artifact_sha256,
            "backend": {
                "package": "openai-whisper",
                "version": self.backend_version,
            },
            "bleu_smoothing": self.bleu_smoothing,
            "decode_options": semantic_options(
                self.decode_options,
                name="decode_options",
            ),
            "decoder_protocol": self.decoder_protocol,
            "metric_protocol": self.metric_protocol,
            "model_name": self.model_name,
            "normalization": {
                "chinese": normalization.chinese,
                "collapse_whitespace": normalization.collapse_whitespace,
                "lowercase": normalization.lowercase,
                "remove_punctuation": normalization.remove_punctuation,
                "strip": normalization.strip,
            },
        }

    def semantic_id(self) -> str:
        return semantic_id("whisper-evaluator-v1", self.semantic_payload())

    def evaluator(
        self,
        load_options: WhisperLoadOptions | None = None,
    ) -> WhisperASREvaluator:
        from ..text import TextComparisonEvaluator
        from .asr import WhisperASREvaluator

        load = WhisperLoadOptions() if load_options is None else load_options
        if not isinstance(load, WhisperLoadOptions):
            raise TypeError("load_options must be WhisperLoadOptions.")
        normalization = self.normalization
        text_evaluator = TextComparisonEvaluator(
            strip=normalization.strip,
            collapse_whitespace=normalization.collapse_whitespace,
            remove_punctuation=normalization.remove_punctuation,
            lowercase=normalization.lowercase,
            chinese=normalization.chinese,
            bleu_smoothing=self.bleu_smoothing,
        )
        backend_load_options = {"in_memory": load.in_memory}
        return WhisperASREvaluator(
            text_evaluator=text_evaluator,
            decode_options=semantic_options(
                self.decode_options,
                name="decode_options",
            ),
            model_name=self.model_name,
            device=load.device,
            download_root=load.download_root,
            load_options=backend_load_options,
            local_files_only=load.local_files_only,
            backend_version=self.backend_version,
        )


@dataclass(frozen=True)
class WhisperLoadOptions:
    device: str | None = None
    download_root: str | Path | None = None
    local_files_only: bool = False
    in_memory: bool = False

    def __post_init__(self) -> None:
        if self.device is not None:
            _non_empty(self.device, name="device")
        if self.download_root is not None and not isinstance(
            self.download_root, (str, Path)
        ):
            raise TypeError("download_root must be a string, Path, or None.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be a boolean.")
        if not isinstance(self.in_memory, bool):
            raise TypeError("in_memory must be a boolean.")


@dataclass(frozen=True)
class UTMOSSemanticConfig:
    repository: str = DEFAULT_UTMOS_REPOSITORY
    revision: str = DEFAULT_UTMOS_REVISION
    model_name: str = DEFAULT_UTMOS_MODEL
    weights: str = DEFAULT_UTMOS_WEIGHTS
    weights_sha256: str = DEFAULT_UTMOS_WEIGHTS_SHA256
    protocol: str = UTMOS_PROTOCOL

    def __post_init__(self) -> None:
        _non_empty(self.repository, name="repository")
        if Path(self.repository).is_absolute():
            raise ValueError("repository must be a logical id, not an absolute path.")
        _commit(self.revision, name="revision")
        _non_empty(self.model_name, name="model_name")
        _non_empty(self.weights, name="weights")
        if Path(self.weights).name != self.weights:
            raise ValueError("weights must be a checkpoint filename, not a path.")
        _sha256(self.weights_sha256, name="weights_sha256")
        if self.protocol != UTMOS_PROTOCOL:
            raise ValueError(f"unsupported UTMOS protocol {self.protocol!r}.")

    @property
    def repo(self) -> str:
        return f"{self.repository}:{self.revision}"

    def semantic_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "protocol": self.protocol,
            "repository": self.repository,
            "revision": self.revision,
            "weights": self.weights,
            "weights_sha256": self.weights_sha256,
        }

    def semantic_id(self) -> str:
        return semantic_id("utmos-evaluator-v1", self.semantic_payload())

    def evaluator(
        self,
        load_options: UTMOSLoadOptions | None = None,
    ) -> UTMOSEvaluator:
        from .utmos import UTMOSEvaluator

        load = UTMOSLoadOptions() if load_options is None else load_options
        if not isinstance(load, UTMOSLoadOptions):
            raise TypeError("load_options must be UTMOSLoadOptions.")
        return UTMOSEvaluator(
            repo=self.repo,
            model_name=self.model_name,
            weights=self.weights,
            weights_sha256=self.weights_sha256,
            device=load.device,
            backend_load_options=semantic_options(
                load.backend_options,
                name="backend_options",
            ),
            allow_remote_code=load.allow_remote_code,
            local_files_only=load.local_files_only,
        )


@dataclass(frozen=True)
class UTMOSLoadOptions:
    device: str | None = None
    allow_remote_code: bool = False
    local_files_only: bool = False
    backend_options: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.device is not None:
            _non_empty(self.device, name="device")
        if not isinstance(self.allow_remote_code, bool):
            raise TypeError("allow_remote_code must be a boolean.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be a boolean.")
        options = semantic_options(self.backend_options, name="backend_options")
        unsupported = sorted(
            set(options) - {"force_reload", "skip_validation", "trust_repo", "verbose"}
        )
        if unsupported:
            raise ValueError(
                "backend_options contains semantic or unsupported options: "
                + ", ".join(unsupported)
            )


@dataclass(frozen=True)
class SpeechEvaluatorConfig:
    whisper: WhisperSemanticConfig = field(default_factory=WhisperSemanticConfig)
    utmos: UTMOSSemanticConfig = field(default_factory=UTMOSSemanticConfig)
    protocol: str = SPEECH_EVALUATOR_PROTOCOL

    def __post_init__(self) -> None:
        if not isinstance(self.whisper, WhisperSemanticConfig):
            raise TypeError("whisper must be WhisperSemanticConfig.")
        if not isinstance(self.utmos, UTMOSSemanticConfig):
            raise TypeError("utmos must be UTMOSSemanticConfig.")
        if self.protocol != SPEECH_EVALUATOR_PROTOCOL:
            raise ValueError(f"unsupported speech evaluator protocol {self.protocol!r}.")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "utmos": self.utmos.semantic_payload(),
            "whisper": self.whisper.semantic_payload(),
        }

    def semantic_id(self) -> str:
        return semantic_id("speech-evaluator-v1", self.semantic_payload())

    def evaluator(
        self,
        load_options: SpeechEvaluatorLoadOptions | None = None,
    ) -> SpeechEvaluator:
        from .evaluator import SpeechEvaluator

        load = SpeechEvaluatorLoadOptions() if load_options is None else load_options
        if not isinstance(load, SpeechEvaluatorLoadOptions):
            raise TypeError("load_options must be SpeechEvaluatorLoadOptions.")
        return SpeechEvaluator(
            asr=self.whisper.evaluator(load.whisper),
            utmos=self.utmos.evaluator(load.utmos),
        )


@dataclass(frozen=True)
class SpeechEvaluatorLoadOptions:
    whisper: WhisperLoadOptions = field(default_factory=WhisperLoadOptions)
    utmos: UTMOSLoadOptions = field(default_factory=UTMOSLoadOptions)

    def __post_init__(self) -> None:
        if not isinstance(self.whisper, WhisperLoadOptions):
            raise TypeError("whisper must be WhisperLoadOptions.")
        if not isinstance(self.utmos, UTMOSLoadOptions):
            raise TypeError("utmos must be UTMOSLoadOptions.")


def _normalization(config: TextNormalizationConfig) -> None:
    for name in ("strip", "collapse_whitespace", "remove_punctuation", "lowercase"):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"normalization {name} must be a boolean.")
    if config.chinese not in {None, "simplified"}:
        raise ValueError("normalization chinese must be 'simplified' or None.")


def _commit(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase 40-character commit hash.")


def _sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _non_empty(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


__all__ = [
    "DEFAULT_UTMOS_MODEL",
    "DEFAULT_UTMOS_REPOSITORY",
    "DEFAULT_UTMOS_REVISION",
    "DEFAULT_UTMOS_WEIGHTS",
    "DEFAULT_UTMOS_WEIGHTS_SHA256",
    "DEFAULT_WHISPER_MODEL",
    "OPENAI_WHISPER_VERSION",
    "SUPPORTED_WHISPER_MODELS",
    "SpeechEvaluatorConfig",
    "SpeechEvaluatorLoadOptions",
    "UTMOSLoadOptions",
    "UTMOSSemanticConfig",
    "WHISPER_MODEL_SHA256",
    "WhisperLoadOptions",
    "WhisperSemanticConfig",
]
