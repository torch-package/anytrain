from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Protocol

import torch.distributed as dist

from anytrain._semantic import semantic_id

from ..abc import EvaluatorABC
from ..text import TextNormalizationConfig
from ..text.normalization import normalize_text_batch
from ..types import MetricDict

COMET_PROTOCOL = "anytrain-reference-comet-v1"
COMET_DISTRIBUTION = "unbabel-comet"


def _default_normalization() -> TextNormalizationConfig:
    return TextNormalizationConfig(remove_punctuation=False, chinese=None)


@dataclass(frozen=True)
class CometSample:
    source: str
    hypothesis: str
    reference: str

    def __post_init__(self) -> None:
        _text(self.source, name="source")
        _text(self.hypothesis, name="hypothesis")
        _text(self.reference, name="reference")


@dataclass(frozen=True)
class CometResult:
    scores: tuple[float, ...]
    system_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.scores, tuple):
            raise TypeError("scores must be a tuple of floats.")
        if not self.scores:
            raise ValueError("scores must contain at least one segment score.")
        for index, score in enumerate(self.scores):
            _float(score, name=f"scores[{index}]")
        _float(self.system_score, name="system_score")


class CometBackend(Protocol):
    def predict(
        self,
        samples: Sequence[CometSample],
        *,
        batch_size: int,
    ) -> CometResult: ...


@dataclass(frozen=True)
class CometSemanticConfig:
    model: str
    revision: str
    weights_sha256: str
    backend_version: str
    batch_size: int = 8
    normalization: TextNormalizationConfig = field(default_factory=_default_normalization)
    protocol: str = COMET_PROTOCOL

    def __post_init__(self) -> None:
        _logical_id(self.model, name="model")
        _commit(self.revision, name="revision")
        _sha256(self.weights_sha256, name="weights_sha256")
        _text(self.backend_version, name="backend_version")
        _positive_int(self.batch_size, name="batch_size")
        if not isinstance(self.normalization, TextNormalizationConfig):
            raise TypeError("normalization must be a TextNormalizationConfig.")
        if self.normalization.chinese not in {None, "simplified"}:
            raise ValueError("normalization.chinese must be 'simplified' or None.")
        if self.protocol != COMET_PROTOCOL:
            raise ValueError(f"unsupported COMET protocol {self.protocol!r}.")

    def semantic_payload(self) -> dict[str, object]:
        normalization = self.normalization
        return {
            "backend": {
                "distribution": COMET_DISTRIBUTION,
                "version": self.backend_version,
            },
            "batch_size": self.batch_size,
            "model": self.model,
            "normalization": {
                "chinese": normalization.chinese,
                "collapse_whitespace": normalization.collapse_whitespace,
                "lowercase": normalization.lowercase,
                "remove_punctuation": normalization.remove_punctuation,
                "strip": normalization.strip,
            },
            "protocol": self.protocol,
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
        }

    def semantic_id(self) -> str:
        return semantic_id("comet-evaluator-v1", self.semantic_payload())

    def evaluator(
        self,
        load_options: CometLoadOptions | None = None,
        *,
        backend: CometBackend | None = None,
    ) -> CometEvaluator:
        return CometEvaluator(self, load_options=load_options, backend=backend)


@dataclass(frozen=True)
class CometLoadOptions:
    checkpoint: str | Path
    gpus: int = 0
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, (str, Path)):
            raise TypeError("checkpoint must be a string or Path.")
        if not str(self.checkpoint):
            raise ValueError("checkpoint must not be empty.")
        _non_negative_int(self.gpus, name="gpus")
        _non_negative_int(self.num_workers, name="num_workers")


class CometEvaluator(EvaluatorABC):
    """Reference-based COMET with aligned segment and corpus scores."""

    def __init__(
        self,
        config: CometSemanticConfig,
        *,
        load_options: CometLoadOptions | None = None,
        backend: CometBackend | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, CometSemanticConfig):
            raise TypeError("config must be a CometSemanticConfig.")
        if backend is None:
            if load_options is None:
                raise ValueError("load_options are required when backend is not provided.")
            if not isinstance(load_options, CometLoadOptions):
                raise TypeError("load_options must be CometLoadOptions.")
            backend = _UnbabelCometBackend(config, load_options)
        elif load_options is not None:
            raise ValueError("load_options cannot be combined with a custom backend.")
        if not callable(getattr(backend, "predict", None)):
            raise TypeError("backend must provide predict(samples, *, batch_size).")
        self._config = config
        self._backend = backend
        self._sources: list[str] = []
        self._hypotheses: list[str] = []
        self._references: list[str] = []

    @property
    def semantic_config(self) -> CometSemanticConfig:
        return self._config

    def semantic_id(self) -> str:
        return self.semantic_config.semantic_id()

    @property
    def backend(self) -> CometBackend:
        return self._backend

    def score(
        self,
        source_text: str | Sequence[str],
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> CometResult:
        sources, hypotheses, references = self._normalize(
            source_text,
            prediction_text,
            target_text,
        )
        samples = tuple(
            CometSample(source, hypothesis, reference)
            for source, hypothesis, reference in zip(
                sources,
                hypotheses,
                references,
                strict=True,
            )
        )
        result = self._backend.predict(samples, batch_size=self._config.batch_size)
        if not isinstance(result, CometResult):
            raise TypeError("COMET backend must return a CometResult.")
        if len(result.scores) != len(samples):
            raise ValueError(
                "COMET segment score count must match the input batch length: "
                f"got {len(result.scores)} and {len(samples)}."
            )
        return result

    def evaluate(
        self,
        source_text: str | Sequence[str],
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> MetricDict:
        return {"comet": self.score(source_text, prediction_text, target_text).system_score}

    def update(
        self,
        source_text: str | Sequence[str],
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> None:
        sources, hypotheses, references = self._normalize(
            source_text,
            prediction_text,
            target_text,
        )
        self._sources.extend(sources)
        self._hypotheses.extend(hypotheses)
        self._references.extend(references)

    def compute(self) -> MetricDict:
        sources, hypotheses, references = self._corpus()
        if not sources:
            raise ValueError("No translation triples have been recorded.")
        return {"comet": self.score(sources, hypotheses, references).system_score}

    def reset(self) -> None:
        self._sources.clear()
        self._hypotheses.clear()
        self._references.clear()

    def _normalize(
        self,
        source_text: str | Sequence[str],
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        normalization = self._config.normalization
        sources = normalize_text_batch(source_text, name="source_text", config=normalization)
        hypotheses = normalize_text_batch(
            prediction_text,
            name="prediction_text",
            config=normalization,
        )
        references = normalize_text_batch(target_text, name="target_text", config=normalization)
        lengths = (len(sources), len(hypotheses), len(references))
        if len(set(lengths)) != 1:
            raise ValueError(
                "source_text, prediction_text, and target_text must have the same batch "
                f"length: got {lengths[0]}, {lengths[1]}, and {lengths[2]}."
            )
        if not sources:
            raise ValueError("translation text inputs must contain at least one item.")
        for name, values in (
            ("source_text", sources),
            ("prediction_text", hypotheses),
            ("target_text", references),
        ):
            for index, value in enumerate(values):
                if not value:
                    raise ValueError(f"{name}[{index}] must not be empty after normalization.")
        return sources, hypotheses, references

    def _corpus(self) -> tuple[list[str], list[str], list[str]]:
        if not dist.is_available() or not dist.is_initialized():
            return (
                list(self._sources),
                list(self._hypotheses),
                list(self._references),
            )

        local = (
            tuple(self._sources),
            tuple(self._hypotheses),
            tuple(self._references),
        )
        gathered: list[
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None
        ] = [None] * dist.get_world_size()
        from anytrain.lightning.audit import audit_collective

        audit_collective(
            "torch.distributed.all_gather_object",
            lambda: dist.all_gather_object(gathered, local),
            source="CometEvaluator._corpus",
            object_type=type(local).__name__,
            local_item_count=len(local[0]),
            world_size=dist.get_world_size(),
        )

        sources: list[str] = []
        hypotheses: list[str] = []
        references: list[str] = []
        for corpus in gathered:
            if corpus is None:
                raise RuntimeError(
                    "Distributed COMET corpus gathering returned no value for a rank."
                )
            rank_sources, rank_hypotheses, rank_references = corpus
            sources.extend(rank_sources)
            hypotheses.extend(rank_hypotheses)
            references.extend(rank_references)
        return sources, hypotheses, references


class _UnbabelCometBackend:
    def __init__(self, config: CometSemanticConfig, load: CometLoadOptions) -> None:
        self.config = config
        self.checkpoint = Path(load.checkpoint)
        self.gpus = load.gpus
        self.num_workers = load.num_workers
        self.model: Any | None = None

    def predict(
        self,
        samples: Sequence[CometSample],
        *,
        batch_size: int,
    ) -> CometResult:
        model = self._load()
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise TypeError("loaded COMET model must provide predict(...).")
        output = predict(
            [
                {"src": sample.source, "mt": sample.hypothesis, "ref": sample.reference}
                for sample in samples
            ],
            batch_size=batch_size,
            gpus=self.gpus,
            num_workers=self.num_workers,
            progress_bar=False,
        )
        raw_scores: Any = getattr(output, "scores", None)
        try:
            values = tuple(raw_scores)
        except TypeError as error:
            raise TypeError("COMET predict output must expose iterable scores.") from error
        scores = tuple(_score(value, name=f"scores[{index}]") for index, value in enumerate(values))
        system_score = _score(
            getattr(output, "system_score", None),
            name="system_score",
        )
        return CometResult(scores, system_score)

    def _load(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"COMET checkpoint does not exist: {self.checkpoint}")
        actual_sha256 = _file_sha256(self.checkpoint)
        if actual_sha256 != self.config.weights_sha256:
            raise RuntimeError(
                "COMET checkpoint SHA-256 mismatch: "
                f"expected {self.config.weights_sha256}, got {actual_sha256}."
            )
        try:
            actual_version = distribution_version(COMET_DISTRIBUTION)
        except PackageNotFoundError as error:
            raise RuntimeError(
                'COMET evaluation requires anytrain[translation-eval].'
            ) from error
        if actual_version != self.config.backend_version:
            raise RuntimeError(
                f"COMET backend version mismatch: expected {self.config.backend_version!r}, "
                f"got {actual_version!r}."
            )
        module = import_module("comet")
        load_from_checkpoint = getattr(module, "load_from_checkpoint", None)
        if not callable(load_from_checkpoint):
            raise RuntimeError("unbabel-comet does not expose load_from_checkpoint(...).")
        self.model = load_from_checkpoint(str(self.checkpoint))
        return self.model


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"COMET {name} must be numeric, not boolean.")
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"COMET {name} must be numeric.") from error
    if not math.isfinite(score):
        raise ValueError(f"COMET {name} must be finite.")
    return score


def _float(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{name} must be a float.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _text(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _logical_id(value: object, *, name: str) -> None:
    _text(value, name=name)
    assert isinstance(value, str)
    if Path(value).is_absolute():
        raise ValueError(f"{name} must be a logical model id, not an absolute path.")


def _commit(value: object, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a 40-character lowercase commit hash.")


def _sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "COMET_PROTOCOL",
    "CometBackend",
    "CometEvaluator",
    "CometLoadOptions",
    "CometResult",
    "CometSample",
    "CometSemanticConfig",
]
