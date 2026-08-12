from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeGuard, runtime_checkable

import torch
from torch import Tensor

from ...env import torch_home
from ..abc import EvaluatorABC
from ..types import MetricDict
from ._config import (
    DEFAULT_UTMOS_MODEL,
    DEFAULT_UTMOS_REPOSITORY,
    DEFAULT_UTMOS_REVISION,
    DEFAULT_UTMOS_WEIGHTS,
    DEFAULT_UTMOS_WEIGHTS_SHA256,
    UTMOSSemanticConfig,
)
from ._torch import freeze_model
from .audio import load_wave_batch, validate_sample_rate


class TorchHubUTMOSBackend:
    def __init__(
        self,
        *,
        repo: str = f"{DEFAULT_UTMOS_REPOSITORY}:{DEFAULT_UTMOS_REVISION}",
        model_name: str = DEFAULT_UTMOS_MODEL,
        weights: str = DEFAULT_UTMOS_WEIGHTS,
        weights_sha256: str = DEFAULT_UTMOS_WEIGHTS_SHA256,
        device: Any | None = None,
        model: Any | None = None,
        load_options: Mapping[str, Any] | None = None,
        allow_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self.repo = self._validate_name(repo, name="repo")
        self.model_name = self._validate_name(model_name, name="model_name")
        self.weights = self._validate_name(weights, name="weights")
        if Path(self.weights).name != self.weights:
            raise ValueError("weights must be a checkpoint filename, not a path.")
        self.weights_sha256 = self._validate_sha256(weights_sha256)
        self.device = device
        self.model = model
        self.load_options = self._validate_load_options(load_options)
        self.allow_remote_code = self._validate_allow_remote_code(allow_remote_code)
        self.local_files_only = self._validate_local_files_only(local_files_only)

    def score(self, audio: Any, sample_rate: int) -> Tensor:
        wave, sample_rate = load_wave_batch(audio, sample_rate)
        device = self._device()
        model = self._prepare_model(self._load_model(), device=device)

        with torch.inference_mode():
            return model(wave.to(device), sample_rate)

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model

        if not self.allow_remote_code:
            raise ValueError(
                "TorchHubUTMOSBackend remote model loading executes repository code. "
                "Pass allow_remote_code=True, or inject a preloaded model/backend."
            )
        torch_home()
        load_options = dict(self.load_options)
        load_options.setdefault("trust_repo", True)
        source = self.repo
        if self.local_files_only:
            source = str(self._local_repository())
            load_options["source"] = "local"
        self.model = torch.hub.load(source, self.model_name, **load_options)
        if not self.local_files_only:
            self._validate_cached_weights()
        return self.model

    def _local_repository(self) -> Path:
        try:
            repository, revision = self.repo.rsplit(":", 1)
            owner, name = repository.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                "Local-only UTMOS loading requires repo='owner/name:revision'."
            ) from exc
        root = Path(torch.hub.get_dir())
        repository_dir = root / f"{owner}_{name}_{revision.replace('/', '_')}"
        if not repository_dir.is_dir():
            raise FileNotFoundError(
                f"Local-only UTMOS loading requires cached repository {repository_dir}."
            )
        checkpoint = self._checkpoint()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Local-only UTMOS loading requires cached weights {checkpoint}."
            )
        self._validate_cached_weights()
        return repository_dir

    def _validate_cached_weights(self) -> None:
        checkpoint = self._checkpoint()
        if not checkpoint.is_file():
            raise RuntimeError(
                f"UTMOS backend did not materialize the pinned weights at {checkpoint}."
            )
        digest = hashlib.sha256()
        with checkpoint.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != self.weights_sha256:
            raise RuntimeError(
                "UTMOS cached weights do not match the pinned SHA-256: "
                f"{actual} != {self.weights_sha256}."
            )

    def _checkpoint(self) -> Path:
        return Path(torch.hub.get_dir()) / "checkpoints" / self.weights

    def _prepare_model(self, model: Any, *, device: torch.device) -> Any:
        model = freeze_model(model, device=device)
        self.model = model
        return model

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_name(self, value: str, *, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return value

    def _validate_load_options(
        self,
        load_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if load_options is None:
            return {}
        if not isinstance(load_options, Mapping):
            raise TypeError("load_options must be a mapping.")
        return dict(load_options)

    def _validate_allow_remote_code(self, value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("allow_remote_code must be a boolean.")
        return value

    def _validate_local_files_only(self, value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("local_files_only must be a boolean.")
        return value

    def _validate_sha256(self, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("weights_sha256 must be a string.")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("weights_sha256 must be a lowercase SHA-256 digest.")
        return value


@runtime_checkable
class UTMOSBackendProtocol(Protocol):
    def score(self, audio: Any, sample_rate: int) -> float | Sequence[float] | Tensor:
        raise NotImplementedError


class UTMOSEvaluator(EvaluatorABC):
    default_repo = f"{DEFAULT_UTMOS_REPOSITORY}:{DEFAULT_UTMOS_REVISION}"
    default_model_name = DEFAULT_UTMOS_MODEL
    default_weights = DEFAULT_UTMOS_WEIGHTS
    default_weights_sha256 = DEFAULT_UTMOS_WEIGHTS_SHA256

    def __init__(
        self,
        *,
        backend: UTMOSBackendProtocol | None = None,
        repo: str = default_repo,
        model_name: str = default_model_name,
        weights: str = default_weights,
        weights_sha256: str = default_weights_sha256,
        device: Any | None = None,
        backend_load_options: Mapping[str, Any] | None = None,
        allow_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self._semantic_config = self._config(
            backend,
            repo=repo,
            model_name=model_name,
            weights=weights,
            weights_sha256=weights_sha256,
        )
        self.backend = self._backend(
            backend,
            repo=repo,
            model_name=model_name,
            weights=weights,
            weights_sha256=weights_sha256,
            device=device,
            backend_load_options=backend_load_options,
            allow_remote_code=allow_remote_code,
            local_files_only=local_files_only,
        )

    @property
    def semantic_config(self) -> UTMOSSemanticConfig:
        if self._semantic_config is None:
            raise RuntimeError(
                "UTMOS semantic identity requires a pinned built-in backend config."
            )
        return self._semantic_config

    def semantic_id(self) -> str:
        return self.semantic_config.semantic_id()

    def evaluate(self, audio: Any, sample_rate: int) -> MetricDict:
        sample_rate = validate_sample_rate(sample_rate)
        score = self.backend.score(audio, sample_rate)
        scores = self._normalize_scores(score)
        return {"utmos": sum(scores) / len(scores)}

    def call_batch(
        self,
        audios: Sequence[Any],
        sample_rates: Sequence[int],
    ) -> tuple[MetricDict, ...]:
        """Return one UTMOS score per independent clip without length padding."""

        audio_values = tuple(audios)
        rate_values = tuple(sample_rates)
        if len(audio_values) != len(rate_values):
            raise ValueError(
                "audios and sample_rates must have the same batch length: "
                f"got {len(audio_values)} and {len(rate_values)}."
            )
        if not audio_values:
            return ()

        waves = tuple(
            self._single_waveform(audio, sample_rate)
            for audio, sample_rate in zip(audio_values, rate_values, strict=True)
        )
        groups: dict[tuple[int, int], list[int]] = {}
        for index, (wave, sample_rate) in enumerate(waves):
            groups.setdefault((sample_rate, wave.shape[-1]), []).append(index)

        output: list[MetricDict | None] = [None] * len(waves)
        for (sample_rate, _length), indexes in groups.items():
            batch = torch.cat([waves[index][0] for index in indexes])
            scores = self._normalize_scores(
                self.backend.score(batch, sample_rate),
                expected=len(indexes),
            )
            for index, score in zip(indexes, scores, strict=True):
                output[index] = self._validate_metric_dict({"utmos": score})

        if any(metrics is None for metrics in output):
            raise RuntimeError("UTMOS batch scoring left an input without metrics.")
        return tuple(metrics for metrics in output if metrics is not None)

    def _single_waveform(self, audio: Any, sample_rate: int) -> tuple[Tensor, int]:
        wave, sample_rate = load_wave_batch(audio, sample_rate)
        if wave.shape[0] != 1:
            raise ValueError(
                "UTMOSEvaluator.call_batch() inputs must each contain exactly one waveform."
            )
        return wave, sample_rate

    def _backend(
        self,
        backend: UTMOSBackendProtocol | None,
        *,
        repo: str,
        model_name: str,
        weights: str,
        weights_sha256: str,
        device: Any | None,
        backend_load_options: Mapping[str, Any] | None,
        allow_remote_code: bool,
        local_files_only: bool,
    ) -> UTMOSBackendProtocol:
        if backend is None:
            return TorchHubUTMOSBackend(
                repo=repo,
                model_name=model_name,
                weights=weights,
                weights_sha256=weights_sha256,
                device=device,
                load_options=backend_load_options,
                allow_remote_code=allow_remote_code,
                local_files_only=local_files_only,
            )
        if (
            repo != self.default_repo
            or model_name != self.default_model_name
            or weights != self.default_weights
            or weights_sha256 != self.default_weights_sha256
            or device is not None
            or backend_load_options is not None
            or allow_remote_code
            or local_files_only
        ):
            raise ValueError(
                "repo, model_name, weights, weights_sha256, device, backend_load_options, "
                "allow_remote_code, and local_files_only are only used when backend "
                "is not provided."
            )
        if not isinstance(backend, UTMOSBackendProtocol):
            raise TypeError("UTMOSEvaluator backend must implement score(audio, sample_rate).")
        return backend

    def _config(
        self,
        backend: UTMOSBackendProtocol | None,
        *,
        repo: str,
        model_name: str,
        weights: str,
        weights_sha256: str,
    ) -> UTMOSSemanticConfig | None:
        if backend is not None:
            return None
        try:
            repository, revision = repo.rsplit(":", 1)
            return UTMOSSemanticConfig(
                repository=repository,
                revision=revision,
                model_name=model_name,
                weights=weights,
                weights_sha256=weights_sha256,
            )
        except (TypeError, ValueError):
            return None

    def _normalize_scores(
        self,
        score: float | Sequence[float] | Tensor,
        *,
        expected: int | None = None,
    ) -> list[float]:
        if isinstance(score, Tensor):
            if score.dtype == torch.bool:
                raise TypeError("UTMOS backend score tensor must contain floats.")
            values = score.detach().cpu().flatten()
            if values.numel() == 0:
                raise ValueError("UTMOS backend score tensor must contain at least one value.")
            scores = [self._finite_score(value, name="tensor") for value in values.tolist()]
            return self._validate_score_count(scores, expected=expected)

        if self._is_number(score):
            scores = [self._finite_score(score, name="scalar")]
            return self._validate_score_count(scores, expected=expected)

        if isinstance(score, (bytes, bytearray, str)) or not isinstance(score, Sequence):
            raise TypeError("UTMOS backend score must be a float or a sequence of floats.")

        scores = list(score)
        if not scores:
            raise ValueError("UTMOS backend score sequence must contain at least one value.")

        for index, value in enumerate(scores):
            if not self._is_number(value):
                raise TypeError(f"UTMOS backend score[{index}] must be a float.")
        values = [self._finite_score(value, name=f"[{index}]") for index, value in enumerate(scores)]
        return self._validate_score_count(values, expected=expected)

    def _validate_score_count(
        self,
        scores: list[float],
        *,
        expected: int | None,
    ) -> list[float]:
        if expected is not None and len(scores) != expected:
            raise ValueError(
                "UTMOS backend score count must match the batch length: "
                f"got {len(scores)} scores for {expected} inputs."
            )
        return scores

    def _finite_score(self, value: object, *, name: str) -> float:
        if not self._is_number(value):
            raise TypeError(f"UTMOS backend score {name} must be a float.")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"UTMOS backend score {name} must be finite.")
        return resolved

    def _is_number(self, value: object) -> TypeGuard[int | float]:
        return not isinstance(value, bool) and isinstance(value, (float, int))
