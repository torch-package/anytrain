from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from ...env import torch_home
from ..abc import EvaluatorABC, MetricDict
from ._torch import freeze_model
from .audio import load_wave_batch, validate_sample_rate


class TorchHubUTMOSBackend:
    def __init__(
        self,
        *,
        repo: str = "tarepan/SpeechMOS:v1.2.0",
        model_name: str = "utmos22_strong",
        device: Any | None = None,
        model: Any | None = None,
        load_options: Mapping[str, Any] | None = None,
        allow_remote_code: bool = False,
    ) -> None:
        self.repo = self._validate_name(repo, name="repo")
        self.model_name = self._validate_name(model_name, name="model_name")
        self.device = device
        self.model = model
        self.load_options = self._validate_load_options(load_options)
        self.allow_remote_code = self._validate_allow_remote_code(allow_remote_code)

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
        self.model = torch.hub.load(self.repo, self.model_name, **load_options)
        return self.model

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


@runtime_checkable
class UTMOSBackendProtocol(Protocol):
    def score(self, audio: Any, sample_rate: int) -> float | Sequence[float] | Tensor:
        raise NotImplementedError


class UTMOSEvaluator(EvaluatorABC):
    default_repo = "tarepan/SpeechMOS:v1.2.0"
    default_model_name = "utmos22_strong"

    def __init__(
        self,
        *,
        backend: UTMOSBackendProtocol | None = None,
        repo: str = default_repo,
        model_name: str = default_model_name,
        device: Any | None = None,
        backend_load_options: Mapping[str, Any] | None = None,
        allow_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.backend = self._backend(
            backend,
            repo=repo,
            model_name=model_name,
            device=device,
            backend_load_options=backend_load_options,
            allow_remote_code=allow_remote_code,
        )

    def evaluate(self, audio: Any, sample_rate: int) -> MetricDict:
        sample_rate = validate_sample_rate(sample_rate)
        score = self.backend.score(audio, sample_rate)
        scores = self._normalize_scores(score)
        return {"utmos": sum(scores) / len(scores)}

    def _backend(
        self,
        backend: UTMOSBackendProtocol | None,
        *,
        repo: str,
        model_name: str,
        device: Any | None,
        backend_load_options: Mapping[str, Any] | None,
        allow_remote_code: bool,
    ) -> UTMOSBackendProtocol:
        if backend is None:
            return TorchHubUTMOSBackend(
                repo=repo,
                model_name=model_name,
                device=device,
                load_options=backend_load_options,
                allow_remote_code=allow_remote_code,
            )
        if (
            repo != self.default_repo
            or model_name != self.default_model_name
            or device is not None
            or backend_load_options is not None
            or allow_remote_code
        ):
            raise ValueError(
                "repo, model_name, device, backend_load_options, and allow_remote_code are "
                "only used when backend is not provided."
            )
        if not isinstance(backend, UTMOSBackendProtocol):
            raise TypeError("UTMOSEvaluator backend must implement score(audio, sample_rate).")
        return backend

    def _normalize_scores(self, score: float | Sequence[float] | Tensor) -> list[float]:
        if isinstance(score, Tensor):
            if score.dtype == torch.bool:
                raise TypeError("UTMOS backend score tensor must contain floats.")
            values = score.detach().cpu().flatten()
            if values.numel() == 0:
                raise ValueError("UTMOS backend score tensor must contain at least one value.")
            return [float(value) for value in values.tolist()]

        if self._is_number(score):
            return [float(score)]

        if isinstance(score, (bytes, bytearray, str)) or not isinstance(score, Sequence):
            raise TypeError("UTMOS backend score must be a float or a sequence of floats.")

        scores = list(score)
        if not scores:
            raise ValueError("UTMOS backend score sequence must contain at least one value.")

        for index, value in enumerate(scores):
            if not self._is_number(value):
                raise TypeError(f"UTMOS backend score[{index}] must be a float.")
        return [float(value) for value in scores]

    def _is_number(self, value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (float, int))
