from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from anytrain._compat import strict_zip

from ...env import whisper_root
from ..text import TextInput
from ._torch import freeze_model
from .audio import load_wave_batch, resample_wave

DEFAULT_MODEL_NAME = "large-v3"
SUPPORTED_MODEL_NAMES = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
)

_SUPPORTED_MODEL_NAME_SET = frozenset(SUPPORTED_MODEL_NAMES)
_WHISPER_CHUNK_SECONDS = 30
_WHISPER_TARGET_SAMPLE_RATE = 16000
_WHISPER_CHUNK_SAMPLES = _WHISPER_CHUNK_SECONDS * _WHISPER_TARGET_SAMPLE_RATE
_WHISPER_MEL_FRAMES = 3000
_DEFAULT_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_SHORT_BATCH_DEFAULT_OPTIONS: dict[str, object] = {
    "verbose": None,
    "condition_on_previous_text": True,
    "initial_prompt": None,
    "carry_initial_prompt": False,
    "word_timestamps": False,
    "clip_timestamps": "0",
    "hallucination_silence_threshold": None,
}


class _OpenAIWhisperTranscriber:
    target_sample_rate = _WHISPER_TARGET_SAMPLE_RATE

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Any | None = None,
        download_root: str | Path | None = None,
        load_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = self._validate_model_name(model_name)
        self.device = device
        root = Path(download_root).expanduser() if download_root is not None else whisper_root()
        self.download_root = str(root)
        self.model: Any | None = None
        self.load_options = self._validate_load_options(load_options)

    def transcribe(
        self,
        audio: Any,
        sample_rate: int,
        **decode_options: Any,
    ) -> TextInput:
        wave, sample_rate = load_wave_batch(audio, sample_rate)
        wave = resample_wave(wave, sample_rate, self.target_sample_rate)
        model = self._prepare_model(self._load_model())
        whisper = self._load_whisper_module()

        with torch.inference_mode():
            predictions = self._transcribe_short_batch(whisper, model, wave, decode_options)
            if predictions is not None:
                return predictions

            # openai-whisper batches only the lower-level 30-second mel decode path.
            # Keep transcribe per-sample to preserve its long-audio orchestration.
            predictions = [
                self._extract_text(model.transcribe(sample.cpu().numpy(), **decode_options))
                for sample in wave
            ]
        return predictions[0] if len(predictions) == 1 else predictions

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model

        whisper = self._load_whisper_module()
        load_options = dict(self.load_options)
        if self.device is not None:
            load_options["device"] = self.device
        if self.download_root is not None:
            load_options["download_root"] = self.download_root
        self.model = whisper.load_model(self.model_name, **load_options)
        return self.model

    def _load_whisper_module(self) -> Any:
        try:
            return importlib.import_module("whisper")
        except ImportError as exc:
            raise ImportError(
                "WhisperASREvaluator requires the openai-whisper package. "
                "Install it with python -m pip install openai-whisper."
            ) from exc

    def _prepare_model(self, model: Any) -> Any:
        model = freeze_model(model, device=self.device)
        self.model = model
        return model

    def _transcribe_short_batch(
        self,
        whisper: Any,
        model: Any,
        wave: torch.Tensor,
        decode_options: Mapping[str, Any],
    ) -> list[str] | None:
        if wave.shape[0] <= 1 or wave.shape[-1] > _WHISPER_CHUNK_SAMPLES:
            return None
        if not self._has_short_batch_decode_api(whisper, model):
            return None

        config = self._short_batch_decode_config(whisper, model, decode_options)
        if config is None:
            return None

        options, temperatures, compression_threshold, logprob_threshold, no_speech_threshold = config
        device = self._model_device(model)
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(wave, length=_WHISPER_CHUNK_SAMPLES, axis=-1),
            n_mels=self._model_n_mels(model),
            device=device,
        )
        mel = whisper.pad_or_trim(mel, length=_WHISPER_MEL_FRAMES, axis=-1)

        results: list[object | None] = [None] * wave.shape[0]
        active = list(range(wave.shape[0]))
        for index, temperature in enumerate(temperatures):
            decode_kwargs = dict(options)
            if temperature > 0:
                decode_kwargs.pop("beam_size", None)
                decode_kwargs.pop("patience", None)
            else:
                decode_kwargs.pop("best_of", None)

            decoding = whisper.DecodingOptions(**decode_kwargs, temperature=temperature)
            batch = self._coerce_decode_result_batch(model.decode(mel[active], decoding), len(active))
            next_active: list[int] = []
            for sample_index, result in strict_zip(active, batch):
                if index < len(temperatures) - 1 and self._needs_temperature_fallback(
                    result,
                    compression_threshold=compression_threshold,
                    logprob_threshold=logprob_threshold,
                    no_speech_threshold=no_speech_threshold,
                ):
                    next_active.append(sample_index)
                    continue
                results[sample_index] = result

            if not next_active:
                break
            active = next_active

        if any(result is None for result in results):
            raise RuntimeError("Whisper short batch decode left some inputs without results.")
        return [
            ""
            if self._should_skip_no_speech(
                result,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
            )
            else self._extract_text(result)
            for result in results
        ]

    def _has_short_batch_decode_api(self, whisper: Any, model: Any) -> bool:
        return (
            callable(getattr(whisper, "pad_or_trim", None))
            and callable(getattr(whisper, "log_mel_spectrogram", None))
            and callable(getattr(whisper, "DecodingOptions", None))
            and callable(getattr(model, "decode", None))
        )

    def _short_batch_decode_config(
        self,
        whisper: Any,
        model: Any,
        decode_options: Mapping[str, Any],
    ) -> (
        tuple[dict[str, Any], tuple[float, ...], float | None, float | None, float | None]
        | None
    ):
        try:
            option_names = {field.name for field in fields(whisper.DecodingOptions)}
        except TypeError:
            return None

        options = dict(decode_options)
        if self._requires_full_transcribe(options):
            return None

        compression_threshold = options.pop("compression_ratio_threshold", 2.4)
        logprob_threshold = options.pop("logprob_threshold", -1.0)
        no_speech_threshold = options.pop("no_speech_threshold", 0.6)
        temperatures = self._coerce_temperature_values(
            options.pop("temperature", _DEFAULT_TEMPERATURES)
        )
        if temperatures is None:
            return None

        unknown = set(options) - option_names
        if unknown:
            return None

        device = self._model_device(model)
        if device.type == "cpu" and options.get("fp16", True):
            options["fp16"] = False

        return options, temperatures, compression_threshold, logprob_threshold, no_speech_threshold

    def _requires_full_transcribe(self, options: dict[str, Any]) -> bool:
        for name, default in _SHORT_BATCH_DEFAULT_OPTIONS.items():
            if name not in options:
                continue
            value = options.pop(name)
            if value != default:
                return True

        if "prepend_punctuations" in options:
            options.pop("prepend_punctuations")
        if "append_punctuations" in options:
            options.pop("append_punctuations")
        return False

    def _coerce_temperature_values(self, temperature: object) -> tuple[float, ...] | None:
        if isinstance(temperature, bool):
            return None
        if isinstance(temperature, (int, float)):
            return (float(temperature),)
        if isinstance(temperature, (bytes, bytearray, str)) or not isinstance(
            temperature, Sequence
        ):
            return None

        values: list[float] = []
        for value in temperature:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            values.append(float(value))
        return tuple(values) if values else None

    def _coerce_decode_result_batch(self, result: object, expected_length: int) -> list[object]:
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            values = list(result)
        else:
            values = [result]
        if len(values) != expected_length:
            raise ValueError(
                "Whisper batch decode result count must match the input batch length: "
                f"got {len(values)} results and {expected_length} inputs."
            )
        return values

    def _needs_temperature_fallback(
        self,
        result: object,
        *,
        compression_threshold: float | None,
        logprob_threshold: float | None,
        no_speech_threshold: float | None,
    ) -> bool:
        needs_fallback = False
        compression = self._result_float(result, "compression_ratio")
        avg_logprob = self._result_float(result, "avg_logprob")
        no_speech_prob = self._result_float(result, "no_speech_prob")

        if compression_threshold is not None and compression is not None:
            needs_fallback = compression > compression_threshold
        if logprob_threshold is not None and avg_logprob is not None:
            needs_fallback = needs_fallback or avg_logprob < logprob_threshold
        if (
            no_speech_threshold is not None
            and logprob_threshold is not None
            and no_speech_prob is not None
            and avg_logprob is not None
            and no_speech_prob > no_speech_threshold
            and avg_logprob < logprob_threshold
        ):
            return False
        return needs_fallback

    def _should_skip_no_speech(
        self,
        result: object,
        *,
        no_speech_threshold: float | None,
        logprob_threshold: float | None,
    ) -> bool:
        no_speech_prob = self._result_float(result, "no_speech_prob")
        if no_speech_threshold is None or no_speech_prob is None:
            return False
        should_skip = no_speech_prob > no_speech_threshold
        avg_logprob = self._result_float(result, "avg_logprob")
        if logprob_threshold is not None and avg_logprob is not None:
            should_skip = should_skip and avg_logprob <= logprob_threshold
        return should_skip

    def _result_float(self, result: object, name: str) -> float | None:
        value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
        if value is None:
            return None
        return float(value)

    def _model_device(self, model: Any) -> torch.device:
        device = getattr(model, "device", None)
        if device is not None:
            return torch.device(device)
        if isinstance(model, torch.nn.Module):
            try:
                return next(model.parameters()).device
            except StopIteration:
                return torch.device("cpu")
        return torch.device("cpu")

    def _model_n_mels(self, model: Any) -> int:
        dims = getattr(model, "dims", None)
        n_mels = getattr(dims, "n_mels", 80)
        if isinstance(n_mels, bool) or not isinstance(n_mels, int) or n_mels <= 0:
            raise TypeError("Whisper model dims.n_mels must be a positive integer.")
        return n_mels

    def _extract_text(self, result: object) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            text = result.get("text")
            if isinstance(text, str):
                return text
            raise TypeError("Whisper transcribe result must contain a string text value.")
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        raise TypeError("Whisper transcribe result must be a string or mapping.")

    def _validate_model_name(self, model_name: str) -> str:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if model_name not in _SUPPORTED_MODEL_NAME_SET:
            names = ", ".join(SUPPORTED_MODEL_NAMES)
            raise ValueError(f"model_name must be one of: {names}.")
        return model_name

    def _validate_load_options(
        self,
        load_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if load_options is None:
            return {}
        if not isinstance(load_options, Mapping):
            raise TypeError("load_options must be a mapping.")
        return dict(load_options)
