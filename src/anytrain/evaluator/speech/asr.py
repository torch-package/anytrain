from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..abc import EvaluatorABC, MetricDict
from ..text import TextComparisonEvaluator, TextInput
from ..text.normalization import coerce_text_batch
from ._whisper import DEFAULT_MODEL_NAME, SUPPORTED_MODEL_NAMES, _OpenAIWhisperTranscriber
from .audio import validate_sample_rate


class _ASRTranscriber(Protocol):
    model: Any | None
    download_root: str

    def transcribe(self, audio: Any, sample_rate: int, **decode_options: Any) -> TextInput:
        ...


class WhisperASREvaluator(EvaluatorABC):
    required_metric_keys = ("bleu", "wer", "chrf")
    default_model_name = DEFAULT_MODEL_NAME
    supported_model_names = SUPPORTED_MODEL_NAMES

    def __init__(
        self,
        *,
        text_evaluator: TextComparisonEvaluator | None = None,
        decode_options: Mapping[str, Any] | None = None,
        model_name: str = default_model_name,
        device: Any | None = None,
        download_root: str | Path | None = None,
        load_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._backend: _ASRTranscriber = _OpenAIWhisperTranscriber(
            model_name=model_name,
            device=device,
            download_root=download_root,
            load_options=load_options,
        )
        self.download_root = self._backend.download_root
        self.text_evaluator = self._text_evaluator(text_evaluator)
        self.decode_options = self._validate_decode_options(decode_options)

    def evaluate(
        self,
        audio: Any,
        sample_rate: int,
        reference_text: TextInput | None = None,
        *,
        target_text: TextInput | None = None,
        **decode_options: Any,
    ) -> MetricDict:
        reference_text = self._reference_text(reference_text, target_text)
        prediction_text = self.transcribe(audio, sample_rate, **decode_options)
        predictions = self._normalize_text_input(prediction_text, label="prediction_text")
        references = self._normalize_text_input(reference_text, label="reference_text")

        if len(predictions) != len(references):
            raise ValueError(
                "prediction/reference text counts must match: "
                f"got {len(predictions)} predictions and {len(references)} references."
            )

        metrics = self.text_evaluator.evaluate(predictions, references)
        return self._extract_required_metrics(metrics)

    def transcribe(self, audio: Any, sample_rate: int, **decode_options: Any) -> TextInput:
        sample_rate = validate_sample_rate(sample_rate)
        options = dict(self.decode_options)
        options.update(decode_options)
        output = self._backend.transcribe(audio, sample_rate, **options)
        if isinstance(output, str):
            return output
        return self._normalize_text_input(output, label="prediction_text")

    def _text_evaluator(
        self,
        text_evaluator: TextComparisonEvaluator | None,
    ) -> TextComparisonEvaluator:
        if text_evaluator is None:
            return TextComparisonEvaluator()
        if not isinstance(text_evaluator, TextComparisonEvaluator):
            raise TypeError("text_evaluator must be a TextComparisonEvaluator.")
        return text_evaluator

    def _reference_text(
        self,
        reference_text: TextInput | None,
        target_text: TextInput | None,
    ) -> TextInput:
        if reference_text is not None and target_text is not None:
            raise ValueError("Provide only one of reference_text or target_text.")
        resolved = reference_text if reference_text is not None else target_text
        if resolved is None:
            raise ValueError(
                "reference_text is required because WhisperASREvaluator.evaluate() "
                "returns only numeric metrics, not transcription text."
            )
        return resolved

    def _extract_required_metrics(self, metrics: object) -> MetricDict:
        if not isinstance(metrics, dict):
            raise TypeError("text_evaluator.evaluate(...) must return a metric dict.")

        missing = [key for key in self.required_metric_keys if key not in metrics]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(
                "text_evaluator.evaluate(...) must return bleu, wer, and chrf metrics; "
                f"missing: {missing_names}."
            )

        return {key: metrics[key] for key in self.required_metric_keys}

    def _normalize_text_input(self, value: TextInput, *, label: str) -> list[str]:
        return list(coerce_text_batch(value, name=label, allow_empty=False))

    def _validate_decode_options(
        self,
        decode_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if decode_options is None:
            return {}
        if not isinstance(decode_options, Mapping):
            raise TypeError("decode_options must be a mapping.")
        return dict(decode_options)
