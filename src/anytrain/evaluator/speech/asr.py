from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..abc import EvaluatorABC
from ..text import TextComparisonEvaluator, TextInput
from ..text.normalization import coerce_text_batch
from ..types import MetricDict
from ._config import OPENAI_WHISPER_VERSION, WhisperSemanticConfig
from ._whisper import DEFAULT_MODEL_NAME, SUPPORTED_MODEL_NAMES, _OpenAIWhisperTranscriber
from .audio import validate_sample_rate


class _ASRTranscriber(Protocol):
    model: Any | None
    download_root: str

    def transcribe(self, audio: Any, sample_rate: int, **decode_options: Any) -> TextInput:
        ...

    def transcribe_batch(
        self,
        audios: Sequence[Any],
        sample_rates: Sequence[int],
        **decode_options: Any,
    ) -> list[str]: ...


class WhisperASREvaluator(EvaluatorABC):
    required_metric_keys = ("bleu", "wer", "chrf", "cer")
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
        local_files_only: bool = False,
        backend_version: str = OPENAI_WHISPER_VERSION,
    ) -> None:
        super().__init__()
        self.text_evaluator = self._text_evaluator(text_evaluator)
        self.decode_options = self._validate_decode_options(decode_options)
        self._semantic_config = WhisperSemanticConfig(
            model_name=model_name,
            decode_options=tuple(sorted(self.decode_options.items())),
            normalization=self.text_evaluator.normalization,
            bleu_smoothing=self.text_evaluator.bleu_smoothing,
            backend_version=backend_version,
        )
        self.model_name = self._semantic_config.model_name
        self.device = device
        self._backend: _ASRTranscriber = _OpenAIWhisperTranscriber(
            model_name=self.model_name,
            device=device,
            download_root=download_root,
            load_options=load_options,
            local_files_only=local_files_only,
            backend_version=backend_version,
            artifact_sha256=self._semantic_config.artifact_sha256,
        )
        self.download_root = self._backend.download_root

    @property
    def semantic_config(self) -> WhisperSemanticConfig:
        return self._semantic_config

    def semantic_id(self) -> str:
        return self.semantic_config.semantic_id()

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

    def call_batch(
        self,
        audios: Sequence[Any],
        sample_rates: Sequence[int],
        reference_texts: Sequence[str],
        **decode_options: Any,
    ) -> tuple[MetricDict, ...]:
        """Return per-clip ASR metrics from one batched transcription call."""

        audio_values = tuple(audios)
        rate_values = tuple(sample_rates)
        references = tuple(
            coerce_text_batch(
                reference_texts,
                name="reference_texts",
            )
        )
        self._validate_batch_lengths(audio_values, rate_values, references)
        if not audio_values:
            return ()

        options = dict(self.decode_options)
        options.update(decode_options)
        predictions = self._backend.transcribe_batch(
            audio_values,
            tuple(validate_sample_rate(rate) for rate in rate_values),
            **options,
        )
        if len(predictions) != len(references):
            raise ValueError(
                "prediction/reference text counts must match: "
                f"got {len(predictions)} predictions and {len(references)} references."
            )

        output: list[MetricDict] = []
        for prediction, reference in zip(predictions, references, strict=True):
            metrics = self.text_evaluator.evaluate(prediction, reference)
            output.append(
                self._validate_metric_dict(self._extract_required_metrics(metrics))
            )
        return tuple(output)

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
                "text_evaluator.evaluate(...) must return bleu, wer, chrf, and cer metrics; "
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

    def _validate_batch_lengths(
        self,
        audios: Sequence[Any],
        sample_rates: Sequence[int],
        reference_texts: Sequence[str],
    ) -> None:
        lengths = (len(audios), len(sample_rates), len(reference_texts))
        if len(set(lengths)) != 1:
            raise ValueError(
                "audios, sample_rates, and reference_texts must have the same "
                f"batch length: got {lengths[0]}, {lengths[1]}, and {lengths[2]}."
            )
