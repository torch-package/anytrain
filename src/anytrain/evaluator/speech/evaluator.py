from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..abc import EvaluatorABC
from ..text import TextInput
from ..types import MetricDict
from ._config import SpeechEvaluatorConfig
from .asr import WhisperASREvaluator
from .utmos import UTMOSEvaluator


class SpeechEvaluator(EvaluatorABC):
    def __init__(
        self,
        *,
        asr: WhisperASREvaluator | None = None,
        utmos: UTMOSEvaluator | None = None,
    ) -> None:
        super().__init__()
        self.asr = self._asr(asr)
        self.utmos = self._utmos(utmos)

    @property
    def semantic_config(self) -> SpeechEvaluatorConfig:
        return SpeechEvaluatorConfig(
            whisper=self.asr.semantic_config,
            utmos=self.utmos.semantic_config,
        )

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
        metrics = self.asr(
            audio,
            sample_rate,
            reference_text=reference_text,
            target_text=target_text,
            **decode_options,
        )
        for name, value in self.utmos(audio, sample_rate).items():
            if name in metrics:
                raise ValueError(f"Duplicate speech metric key {name!r}.")
            metrics[name] = value
        return metrics

    def call_batch(
        self,
        audios: Sequence[Any],
        sample_rates: Sequence[int],
        reference_texts: Sequence[str],
        **decode_options: Any,
    ) -> tuple[MetricDict, ...]:
        """Return one speech metric dict per independent audio clip."""

        asr_metrics = self.asr.call_batch(
            audios,
            sample_rates,
            reference_texts,
            **decode_options,
        )
        utmos_metrics = self.utmos.call_batch(audios, sample_rates)
        if len(asr_metrics) != len(utmos_metrics):
            raise ValueError(
                "ASR and UTMOS batch evaluators returned different counts: "
                f"{len(asr_metrics)} and {len(utmos_metrics)}."
            )

        output: list[MetricDict] = []
        for asr_item, utmos_item in zip(asr_metrics, utmos_metrics, strict=True):
            metrics = dict(asr_item)
            for name, value in utmos_item.items():
                if name in metrics:
                    raise ValueError(f"Duplicate speech metric key {name!r}.")
                metrics[name] = value
            output.append(self._validate_metric_dict(metrics))
        return tuple(output)

    def _asr(self, asr: WhisperASREvaluator | None) -> WhisperASREvaluator:
        if asr is None:
            return WhisperASREvaluator()
        if not isinstance(asr, WhisperASREvaluator):
            raise TypeError("asr must be a WhisperASREvaluator.")
        return asr

    def _utmos(self, utmos: UTMOSEvaluator | None) -> UTMOSEvaluator:
        if utmos is None:
            return UTMOSEvaluator()
        if not isinstance(utmos, UTMOSEvaluator):
            raise TypeError("utmos must be a UTMOSEvaluator.")
        return utmos
