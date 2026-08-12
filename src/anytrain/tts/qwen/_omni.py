"""Qwen3-TTS CustomVoice request construction for vLLM-Omni."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from anytrain.tts import TTSOptions
from anytrain.tts._common import positive_int


class PromptLengthEstimator(Protocol):
    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        task_type: str,
        tokenize_prompt: Callable[[str], list[int]],
        codec_language_id: Mapping[str, int] | None,
        spk_is_dialect: Mapping[str, object] | None,
        estimate_ref_code_len: Callable[[object], int | None] | None = None,
    ) -> int: ...


@dataclass
class QwenOmniPromptBuilder:
    tokenizer: Any
    talker_config: Any
    estimator: type[PromptLengthEstimator]

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        cache_dir: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> QwenOmniPromptBuilder:
        try:
            from transformers import AutoTokenizer
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (  # pyright: ignore[reportMissingImports]
                Qwen3TTSConfig,
            )
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (  # pyright: ignore[reportMissingImports]
                Qwen3TTSPromptEmbedsBuilder,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS vLLM-Omni prompt construction requires "
                "`transformers` and a vLLM-Omni build with Qwen3-TTS support."
            ) from exc
        common_kwargs: dict[str, object] = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if cache_dir is not None:
            common_kwargs["cache_dir"] = cache_dir
        if revision is not None:
            common_kwargs["revision"] = revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                padding_side="left",
                **common_kwargs,
            )
            config = Qwen3TTSConfig.from_pretrained(model, **common_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Qwen3-TTS vLLM-Omni prompt metadata {model!r}"
            ) from exc
        return cls(
            tokenizer=tokenizer,
            talker_config=getattr(config, "talker_config", None),
            estimator=cast(type[PromptLengthEstimator], Qwen3TTSPromptEmbedsBuilder),
        )

    def supported_speakers(self) -> tuple[str, ...]:
        values = getattr(self.talker_config, "spk_id", None)
        if not isinstance(values, Mapping):
            return ()
        speakers = tuple(str(speaker) for speaker in values)
        if any(not speaker for speaker in speakers):
            raise RuntimeError("Qwen3-TTS returned invalid supported speaker ids.")
        return speakers

    def build(
        self,
        texts: Sequence[str],
        speakers: Sequence[str],
        languages: Sequence[str],
        instructs: Sequence[str] | None,
        options: TTSOptions,
    ) -> list[dict[str, object]]:
        prompts: list[dict[str, object]] = []
        max_new_tokens = options.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = options.extra.get("max_new_tokens", 2048)
        max_new_tokens = positive_int(max_new_tokens, "max_new_tokens")
        for index, (text, speaker, language) in enumerate(
            zip(texts, speakers, languages, strict=True)
        ):
            instruct = "" if instructs is None else instructs[index]
            info: dict[str, Any] = {
                "task_type": ["CustomVoice"],
                "text": [text],
                "language": [language],
                "speaker": [speaker],
                "instruct": [instruct],
                "max_new_tokens": [max_new_tokens],
            }
            prompt_length = self.estimator.estimate_prompt_len_from_additional_information(
                info,
                task_type="CustomVoice",
                tokenize_prompt=lambda value: list(
                    self.tokenizer(value, padding=False)["input_ids"]
                ),
                codec_language_id=getattr(self.talker_config, "codec_language_id", None),
                spk_is_dialect=getattr(self.talker_config, "spk_is_dialect", None),
            )
            if isinstance(prompt_length, bool) or not isinstance(prompt_length, int):
                raise TypeError("Qwen3-TTS prompt estimator must return an integer.")
            if prompt_length <= 0:
                raise ValueError("Qwen3-TTS prompt estimator returned a non-positive length.")
            prompts.append(
                {
                    "prompt_token_ids": [0] * prompt_length,
                    "additional_information": info,
                }
            )
        return prompts
