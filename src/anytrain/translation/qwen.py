"""Qwen causal-language-model adapter for deterministic universal translation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .protocol import validate_translation_batch, validate_translation_outputs

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class QwenTranslatorConfig:
    model: str = DEFAULT_MODEL
    revision: str | None = None
    max_new_tokens: int = 256
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    generate_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty("model", self.model)
        if self.revision is not None:
            _non_empty("revision", self.revision)
        _positive_int("max_new_tokens", self.max_new_tokens)
        _mapping("load_kwargs", self.load_kwargs)
        _mapping("generate_kwargs", self.generate_kwargs)
        if "max_new_tokens" in self.generate_kwargs:
            raise ValueError(
                "generate_kwargs must not contain max_new_tokens; use the explicit option."
            )

    def hash_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "revision": self.revision,
            "max_new_tokens": self.max_new_tokens,
            "load_kwargs": _hashable(self.load_kwargs),
            "generate_kwargs": _hashable(self.generate_kwargs),
        }


@dataclass
class QwenTranslator:
    model: Any
    tokenizer: Any
    config: QwenTranslatorConfig = field(default_factory=QwenTranslatorConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_loaded_from: str = "custom"

    name: str = "qwen-translator"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is None:
                raise ValueError(
                    "Qwen translator tokenizer requires pad_token_id or eos_token."
                )
            self.tokenizer.pad_token = eos

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str | torch.device | None = None,
        max_new_tokens: int = 256,
        generate_kwargs: Mapping[str, object] | None = None,
        **load_kwargs: object,
    ) -> QwenTranslator:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "anytrain.translation.qwen requires transformers; install "
                "`anytrain[translation]`."
            ) from exc

        source = str(model)
        config = QwenTranslatorConfig(
            model=source,
            revision=revision,
            max_new_tokens=max_new_tokens,
            load_kwargs=load_kwargs,
            generate_kwargs=dict(generate_kwargs or {}),
        )
        resolved_source = _resolve_pretrained_source(
            source,
            cache_dir=load_kwargs.get("cache_dir"),
            local_files_only=load_kwargs.get("local_files_only", False),
            revision=revision,
        )
        tokenizer_options: dict[str, object] = {
            name: value
            for name, value in load_kwargs.items()
            if name
            in {
                "cache_dir",
                "force_download",
                "local_files_only",
                "proxies",
                "subfolder",
                "token",
                "trust_remote_code",
            }
        }
        if revision is not None:
            tokenizer_options["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(resolved_source, **tokenizer_options)
        options = dict(load_kwargs)
        if revision is not None:
            options["revision"] = revision
        if device is not None and "device_map" not in options:
            options["device_map"] = str(device)
        loaded = AutoModelForCausalLM.from_pretrained(resolved_source, **options).eval()
        return cls(
            model=loaded,
            tokenizer=tokenizer,
            config=config,
            device=_device(loaded, device),
            model_loaded_from="transformers",
        )

    def config_hash(self) -> str:
        encoded = json.dumps(
            self.config.hash_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def translate(
        self,
        texts: Sequence[str],
        source_languages: Sequence[str],
        target_languages: Sequence[str],
    ) -> list[str]:
        values, sources, targets = validate_translation_batch(
            texts,
            source_languages,
            target_languages,
        )
        prompts = [
            translation_prompt(text, source_language=source, target_language=target, tokenizer=self.tokenizer)
            for text, source, target in zip(values, sources, targets, strict=False)
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        move = getattr(inputs, "to", None)
        if callable(move):
            inputs = move(self.device)
        if not isinstance(inputs, Mapping):
            raise TypeError("Qwen translator tokenizer output must be a mapping.")
        inputs = cast(Mapping[str, Any], inputs)
        input_ids = inputs["input_ids"]
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise TypeError("Qwen translator tokenizer must return rank-2 input_ids.")
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            **dict(self.config.generate_kwargs),
        )
        if not isinstance(generated, Tensor) or generated.ndim != 2:
            raise TypeError("Qwen translator generate() must return a rank-2 Tensor.")
        if generated.shape[0] != len(values):
            raise ValueError("Qwen translator output count must match the input batch.")
        decoded = self.tokenizer.batch_decode(
            generated[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        outputs = [_translation_text(value) for value in decoded]
        return validate_translation_outputs(outputs, expected=len(values))


def translation_prompt(
    text: str,
    *,
    source_language: str,
    target_language: str,
    tokenizer: Any,
) -> str:
    """Build a prompt whose response contains translated text only."""

    source = _non_empty("source_language", source_language)
    target = _non_empty("target_language", target_language)
    value = _non_empty("text", text)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise translation engine. Return only the translated "
                "text, without explanations, labels, or quotation marks."
            ),
        },
        {
            "role": "user",
            "content": f"Translate this {source} text into {target}:\n{value}",
        },
    ]
    template = tokenizer.apply_chat_template
    options: dict[str, object] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _accepts_keyword(template, "enable_thinking"):
        options["enable_thinking"] = False
    prompt = template(messages, **options)
    if not isinstance(prompt, str) or not prompt:
        raise TypeError("Qwen translator chat template must return a non-empty string.")
    return prompt


def _translation_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Qwen translator decoder outputs must be strings.")
    text = value.rsplit("</think>", 1)[-1].strip()
    if not text:
        raise ValueError("Qwen translator generated empty text.")
    return text


def _device(model: object, value: str | torch.device | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    current = getattr(model, "device", None)
    if current is not None:
        return torch.device(current)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return torch.device(next(iter(cast(Any, parameters)())).device)
        except StopIteration:
            pass
    return torch.device("cpu")


def _resolve_pretrained_source(
    source: str,
    *,
    cache_dir: object,
    local_files_only: object,
    revision: str | None,
) -> str:
    if Path(source).expanduser().exists() or local_files_only is not True:
        return source
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Local-only Qwen translation loading requires huggingface-hub; install "
            '`anytrain[translation]`.'
        ) from exc
    return snapshot_download(
        repo_id=source,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=True,
        revision=revision,
    )


def _accepts_keyword(method: Any, name: str) -> bool:
    parameters = signature(method).parameters.values()
    return any(
        parameter.name == name or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _hashable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _hashable(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_hashable(item) for item in value]
    return str(value)


def _mapping(name: str, value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings.")


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


__all__ = [
    "DEFAULT_MODEL",
    "QwenTranslator",
    "QwenTranslatorConfig",
    "translation_prompt",
]
