"""Pinned Qwen translation semantics, lazy loading, and token planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from anytrain._semantic import semantic_hash, semantic_id, semantic_options

from ._validation import non_empty_string, positive_int
from .protocol import validate_translation_batch, validate_translation_outputs

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
PROMPT_PROTOCOL = "qwen-translation-prompt-v1"
DECODER_PROTOCOL = "qwen-translation-decoder-v1"

_SEMANTIC_ID_PREFIX = "qwen-translation-v1"
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_RESERVED_GENERATION_OPTIONS = frozenset(
    {"do_sample", "max_new_tokens", "pad_token_id", "use_cache"}
)


@dataclass(frozen=True)
class QwenTranslationSpec:
    """Output-affecting Qwen translation semantics.

    ``revision`` identifies the model artifact. Physical cache paths, device
    placement, and offline policy belong to :class:`QwenTranslationLoadOptions`.
    """

    revision: str
    model: str = DEFAULT_MODEL
    dtype: str | None = None
    max_new_tokens: int = 256
    generation_options: tuple[tuple[str, object], ...] = ()
    prompt_protocol: str = PROMPT_PROTOCOL
    decoder_protocol: str = DECODER_PROTOCOL

    def __post_init__(self) -> None:
        non_empty_string(self.model, "model")
        if Path(self.model).is_absolute():
            raise ValueError("model must be a logical artifact id, not an absolute path.")
        _commit("revision", self.revision)
        if self.dtype is not None:
            if not isinstance(self.dtype, str):
                raise TypeError("dtype must be a string or None.")
            if self.dtype not in _DTYPES:
                raise ValueError(f"dtype must be one of {tuple(_DTYPES)!r} or None.")
        positive_int(self.max_new_tokens, "max_new_tokens")
        options = semantic_options(self.generation_options, name="generation_options")
        reserved = sorted(set(options) & _RESERVED_GENERATION_OPTIONS)
        if reserved:
            raise ValueError(
                "generation_options contains reserved options: " + ", ".join(reserved)
            )
        non_empty_string(self.prompt_protocol, "prompt_protocol")
        if self.prompt_protocol != PROMPT_PROTOCOL:
            raise ValueError(f"unsupported Qwen prompt protocol {self.prompt_protocol!r}.")
        non_empty_string(self.decoder_protocol, "decoder_protocol")
        if self.decoder_protocol != DECODER_PROTOCOL:
            raise ValueError(f"unsupported Qwen decoder protocol {self.decoder_protocol!r}.")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "decoder_protocol": self.decoder_protocol,
            "dtype": self.dtype,
            "generation_options": semantic_options(
                self.generation_options,
                name="generation_options",
            ),
            "max_new_tokens": self.max_new_tokens,
            "model": self.model,
            "prompt_protocol": self.prompt_protocol,
            "revision": self.revision,
        }

    def config_hash(self) -> str:
        return semantic_hash(self.semantic_payload())

    def semantic_id(self) -> str:
        return semantic_id(_SEMANTIC_ID_PREFIX, self.semantic_payload())

    def factory(
        self,
        load_options: QwenTranslationLoadOptions | None = None,
    ) -> QwenTranslatorFactory:
        return QwenTranslatorFactory(
            self,
            QwenTranslationLoadOptions() if load_options is None else load_options,
        )

    def token_estimator(
        self,
        load_options: QwenTranslationLoadOptions | None = None,
    ) -> QwenTranslationTokenEstimator:
        return QwenTranslationTokenEstimator(
            self,
            QwenTranslationLoadOptions() if load_options is None else load_options,
        )


@dataclass(frozen=True)
class QwenTranslationLoadOptions:
    """Physical loading policy excluded from translation semantic identity."""

    cache_dir: str | Path | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.cache_dir is not None and not isinstance(self.cache_dir, (str, Path)):
            raise TypeError("cache_dir must be a string, Path, or None.")
        if isinstance(self.cache_dir, str):
            non_empty_string(self.cache_dir, "cache_dir")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be a boolean.")


@dataclass(frozen=True)
class QwenTranslatorFactory:
    """Pickle-safe factory that loads one backend only when a worker calls it."""

    spec: QwenTranslationSpec
    load_options: QwenTranslationLoadOptions = field(
        default_factory=QwenTranslationLoadOptions
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, QwenTranslationSpec):
            raise TypeError("spec must be a QwenTranslationSpec.")
        if not isinstance(self.load_options, QwenTranslationLoadOptions):
            raise TypeError("load_options must be QwenTranslationLoadOptions.")

    def __call__(self, device: str) -> QwenTranslator:
        non_empty_string(device, "device")
        return QwenTranslator.from_pretrained(
            self.spec,
            device=device,
            load_options=self.load_options,
        )


@dataclass
class QwenTranslationTokenEstimator:
    """Count exact prompt tokens and decoded-text token proxies without a model."""

    spec: QwenTranslationSpec
    load_options: QwenTranslationLoadOptions = field(
        default_factory=QwenTranslationLoadOptions
    )
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, QwenTranslationSpec):
            raise TypeError("spec must be a QwenTranslationSpec.")
        if not isinstance(self.load_options, QwenTranslationLoadOptions):
            raise TypeError("load_options must be QwenTranslationLoadOptions.")

    @property
    def tokenizer_loaded(self) -> bool:
        return self._tokenizer is not None

    def prompt_tokens(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> int:
        tokenizer = self._load_tokenizer()
        prompt = translation_prompt(
            text,
            source_language=source_language,
            target_language=target_language,
            tokenizer=tokenizer,
        )
        return _token_count(tokenizer, prompt)

    def planned_tokens(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> int:
        return self.prompt_tokens(
            text,
            source_language=source_language,
            target_language=target_language,
        ) + self.spec.max_new_tokens

    def decoded_target_tokens(self, text: str) -> int:
        return _token_count(self._load_tokenizer(), non_empty_string(text, "text"))

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.spec, self.load_options)
        return self._tokenizer


@dataclass
class QwenTranslator:
    model: Any
    tokenizer: Any
    spec: QwenTranslationSpec
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_loaded_from: str = "custom"

    name: ClassVar[str] = "qwen-translator"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, QwenTranslationSpec):
            raise TypeError("spec must be a QwenTranslationSpec.")
        non_empty_string(self.model_loaded_from, "model_loaded_from")
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
        spec: QwenTranslationSpec,
        *,
        device: str | torch.device | None = None,
        load_options: QwenTranslationLoadOptions | None = None,
    ) -> QwenTranslator:
        if not isinstance(spec, QwenTranslationSpec):
            raise TypeError("spec must be a QwenTranslationSpec.")
        resolved_load = (
            QwenTranslationLoadOptions() if load_options is None else load_options
        )
        if not isinstance(resolved_load, QwenTranslationLoadOptions):
            raise TypeError("load_options must be QwenTranslationLoadOptions.")
        if device is not None and not isinstance(device, (str, torch.device)):
            raise TypeError("device must be a string, torch.device, or None.")
        if isinstance(device, str):
            non_empty_string(device, "device")
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "anytrain.translation.qwen requires transformers; install "
                "`anytrain[translation]`."
            ) from exc

        source = _pretrained_source(spec, resolved_load)
        tokenizer = _load_tokenizer(spec, resolved_load, source=source)
        options = _hub_options(spec, resolved_load)
        if spec.dtype is not None:
            options["dtype"] = _DTYPES[spec.dtype]
        if device is not None:
            options["device_map"] = str(device)
        loaded = AutoModelForCausalLM.from_pretrained(source, **options).eval()
        return cls(
            model=loaded,
            tokenizer=tokenizer,
            spec=spec,
            device=_device(loaded, device),
            model_loaded_from="transformers",
        )

    def config_hash(self) -> str:
        return self.spec.config_hash()

    def semantic_id(self) -> str:
        return self.spec.semantic_id()

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
            translation_prompt(
                text,
                source_language=source,
                target_language=target,
                tokenizer=self.tokenizer,
            )
            for text, source, target in zip(values, sources, targets, strict=True)
        ]
        encode = self.tokenizer
        if not callable(encode):
            raise TypeError("Qwen translator tokenizer must be callable.")
        inputs = encode(
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
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise TypeError("Qwen translator tokenizer must return rank-2 input_ids.")
        if input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex():
            raise TypeError("Qwen translator input_ids must use an integer dtype.")
        if input_ids.shape[0] != len(values):
            raise ValueError("Qwen translator input_ids batch dimension must match the input batch.")
        generate = getattr(self.model, "generate", None)
        if not callable(generate):
            raise TypeError("Qwen translator model must provide generate().")
        generated = generate(
            **inputs,
            max_new_tokens=self.spec.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            **semantic_options(
                self.spec.generation_options,
                name="generation_options",
            ),
        )
        if not isinstance(generated, Tensor) or generated.ndim != 2:
            raise TypeError("Qwen translator generate() must return a rank-2 Tensor.")
        if generated.dtype == torch.bool or generated.is_floating_point() or generated.is_complex():
            raise TypeError("Qwen translator generated token ids must use an integer dtype.")
        if generated.shape[0] != len(values):
            raise ValueError("Qwen translator output count must match the input batch.")
        if generated.shape[1] < input_ids.shape[1]:
            raise ValueError("Qwen translator generated ids must include the prompt prefix.")
        batch_decode = getattr(self.tokenizer, "batch_decode", None)
        if not callable(batch_decode):
            raise TypeError("Qwen translator tokenizer must provide batch_decode().")
        decoded = validate_translation_outputs(
            cast(
                Sequence[str],
                batch_decode(
                    generated[:, input_ids.shape[1] :],
                    skip_special_tokens=True,
                ),
            ),
            expected=len(values),
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
    """Build the v1 prompt whose response contains translated text only."""

    source = non_empty_string(source_language, "source_language")
    target = non_empty_string(target_language, "target_language")
    value = non_empty_string(text, "text")
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
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise TypeError("Qwen translator tokenizer must provide apply_chat_template().")
    options: dict[str, object] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _accepts_keyword(template, "enable_thinking"):
        options["enable_thinking"] = False
    prompt = template(messages, **options)
    return non_empty_string(prompt, "Qwen translator chat template output")


def _load_tokenizer(
    spec: QwenTranslationSpec,
    load_options: QwenTranslationLoadOptions,
    *,
    source: str | None = None,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Qwen token estimation requires transformers; install "
            "`anytrain[translation]`."
        ) from exc
    resolved_source = _pretrained_source(spec, load_options) if source is None else source
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source,
        **_hub_options(spec, load_options),
    )
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    return tokenizer


def _pretrained_source(
    spec: QwenTranslationSpec,
    load_options: QwenTranslationLoadOptions,
) -> str:
    source = spec.model
    if Path(source).expanduser().exists() or not load_options.local_files_only:
        return source
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Local-only Qwen translation loading requires huggingface-hub; install "
            "`anytrain[translation]`."
        ) from exc
    snapshot = snapshot_download(
        repo_id=source,
        cache_dir=(
            None if load_options.cache_dir is None else str(load_options.cache_dir)
        ),
        local_files_only=True,
        revision=spec.revision,
    )
    return non_empty_string(snapshot, "Qwen translation snapshot path")


def _hub_options(
    spec: QwenTranslationSpec,
    load_options: QwenTranslationLoadOptions,
) -> dict[str, object]:
    options: dict[str, object] = {"revision": spec.revision}
    if load_options.cache_dir is not None:
        options["cache_dir"] = str(load_options.cache_dir)
    if load_options.local_files_only:
        options["local_files_only"] = True
    return options


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True)
    if not isinstance(encoded, Mapping):
        raise TypeError("Qwen tokenizer output must be a mapping.")
    input_ids = encoded.get("input_ids")
    if isinstance(input_ids, Tensor):
        if input_ids.ndim != 1:
            raise TypeError("Qwen tokenizer input_ids must be rank 1 for one text.")
        count = input_ids.numel()
    elif isinstance(input_ids, Sequence) and not isinstance(
        input_ids, (str, bytes, bytearray)
    ):
        count = len(input_ids)
    else:
        raise TypeError("Qwen tokenizer input_ids must be a sequence.")
    if count <= 0:
        raise ValueError("Qwen tokenizer input_ids must not be empty.")
    return count


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


def _accepts_keyword(method: Any, name: str) -> bool:
    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _commit(name: str, value: object) -> str:
    revision = non_empty_string(value, name)
    if len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision
    ):
        raise ValueError(f"{name} must be a lowercase 40-character commit hash.")
    return revision


__all__ = [
    "DECODER_PROTOCOL",
    "DEFAULT_MODEL",
    "PROMPT_PROTOCOL",
    "QwenTranslationLoadOptions",
    "QwenTranslationSpec",
    "QwenTranslationTokenEstimator",
    "QwenTranslator",
    "QwenTranslatorFactory",
    "translation_prompt",
]
