from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ._deps import load_transformers_auto_model_class, load_transformers_auto_processor_class
from ._types import AttentionModel, Processor


def load_pretrained_pair(
    model: str | Path,
    *,
    cache_dir: str | Path | None,
    codec_model: str | Path,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
    load_kwargs: Mapping[str, object],
) -> tuple[Any, Processor]:
    try:
        auto_model_cls = load_transformers_auto_model_class()
        auto_processor_cls = load_transformers_auto_processor_class()
    except ImportError as exc:
        raise ImportError(
            "`anytrain.tts.moss` requires `transformers` to load remote-code "
            "MOSS-TTS v1.5 checkpoints and processors. Install Moss TTS "
            'dependencies with `python -m pip install "torchaudio>=2.0" transformers`.'
        ) from exc

    try:
        loaded = auto_model_cls.from_pretrained(
            str(model),
            **pretrained_kwargs(
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                trust_remote_code=trust_remote_code,
                load_kwargs=load_kwargs,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Hugging Face MOSS-TTS v1.5 checkpoint {model!s}"
        ) from exc

    processor = load_pretrained_processor(
        model,
        cache_dir=cache_dir,
        codec_model=codec_model,
        local_files_only=local_files_only,
        revision=revision,
        trust_remote_code=trust_remote_code,
        auto_processor_cls=auto_processor_cls,
    )

    set_attention_implementation(loaded, load_kwargs.get("attn_implementation", "sdpa"))
    return loaded, cast(Processor, processor)


def load_pretrained_processor(
    model: str | Path,
    *,
    cache_dir: str | Path | None,
    codec_model: str | Path,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
    auto_processor_cls: Any | None = None,
) -> Processor:
    """Load only the processor needed to construct MOSS prompts."""

    if auto_processor_cls is None:
        try:
            auto_processor_cls = load_transformers_auto_processor_class()
        except ImportError as exc:
            raise ImportError(
                "`anytrain.tts.moss` requires `transformers` to load the "
                "MOSS-TTS processor."
            ) from exc
    try:
        processor = auto_processor_cls.from_pretrained(
            pretrained_source(
                model,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            ),
            **processor_pretrained_kwargs(
                cache_dir=cache_dir,
                codec_model=codec_model,
                local_files_only=local_files_only,
                revision=revision,
                trust_remote_code=trust_remote_code,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Hugging Face MOSS-TTS v1.5 processor {model!s}"
        ) from exc
    return cast(Processor, processor)


def pretrained_kwargs(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
    load_kwargs: Mapping[str, object],
) -> dict[str, object]:
    kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    kwargs.update(load_kwargs)
    return kwargs


def processor_pretrained_kwargs(
    *,
    cache_dir: str | Path | None,
    codec_model: object,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    kwargs["codec_path"] = pretrained_source(
        str(codec_model),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=None,
    )
    return kwargs


def pretrained_source(
    source: str | Path,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
) -> str:
    raw = str(source)
    if Path(raw).expanduser().exists() or not local_files_only:
        return raw
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Resolving a cached MOSS-TTS repository requires `huggingface-hub`. "
            'Install it with `python -m pip install "huggingface-hub>=0.23"`.'
        ) from exc
    return snapshot_download(
        repo_id=raw,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=True,
        revision=revision,
    )


def set_attention_implementation(model: object, value: object) -> None:
    attention = attention_model(model)
    if attention is not None:
        attention._set_attention_implementation(str(value))


def attention_model(model: object) -> AttentionModel | None:
    setter = getattr(model, "_set_attention_implementation", None)
    return cast(AttentionModel, model) if callable(setter) else None
