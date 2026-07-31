from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

from torch import nn

_QWEN3_MODELING_NAMES = {
    "Qwen3Attention",
    "Qwen3DecoderLayer",
    "Qwen3ForCausalLM",
    "Qwen3MLP",
    "Qwen3Model",
    "Qwen3PreTrainedModel",
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "rotate_half",
}

__all__ = [
    "build_qwen3_attention",
    "build_qwen3_decoder_layer",
    "build_qwen3_mlp",
    "build_qwen3_model",
    "build_qwen3_rms_norm",
    "build_qwen3_rotary_embedding",
    "make_qwen3_config",
    "require_qwen3_class",
]


def __getattr__(name: str) -> Any:
    if name == "Qwen3Config":
        return _load_qwen3_config_class()
    if name in _QWEN3_MODELING_NAMES:
        return require_qwen3_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_qwen3_config(**kwargs: Any) -> Any:
    config_cls = _load_qwen3_config_class()
    return config_cls(**kwargs)


def build_qwen3_rms_norm(hidden_size: int, *, eps: float = 1e-6) -> nn.Module:
    cls = require_qwen3_class("Qwen3RMSNorm")
    return cls(hidden_size, eps=eps)


def build_qwen3_mlp(
    hidden_size: int,
    intermediate_size: int,
    *,
    hidden_act: str = "silu",
    **config_overrides: Any,
) -> nn.Module:
    cls = require_qwen3_class("Qwen3MLP")
    config = make_qwen3_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        **config_overrides,
    )
    return cls(config)


def build_qwen3_rotary_embedding(
    head_dim: int,
    *,
    max_position_embeddings: int | None = None,
    rope_theta: float = 10000.0,
    rope_parameters: Mapping[str, Any] | None = None,
    device: Any | None = None,
    **config_overrides: Any,
) -> nn.Module:
    cls = require_qwen3_class("Qwen3RotaryEmbedding")
    config_kwargs = _merge_config_kwargs(
        config_overrides,
        head_dim=head_dim,
        rope_parameters=_default_rope_parameters(rope_theta, rope_parameters),
    )
    if max_position_embeddings is not None:
        config_kwargs["max_position_embeddings"] = max_position_embeddings
    config = make_qwen3_config(**config_kwargs)
    return cls(config, device=device)


def build_qwen3_attention(
    hidden_size: int,
    *,
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    layer_idx: int = 0,
    layer_type: str = "full_attention",
    layer_types: Sequence[str] | None = None,
    attention_bias: bool = False,
    attention_dropout: float = 0.0,
    rms_norm_eps: float = 1e-6,
    **config_overrides: Any,
) -> nn.Module:
    cls = require_qwen3_class("Qwen3Attention")
    config_kwargs = _merge_config_kwargs(
        config_overrides,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=max(layer_idx + 1, 1),
        layer_types=_layer_types_for(layer_idx, layer_type, layer_types),
        attention_bias=attention_bias,
        attention_dropout=attention_dropout,
        rms_norm_eps=rms_norm_eps,
    )
    if head_dim is not None:
        config_kwargs["head_dim"] = head_dim
    config = make_qwen3_config(**config_kwargs)
    return cls(config, layer_idx=layer_idx)


def build_qwen3_decoder_layer(
    hidden_size: int,
    intermediate_size: int,
    *,
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    layer_idx: int = 0,
    layer_type: str = "full_attention",
    layer_types: Sequence[str] | None = None,
    hidden_act: str = "silu",
    attention_bias: bool = False,
    attention_dropout: float = 0.0,
    rms_norm_eps: float = 1e-6,
    **config_overrides: Any,
) -> nn.Module:
    cls = require_qwen3_class("Qwen3DecoderLayer")
    config_kwargs = _merge_config_kwargs(
        config_overrides,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=max(layer_idx + 1, 1),
        layer_types=_layer_types_for(layer_idx, layer_type, layer_types),
        hidden_act=hidden_act,
        attention_bias=attention_bias,
        attention_dropout=attention_dropout,
        rms_norm_eps=rms_norm_eps,
    )
    if head_dim is not None:
        config_kwargs["head_dim"] = head_dim
    config = make_qwen3_config(**config_kwargs)
    return cls(config, layer_idx=layer_idx)


def build_qwen3_model(
    hidden_size: int,
    intermediate_size: int,
    *,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    vocab_size: int = 151936,
    hidden_act: str = "silu",
    max_position_embeddings: int | None = None,
    rope_theta: float = 10000.0,
    rope_parameters: Mapping[str, Any] | None = None,
    use_cache: bool = False,
    **config_overrides: Any,
) -> nn.Module:
    cls = require_qwen3_class("Qwen3Model")
    config_kwargs = _merge_config_kwargs(
        config_overrides,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        hidden_act=hidden_act,
        rope_parameters=_default_rope_parameters(rope_theta, rope_parameters),
        use_cache=use_cache,
    )
    if head_dim is not None:
        config_kwargs["head_dim"] = head_dim
    if max_position_embeddings is not None:
        config_kwargs["max_position_embeddings"] = max_position_embeddings
    config = make_qwen3_config(**config_kwargs)
    return cls(config)


def require_qwen3_class(name: str) -> Any:
    if name == "Qwen3Config":
        return _load_qwen3_config_class()
    if name not in _QWEN3_MODELING_NAMES:
        supported = ", ".join(sorted([*_QWEN3_MODELING_NAMES, "Qwen3Config"]))
        raise ValueError(f"Unknown Qwen3 class {name!r}. Supported names: {supported}.")
    module = _load_qwen3_modeling_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(
            "Installed transformers does not expose "
            f"{name}. Install anytrain[module], which requires transformers>=4.51.0."
        ) from exc


def _load_qwen3_config_class() -> type[Any]:
    try:
        from transformers import Qwen3Config
    except ImportError as exc:
        raise _missing_transformers_error() from exc
    return Qwen3Config


def _load_qwen3_modeling_module() -> ModuleType:
    try:
        from transformers.models.qwen3 import modeling_qwen3
    except ImportError as exc:
        raise _missing_transformers_error() from exc
    return modeling_qwen3


def _default_rope_parameters(
    rope_theta: float,
    rope_parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if rope_parameters is not None:
        return dict(rope_parameters)
    return {"rope_type": "default", "rope_theta": rope_theta}


def _layer_types_for(
    layer_idx: int,
    layer_type: str,
    layer_types: Sequence[str] | None,
) -> list[str]:
    if layer_types is not None:
        return list(layer_types)
    return ["full_attention"] * layer_idx + [layer_type]


def _merge_config_kwargs(config_overrides: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    merged = dict(kwargs)
    merged.update(config_overrides)
    return merged


def _missing_transformers_error() -> ImportError:
    return ImportError(
        "Qwen3 modules reuse Hugging Face transformers>=4.51.0. Install the module "
        "extra with python -m pip install 'anytrain[module]'."
    )
