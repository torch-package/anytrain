from __future__ import annotations

import math
from dataclasses import dataclass
from enum import auto
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anytrain._buffer import register_buffer
from anytrain._compat import StrEnum


class DiTConditionType(StrEnum):
    NONE = auto()
    FRAME_FILM = auto()
    FILM = auto()
    CROSS_ATTN = auto()


class DiTAttentionBackend(StrEnum):
    AUTO = auto()
    EAGER = auto()
    SDPA = auto()


@dataclass(eq=False)
class AttentionKV:
    key: Tensor
    value: Tensor


@dataclass(eq=False)
class DiTConditionState:
    condition_type: DiTConditionType
    film: Tensor | None = None
    cross_kv: tuple[AttentionKV, ...] | None = None
    condition_mask: Tensor | None = None


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    raise RuntimeError("a positive hidden dimension must have an attention head divisor")


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when configured.")


def _position(
    length: int,
    hidden_dim: int,
    reference: Tensor,
    frequency: Tensor,
) -> Tensor:
    angle = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
    embedding = torch.cat(
        ((angle * frequency).cos(), (angle * frequency).sin()),
        dim=-1,
    )
    if hidden_dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding.to(dtype=reference.dtype)


def _expand_time(t: Tensor, batch_size: int, *, validate: bool = True) -> Tensor:
    if t.ndim == 0:
        return t.expand(batch_size)
    if t.shape == (1,) and batch_size != 1:
        return t.expand(batch_size)
    if validate and t.shape != (batch_size,):
        raise ValueError("time must have shape [batch].")
    return t


def _mask(
    reference: Tensor,
    mask: Tensor | None,
    name: str,
    *,
    validate: bool = True,
) -> Tensor:
    if mask is None:
        return torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    if validate and mask.shape != reference.shape[:2]:
        raise ValueError(f"{name} must align with the sequence batch and length.")
    if validate and mask.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean.")
    if validate and mask.device != reference.device:
        raise ValueError(f"{name} and sequence tensor must use the same device.")
    return mask


def _require_valid_row(mask: Tensor, name: str) -> None:
    if mask.size(0) < 1 or not bool(mask.any(dim=1).all()):
        raise ValueError(f"each {name} row must contain at least one valid item.")


def _require_sequence(value: Tensor, dim: int, name: str) -> None:
    if value.dim() != 3 or value.size(-1) != dim:
        raise ValueError(f"{name} must have shape [batch, length, {name}_dim].")


def _require_vector(value: Tensor, dim: int, batch_size: int, name: str) -> None:
    if value.dim() != 2 or value.shape != (batch_size, dim):
        raise ValueError(f"{name} must have shape [batch, {name}_dim].")


def _condition_type(value: DiTConditionType | str) -> DiTConditionType:
    if isinstance(value, DiTConditionType):
        return value
    return DiTConditionType(value)


def _attention_backend(value: DiTAttentionBackend | str) -> DiTAttentionBackend:
    if isinstance(value, DiTAttentionBackend):
        return value
    return DiTAttentionBackend(value)


class TimeEmbedding(nn.Module):
    frequency: Tensor

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        half = hidden_dim // 2
        register_buffer(
            self,
            "frequency",
            torch.exp(
                -math.log(10_000)
                * torch.arange(half, dtype=torch.float32)
                / max(half - 1, 1)
            ),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        angle = time.float()[:, None] * self.frequency[None]
        embedding = torch.cat((angle.cos(), angle.sin()), dim=-1)
        if self.hidden_dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        projection = cast(nn.Linear, self.projection[0])
        return self.projection(embedding.to(dtype=projection.weight.dtype))


class SequenceAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        backend: DiTAttentionBackend,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.backend = backend
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def project_kv(self, source: Tensor) -> AttentionKV:
        return AttentionKV(
            key=self._split(self.key(source)),
            value=self._split(self.value(source)),
        )

    def forward(
        self,
        query: Tensor,
        *,
        key_value: Tensor | None = None,
        kv_cache: AttentionKV | None = None,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        if (key_value is None) == (kv_cache is None):
            raise ValueError("exactly one of key_value or kv_cache must be provided.")
        q = self._split(self.query(query))
        kv = self.project_kv(key_value) if key_value is not None else kv_cache
        if kv is None:
            raise RuntimeError("attention key/value cache was not constructed.")
        attended = self._attention(q, kv.key, kv.value, key_mask=key_mask)
        attended = attended.transpose(1, 2).contiguous().view(query.shape)
        return self.output(attended)

    def _split(self, value: Tensor) -> Tensor:
        return value.view(value.size(0), value.size(1), self.heads, self.head_dim).transpose(1, 2)

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_mask: Tensor | None,
    ) -> Tensor:
        backend = self.backend
        if backend is DiTAttentionBackend.AUTO:
            backend = (
                DiTAttentionBackend.SDPA
                if hasattr(F, "scaled_dot_product_attention")
                else DiTAttentionBackend.EAGER
            )
        if backend is DiTAttentionBackend.SDPA:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None if key_mask is None else key_mask[:, None, None, :],
                dropout_p=0.0,
                is_causal=False,
            )
        if backend is DiTAttentionBackend.EAGER:
            score = query @ key.transpose(-2, -1) * (self.head_dim**-0.5)
            if key_mask is not None:
                score = score.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            return score.softmax(dim=-1) @ value
        raise ValueError(f"unsupported attention backend: {backend}")


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_ratio: int,
        *,
        cross_attention: bool,
        attention_backend: DiTAttentionBackend,
    ) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.attention_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = SequenceAttention(hidden_dim, heads, backend=attention_backend)
        if cross_attention:
            self.context_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.context_attention = SequenceAttention(hidden_dim, heads, backend=attention_backend)
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
        )
        chunks = 9 if cross_attention else 6
        self.film = nn.Linear(hidden_dim, hidden_dim * chunks)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def project_context(self, context: Tensor) -> AttentionKV:
        if not self.cross_attention:
            raise RuntimeError("cross-attention is not configured.")
        return self.context_attention.project_kv(context)

    def forward(
        self,
        hidden: Tensor,
        film: Tensor,
        mask: Tensor,
        *,
        context_kv: AttentionKV | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        chunks = self.film(film).chunk(9 if self.cross_attention else 6, dim=-1)
        attention_shift, attention_scale, attention_gate = chunks[:3]
        if self.cross_attention:
            context_shift, context_scale, context_gate = chunks[3:6]
            ffn_shift, ffn_scale, ffn_gate = chunks[6:9]
        else:
            ffn_shift, ffn_scale, ffn_gate = chunks[3:6]

        normalized = self.attention_norm(hidden)
        normalized = normalized * (1 + attention_scale) + attention_shift
        attended = self.attention(normalized, key_value=normalized, key_mask=mask)
        hidden = hidden + attention_gate * attended
        hidden = hidden.masked_fill(~mask[..., None], 0)

        if self.cross_attention:
            if context_kv is None or context_mask is None:
                raise ValueError("cross-attention requires condition_state from prepare_condition.")
            normalized = self.context_norm(hidden)
            normalized = normalized * (1 + context_scale) + context_shift
            attended = self.context_attention(
                normalized,
                kv_cache=context_kv,
                key_mask=context_mask,
            )
            hidden = hidden + context_gate * attended
            hidden = hidden.masked_fill(~mask[..., None], 0)

        normalized = self.ffn_norm(hidden)
        normalized = normalized * (1 + ffn_scale) + ffn_shift
        hidden = hidden + ffn_gate * self.ffn(normalized)
        return hidden.masked_fill(~mask[..., None], 0)


class DiT(nn.Module):
    position_embedding: Tensor
    position_frequency: Tensor

    """Sequence DiT with one explicit condition mode and reusable condition state."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int | None = None,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
        condition_dim: int | None = None,
        condition_type: DiTConditionType | str = DiTConditionType.NONE,
        attention_backend: DiTAttentionBackend | str = DiTAttentionBackend.AUTO,
        feature_dim: int | None = None,
        feature_layer: int | None = None,
    ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        condition_type = _condition_type(condition_type)
        attention_backend = _attention_backend(attention_backend)
        _positive("input_dim", input_dim)
        _positive("output_dim", output_dim)
        _positive("hidden_dim", hidden_dim)
        _positive("layers", layers)
        _positive("heads", heads)
        _positive("ffn_ratio", ffn_ratio)
        _positive_optional("condition_dim", condition_dim)
        if condition_type is DiTConditionType.NONE:
            if condition_dim is not None:
                raise ValueError("condition_dim requires a non-none condition_type.")
        elif condition_dim is None:
            raise ValueError("condition_dim is required when condition_type is not none.")
        if feature_dim is None:
            if feature_layer is not None:
                raise ValueError("feature_layer requires feature_dim.")
        else:
            feature_layer = (layers + 1) // 2 if feature_layer is None else feature_layer
            if feature_dim <= 0 or not 1 <= feature_layer <= layers:
                raise ValueError("feature_dim must be positive and feature_layer must exist.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = input_dim
        self.condition_dim = condition_dim
        self.condition_type = condition_type
        self.attention_backend = attention_backend
        half = hidden_dim // 2
        register_buffer(
            self,
            "position_frequency",
            torch.exp(
                -math.log(10_000)
                * torch.arange(half, dtype=torch.float32)
                / max(half - 1, 1)
            ),
            persistent=False,
        )
        register_buffer(
            self,
            "position_embedding",
            torch.empty(0, hidden_dim),
            persistent=False,
        )
        self.input = nn.Linear(input_dim, hidden_dim)
        self.time = TimeEmbedding(hidden_dim)
        self.condition = None if condition_dim is None else nn.Linear(condition_dim, hidden_dim)
        attention_heads = _heads(hidden_dim, heads)
        self.blocks = nn.ModuleList(
            DiTBlock(
                hidden_dim,
                attention_heads,
                ffn_ratio,
                cross_attention=condition_type is DiTConditionType.CROSS_ATTN,
                attention_backend=attention_backend,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.feature_projection = None if feature_dim is None else nn.Linear(hidden_dim, feature_dim)
        self.feature_layer = feature_layer

    def prepare_condition(
        self,
        condition: Tensor | None = None,
        *,
        condition_mask: Tensor | None = None,
        validate: bool = True,
    ) -> DiTConditionState:
        if self.condition_type is DiTConditionType.NONE:
            if validate and (condition is not None or condition_mask is not None):
                raise ValueError("condition_type none does not accept condition inputs.")
            return DiTConditionState(condition_type=self.condition_type)
        if condition is None:
            raise ValueError("condition is required for this DiT condition_type.")
        if self.condition is None or self.condition_dim is None:
            raise RuntimeError("condition projection is not configured.")

        if self.condition_type is DiTConditionType.FILM:
            if validate:
                _require_vector(condition, self.condition_dim, condition.size(0), "condition")
            if validate and condition_mask is not None:
                raise ValueError("condition_mask is only valid for cross_attn conditions.")
            return DiTConditionState(
                condition_type=self.condition_type,
                film=self.condition(condition)[:, None],
            )

        if self.condition_type is DiTConditionType.FRAME_FILM:
            if validate:
                _require_sequence(condition, self.condition_dim, "condition")
            if validate and condition_mask is not None:
                raise ValueError("condition_mask is only valid for cross_attn conditions.")
            return DiTConditionState(
                condition_type=self.condition_type,
                film=self.condition(condition),
            )

        if self.condition_type is DiTConditionType.CROSS_ATTN:
            if validate:
                _require_sequence(condition, self.condition_dim, "condition")
            valid_mask = _mask(condition, condition_mask, "condition_mask", validate=validate)
            if validate:
                _require_valid_row(valid_mask, "condition_mask")
            context = self.condition(condition).masked_fill(~valid_mask[..., None], 0)
            return DiTConditionState(
                condition_type=self.condition_type,
                cross_kv=tuple(block.project_context(context) for block in self.blocks),
                condition_mask=valid_mask,
            )
        raise ValueError(f"unsupported condition_type: {self.condition_type}")

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor | None = None,
        condition_mask: Tensor | None = None,
        condition_state: DiTConditionState | None = None,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        output, _ = self._forward(
            x_t,
            t,
            condition=condition,
            condition_mask=condition_mask,
            condition_state=condition_state,
            mask=mask,
            validate=validate,
        )
        return output

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor | None = None,
        condition_mask: Tensor | None = None,
        condition_state: DiTConditionState | None = None,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if self.feature_projection is None:
            raise RuntimeError("feature projection is not configured.")
        output, representation = self._forward(
            x_t,
            t,
            condition=condition,
            condition_mask=condition_mask,
            condition_state=condition_state,
            mask=mask,
            validate=validate,
        )
        return output, self.feature_projection(representation)

    def _forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor | None,
        condition_mask: Tensor | None,
        condition_state: DiTConditionState | None,
        mask: Tensor | None,
        validate: bool,
    ) -> tuple[Tensor, Tensor]:
        if condition_state is not None and (condition is not None or condition_mask is not None):
            raise ValueError("condition_state cannot be combined with raw condition inputs.")
        if condition_state is None:
            condition_state = self.prepare_condition(
                condition,
                condition_mask=condition_mask,
                validate=validate,
            )
        if validate:
            self._validate_condition_state(condition_state)

        if validate and (x_t.dim() != 3 or x_t.size(-1) != self.input_dim):
            raise ValueError("x_t must have shape [batch, frame, input_dim].")
        t = _expand_time(t, x_t.size(0), validate=validate)
        frame_mask = _mask(x_t, mask, "mask", validate=validate)
        if validate:
            _require_valid_row(frame_mask, "mask")
        if validate:
            self._validate_condition_batch(condition_state, x_t)

        hidden = self.input(x_t)
        hidden = hidden + self._position(hidden)[None]
        hidden = hidden.masked_fill(~frame_mask[..., None], 0)
        film = self.time(t)[:, None]
        if condition_state.film is not None:
            film = film + condition_state.film.to(device=hidden.device, dtype=hidden.dtype)

        representation = hidden
        cross_kv = condition_state.cross_kv
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                film,
                frame_mask,
                context_kv=None if cross_kv is None else cross_kv[index],
                context_mask=condition_state.condition_mask,
            )
            if index + 1 == self.feature_layer:
                representation = hidden
        output = self.output(self.output_norm(hidden))
        output = output.masked_fill(~frame_mask[..., None], 0)
        return output, representation

    def _validate_condition_state(self, state: DiTConditionState) -> None:
        if state.condition_type is not self.condition_type:
            raise ValueError("condition_state condition_type does not match this DiT.")
        if self.condition_type is DiTConditionType.NONE:
            if state.film is not None or state.cross_kv is not None or state.condition_mask is not None:
                raise ValueError("none condition_state must not contain cached tensors.")
        elif self.condition_type in {DiTConditionType.FILM, DiTConditionType.FRAME_FILM}:
            if state.film is None or state.cross_kv is not None or state.condition_mask is not None:
                raise ValueError("FiLM condition_state must contain only film.")
        elif self.condition_type is DiTConditionType.CROSS_ATTN:
            if state.cross_kv is None or state.condition_mask is None or state.film is not None:
                raise ValueError("cross_attn condition_state must contain cross_kv and condition_mask.")
            if len(state.cross_kv) != len(self.blocks):
                raise ValueError("cross_attn condition_state must provide one KV cache per layer.")

    def _validate_condition_batch(self, state: DiTConditionState, x_t: Tensor) -> None:
        if state.film is not None:
            if state.film.size(0) != x_t.size(0) or state.film.size(-1) != self.hidden_dim:
                raise ValueError("condition_state film must match batch size and hidden_dim.")
            if state.condition_type is DiTConditionType.FRAME_FILM and state.film.shape[:2] != x_t.shape[:2]:
                raise ValueError("frame FiLM condition must align with x_t on [batch, frame].")
            if state.condition_type is DiTConditionType.FILM and state.film.size(1) != 1:
                raise ValueError("FiLM condition_state must broadcast from shape [batch, 1, hidden].")
        if state.condition_mask is not None:
            if state.condition_mask.size(0) != x_t.size(0):
                raise ValueError("condition_state mask batch size must match x_t.")
            if state.condition_mask.device != x_t.device:
                raise ValueError("condition_state mask and x_t must use the same device.")
        if state.cross_kv is not None:
            for kv in state.cross_kv:
                if kv.key.size(0) != x_t.size(0) or kv.value.size(0) != x_t.size(0):
                    raise ValueError("condition_state KV batch size must match x_t.")
                if kv.key.device != x_t.device or kv.value.device != x_t.device:
                    raise ValueError("condition_state KV tensors and x_t must use the same device.")

    def _position(self, reference: Tensor) -> Tensor:
        length = reference.size(1)
        cached = self.position_embedding
        if cached.size(0) < length or cached.dtype != reference.dtype:
            self.position_embedding = _position(
                length,
                reference.size(2),
                reference,
                self.position_frequency,
            )
        return self.position_embedding[:length]


__all__ = [
    "AttentionKV",
    "DiT",
    "DiTAttentionBackend",
    "DiTConditionState",
    "DiTConditionType",
]
