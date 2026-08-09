from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from anytrain._compat import strict_zip
from anytrain.module.qwen.qwen3 import build_qwen3_model
from anytrain.utils import is_signed_integer_dtype
from anytrain.utils import top_p_filter as _top_p_filter


class QwenMTPCodebookPredictor(nn.Module):
    """Qwen-style temporal AR plus intra-frame MTP codebook predictor.

    The temporal Qwen backbone models frame-to-frame dependency through the
    first codebook. A smaller MTP Qwen predictor fills the remaining codebooks
    inside each frame from the temporal frame state.
    """

    def __init__(
        self,
        condition_dim: int,
        codebooks: int,
        codebook_size: int | Sequence[int],
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
        mtp_layers: int = 2,
        mtp_heads: int = 4,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or codebooks <= 0:
            raise ValueError("condition_dim and codebooks must be positive.")
        if layers <= 0 or heads <= 0 or ffn_ratio <= 0 or mtp_layers <= 0 or mtp_heads <= 0:
            raise ValueError("decoder depth, heads, and FFN ratio must be positive.")
        hidden_dim = condition_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("decoder hidden dimension must be positive.")
        sizes = (codebook_size,) * codebooks if isinstance(codebook_size, int) else tuple(codebook_size)
        if len(sizes) != codebooks or any(size <= 0 for size in sizes):
            raise ValueError("codebook_size must provide one positive size per codebook.")
        if codebook_embeddings is not None:
            embedding_values = _used_codebook_embeddings(codebook_embeddings, sizes)
            embedding_dim = embedding_values[0].size(-1)
        else:
            embedding_values = None
            embedding_dim = hidden_dim

        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.codebooks = codebooks
        self.codebook_sizes = sizes
        self.embedding_dim = embedding_dim
        # MTP predicts codebooks 1..Q-1 and only consumes preceding
        # embeddings. For Q=1, embedding zero remains needed by temporal AR.
        used_sizes = _embedding_sizes(sizes)
        self.codebook_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in used_sizes
        )
        if codebook_embeddings is None:
            for module in self.codebook_embeddings:
                embedding = cast(nn.Embedding, cast(object, module))
                nn.init.normal_(embedding.weight, std=embedding_dim**-0.5)
        else:
            with torch.no_grad():
                if embedding_values is None:
                    raise RuntimeError("MTP embedding values are unexpectedly unavailable.")
                for index, module in enumerate(self.codebook_embeddings):
                    embedding = cast(nn.Embedding, cast(object, module))
                    embedding.weight.copy_(embedding_values[index])

        self.embedding_projections = nn.ModuleList(
            nn.Identity()
            if embedding_dim == hidden_dim
            else nn.Linear(embedding_dim, hidden_dim)
            for _ in used_sizes
        )
        self.condition = nn.Identity() if condition_dim == hidden_dim else nn.Linear(condition_dim, hidden_dim)
        self.first_bos = nn.Parameter(torch.zeros(hidden_dim))
        self.codebook_bos = nn.Parameter(torch.zeros(codebooks - 1, hidden_dim))
        self.temporal = _qwen3_model(
            hidden_dim=hidden_dim,
            ffn_ratio=ffn_ratio,
            layers=layers,
            attention_heads=_heads(hidden_dim, heads),
        )
        self.temporal.embed_tokens.requires_grad_(False)
        self.mtp: nn.Module | None = None
        if codebooks > 1:
            self.mtp = _qwen3_model(
                hidden_dim=hidden_dim,
                ffn_ratio=ffn_ratio,
                layers=mtp_layers,
                attention_heads=_heads(hidden_dim, mtp_heads),
            )
            self.mtp.embed_tokens.requires_grad_(False)
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in sizes)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load checkpoints written before unused MTP parameters were removed."""
        # Older versions registered one embedding/projection for every
        # codebook. The last pair was never read, so discard it during load.
        if self.codebooks > 1:
            dead_index = self.codebooks - 1
            for name in ("codebook_embeddings", "embedding_projections"):
                dead_prefix = f"{prefix}{name}.{dead_index}."
                for key in tuple(state_dict):
                    if key.startswith(dead_prefix):
                        del state_dict[key]
        else:
            legacy_mtp_prefix = f"{prefix}mtp."
            for key in tuple(state_dict):
                if key.startswith(legacy_mtp_prefix):
                    del state_dict[key]

        bos_key = f"{prefix}codebook_bos"
        if bos_key in state_dict:
            old_bos = state_dict[bos_key]
            if old_bos.shape == (self.codebooks, self.hidden_dim):
                # The old row zero was unused; the remaining rows map directly to
                # the compact residual-codebook BOS parameter.
                state_dict[bos_key] = old_bos[1:].clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.dim() != 3 or condition.size(-1) != self.condition_dim:
            raise ValueError("condition must have shape [batch, frame, condition_dim].")

    def _embedding(self, codebook: int, codes: Tensor) -> Tensor:
        embedding = cast(nn.Embedding, cast(object, self.codebook_embeddings[codebook]))
        projection = cast(nn.Module, cast(object, self.embedding_projections[codebook]))
        value = embedding(codes.to(dtype=torch.long))
        return projection(value)

    def forward(
        self,
        condition: Tensor,
        target_codes: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, ...]:
        """Return one teacher-forced [B, F, vocab_q] logits tensor per codebook.

        Set ``validate=False`` only for batches validated by the owning
        datamodule; this avoids repeated shape, dtype, and range checks.
        """
        if validate:
            self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask, validate=validate)
        if target_codes is not None:
            if validate:
                _validate_targets(
                    target_codes,
                    condition,
                    self.codebooks,
                    self.codebook_sizes,
                    frame_mask,
                )
            previous_first = _shift_right(target_codes[..., 0], frame_mask)
        else:
            previous_first = torch.zeros(condition.shape[:2], dtype=torch.long, device=condition.device)

        condition_hidden = self.condition(condition)
        temporal_input = condition_hidden + self.first_bos + self._embedding(0, previous_first)
        temporal_hidden = self.temporal(
            inputs_embeds=temporal_input,
            attention_mask=frame_mask.to(dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        logits = [cast(nn.Linear, cast(object, self.heads[0]))(temporal_hidden)]
        if self.codebooks == 1:
            return (logits[0].masked_fill(~frame_mask[..., None], 0),)

        packed_hidden = temporal_hidden.flatten(0, 1)[frame_mask.flatten()]
        if target_codes is not None:
            packed_targets = target_codes.flatten(0, 1)[frame_mask.flatten()]
        else:
            packed_targets = None
        logits.extend(self._mtp_logits(packed_hidden, packed_targets, frame_mask))
        return tuple(value.masked_fill(~frame_mask[..., None], 0) for value in logits)

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask)
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1].")

        condition_hidden = self.condition(condition)
        previous_first = torch.zeros(condition.size(0), dtype=torch.long, device=condition.device)
        generated = condition.new_zeros((*condition.shape[:2], self.codebooks), dtype=torch.long)
        past_key_values = None
        for frame in range(condition.size(1)):
            decoder_input = condition_hidden[:, frame] + self.first_bos + self._embedding(0, previous_first)
            state_output = self.temporal(
                inputs_embeds=decoder_input[:, None],
                attention_mask=frame_mask[:, : frame + 1].to(dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = state_output.past_key_values
            if past_key_values is None:
                raise RuntimeError("Qwen MTP temporal decoder did not return a generation cache.")
            frame_state = state_output.last_hidden_state[:, -1]
            first = _sample_logits(
                cast(nn.Linear, cast(object, self.heads[0]))(frame_state),
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            residual = self._sample_mtp(
                frame_state,
                first,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            frame_codes = torch.cat((first[:, None], residual), dim=-1)
            valid = frame_mask[:, frame]
            generated[:, frame] = frame_codes.masked_fill(~valid[:, None], 0)
            previous_first = first.masked_fill(~valid, 0)
        return generated

    def _mtp_logits(
        self,
        frame_state: Tensor,
        packed_targets: Tensor | None,
        frame_mask: Tensor,
    ) -> list[Tensor]:
        if self.mtp is None:
            raise RuntimeError("MTP logits require more than one codebook.")
        inputs = []
        for codebook in range(1, self.codebooks):
            if packed_targets is None:
                previous = torch.zeros(frame_state.size(0), dtype=torch.long, device=frame_state.device)
            else:
                previous = packed_targets[..., codebook - 1]
            inputs.append(
                frame_state
                + self.codebook_bos[codebook - 1]
                + self._embedding(codebook - 1, previous)
            )
        hidden = self.mtp(
            inputs_embeds=torch.stack(inputs, dim=1),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return [
            _scatter(
                cast(nn.Linear, cast(object, self.heads[codebook]))(hidden[:, codebook - 1]),
                frame_mask,
            )
            for codebook in range(1, self.codebooks)
        ]

    def _sample_mtp(
        self,
        frame_state: Tensor,
        first: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.codebooks == 1:
            return first.new_zeros((first.size(0), 0))
        if self.mtp is None:
            raise RuntimeError("MTP sampling requires more than one codebook.")
        output: list[Tensor] = []
        past_key_values = None
        previous = first
        for codebook in range(1, self.codebooks):
            decoder_input = (
                frame_state
                + self.codebook_bos[codebook - 1]
                + self._embedding(codebook - 1, previous)
            )
            state_output = self.mtp(
                inputs_embeds=decoder_input[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = state_output.past_key_values
            if past_key_values is None:
                raise RuntimeError("Qwen MTP predictor did not return a generation cache.")
            state = state_output.last_hidden_state[:, -1]
            previous = _sample_logits(
                cast(nn.Linear, cast(object, self.heads[codebook]))(state),
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            output.append(previous)
        return torch.stack(output, dim=-1)


def _qwen3_model(
    *,
    hidden_dim: int,
    ffn_ratio: int,
    layers: int,
    attention_heads: int,
) -> nn.Module:
    try:
        return build_qwen3_model(
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * ffn_ratio,
            num_layers=layers,
            num_attention_heads=attention_heads,
            num_key_value_heads=attention_heads,
            head_dim=hidden_dim // attention_heads,
            vocab_size=1,
            use_cache=True,
        )
    except ImportError as exc:
        raise ImportError(
            "QwenMTPCodebookPredictor requires transformers>=4.51.0 with Qwen3Model; "
            "install anytrain[module]."
        ) from exc


def _used_codebook_embeddings(values: Sequence[Tensor], sizes: Sequence[int]) -> tuple[Tensor, ...]:
    """Accept both the historical Q embeddings and the compact Q-1 form."""
    used_sizes = _embedding_sizes(sizes)
    if len(values) == len(sizes):
        _validate_embeddings(values, sizes)
        return tuple(values[: len(used_sizes)])
    if len(values) == len(used_sizes):
        _validate_embeddings(values, used_sizes)
        return tuple(values)
    raise ValueError(
        "codebook_embeddings must provide one tensor per codebook or one "
        "tensor per predicted residual codebook."
    )


def _embedding_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    # The temporal first-codebook input still needs embedding zero when Q=1.
    return tuple(sizes if len(sizes) == 1 else sizes[:-1])


def _validate_embeddings(values: Sequence[Tensor], sizes: Sequence[int]) -> None:
    if len(values) != len(sizes):
        raise ValueError("codebook_embeddings must provide one tensor per codebook.")
    if not values:
        return
    if any(not torch.is_floating_point(value) for value in values):
        raise TypeError("codebook_embeddings must be floating point.")
    if any(value.dim() != 2 for value in values):
        raise ValueError("each codebook embedding must have shape [size_q, dim].")
    if any(value.size(0) != size for value, size in strict_zip(values, sizes)):
        raise ValueError("codebook embeddings must match codebook sizes.")
    embedding_dim = values[0].size(-1)
    if any(value.size(-1) != embedding_dim for value in values):
        raise ValueError("all codebook embeddings must have the same dimension.")


def _validate_targets(
    target_codes: Tensor,
    condition: Tensor,
    codebooks: int,
    codebook_sizes: Sequence[int],
    frame_mask: Tensor,
) -> None:
    if target_codes.shape != (condition.size(0), condition.size(1), codebooks):
        raise ValueError("target_codes must have shape [B, F, codebooks].")
    if not is_signed_integer_dtype(target_codes.dtype):
        raise TypeError("target_codes must use a signed integer dtype.")
    packed_targets = target_codes.flatten(0, 1)[frame_mask.flatten()]
    limits = torch.tensor(codebook_sizes, device=packed_targets.device, dtype=torch.long)
    if bool(((packed_targets < 0) | (packed_targets >= limits)).any()):
        raise ValueError("target_codes contains an ID outside its codebook.")


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0 and (hidden_dim // heads) % 2 == 0:
            return heads
    raise RuntimeError("Qwen MTP predictor requires an even attention head dimension.")


def _frame_mask(condition: Tensor, mask: Tensor | None, *, validate: bool = True) -> Tensor:
    if mask is None:
        frame_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=condition.device)
    else:
        if validate:
            if mask.shape != condition.shape[:2]:
                raise ValueError("frame mask must align with condition.")
            if mask.dtype != torch.bool:
                raise TypeError("frame mask must be boolean.")
            if mask.device != condition.device:
                raise ValueError("frame mask and condition must use the same device.")
        frame_mask = mask
    if validate and (frame_mask.size(0) < 1 or not bool(frame_mask.any(dim=1).all())):
        raise ValueError("each condition row must contain a valid frame.")
    return frame_mask


def _shift_right(values: Tensor, mask: Tensor) -> Tensor:
    shifted = values.new_zeros(values.shape)
    shifted[:, 1:] = values[:, :-1].masked_fill(~mask[:, :-1], 0)
    return shifted.masked_fill(~mask, 0)


def _scatter(values: Tensor, mask: Tensor) -> Tensor:
    frame_indices = mask.flatten().nonzero().flatten()
    output = values.new_zeros((mask.numel(), *values.shape[1:]))
    output = output.index_copy(0, frame_indices, values)
    return output.unflatten(0, mask.shape)


def _sample_logits(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> Tensor:
    scaled = logits / temperature
    if top_p < 1.0:
        scaled = _top_p_filter(scaled, top_p)
    return torch.multinomial(scaled.softmax(dim=-1), 1, generator=generator)[:, 0]
