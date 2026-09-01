from __future__ import annotations

import re
from dataclasses import dataclass
from enum import auto
from typing import TypeAlias

import torch
from torch import Tensor

from anytrain._compat import StrEnum

IGNORE_INDEX = -100
_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)
_TAG_SEGMENT = re.compile(r"[a-z][a-z0-9_]*")


class Modality(StrEnum):
    TEXT = auto()
    AUDIO = auto()


@dataclass(eq=False, frozen=True, slots=True)
class Audio:
    """One materialized waveform with an explicit sample rate."""

    waveform: Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, Tensor):
            raise TypeError("audio waveform must be a Tensor.")
        if self.waveform.ndim != 2:
            raise ValueError("audio waveform must have shape [channels, time].")
        if self.waveform.shape[0] <= 0 or self.waveform.shape[1] <= 0:
            raise ValueError("audio waveform must not be empty.")
        if not torch.is_floating_point(self.waveform):
            raise TypeError("audio waveform must use a floating-point dtype.")
        if _cpu(self.waveform) and not bool(torch.isfinite(self.waveform).all()):
            raise ValueError("audio waveform must contain only finite values.")
        _positive_int(self.sample_rate, "audio sample_rate")

    @property
    def duration_seconds(self) -> float:
        return self.waveform.shape[1] / self.sample_rate


@dataclass(eq=False, frozen=True, slots=True)
class Content:
    """Semantic text and/or audio at the model-provider boundary."""

    text: str | None = None
    audio: Audio | None = None

    def __post_init__(self) -> None:
        if self.text is not None:
            _non_empty_string(self.text, "content text")
        if self.audio is not None and not isinstance(self.audio, Audio):
            raise TypeError("content audio must be Audio or None.")
        if self.text is None and self.audio is None:
            raise ValueError("content must contain text, audio, or both.")

    @property
    def modalities(self) -> frozenset[Modality]:
        values: set[Modality] = set()
        if self.text is not None:
            values.add(Modality.TEXT)
        if self.audio is not None:
            values.add(Modality.AUDIO)
        return frozenset(values)


@dataclass(frozen=True, slots=True)
class Route:
    """Stable task/trace lineage used for grouping and metric tags."""

    task: str
    trace: str = "direct"

    def __post_init__(self) -> None:
        _tag_segment(self.task, "route task")
        _tag_segment(self.trace, "route trace")

    @property
    def tag(self) -> str:
        return f"{self.task}/{self.trace}"


@dataclass(eq=False, frozen=True, slots=True)
class GenerationRequest:
    source: Content
    route: Route
    response_modalities: frozenset[Modality]
    target_language: str | None = None
    sample_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, Content):
            raise TypeError("generation source must be Content.")
        if not isinstance(self.route, Route):
            raise TypeError("generation route must be Route.")
        _modalities(self.response_modalities, "generation response_modalities")
        if self.target_language is not None:
            _non_empty_string(self.target_language, "generation target_language")
        if self.sample_id is not None:
            _non_empty_string(self.sample_id, "generation sample_id")


@dataclass(eq=False, frozen=True, slots=True)
class GenerationResponse:
    text: str | None = None
    audio: Audio | None = None

    def __post_init__(self) -> None:
        if self.text is not None:
            _non_empty_string(self.text, "generation response text")
        if self.audio is not None and not isinstance(self.audio, Audio):
            raise TypeError("generation response audio must be Audio or None.")
        if self.text is None and self.audio is None:
            raise ValueError("generation response must contain text, audio, or both.")

    @property
    def modalities(self) -> frozenset[Modality]:
        values: set[Modality] = set()
        if self.text is not None:
            values.add(Modality.TEXT)
        if self.audio is not None:
            values.add(Modality.AUDIO)
        return frozenset(values)


@dataclass(eq=False, frozen=True, slots=True)
class Sample:
    """One semantic request and its aligned reference response."""

    request: GenerationRequest
    target: Content

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationRequest):
            raise TypeError("sample request must be a GenerationRequest.")
        if not isinstance(self.target, Content):
            raise TypeError("sample target must be Content.")
        if not self.target.modalities <= self.request.response_modalities:
            raise ValueError(
                "sample target modalities must be requested response modalities."
            )


@dataclass(eq=False, frozen=True, slots=True)
class TokenAudioCondition:
    """Positions whose ``input_ids`` values are discrete source-audio tokens."""

    positions: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        _audio_positions(self.positions, self.mask)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TokenAudioCondition:
        return TokenAudioCondition(
            positions=self.positions.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> TokenAudioCondition:
        return TokenAudioCondition(
            positions=self.positions.pin_memory(),
            mask=self.mask.pin_memory(),
        )


@dataclass(eq=False, frozen=True, slots=True)
class EmbeddedAudioCondition:
    """Positions overlaid by provider-ready continuous source-audio embeddings."""

    positions: Tensor
    mask: Tensor
    embeddings: Tensor

    def __post_init__(self) -> None:
        _audio_positions(self.positions, self.mask)
        if not isinstance(self.embeddings, Tensor):
            raise TypeError("audio condition embeddings must be a Tensor.")
        if self.embeddings.ndim != 3:
            raise ValueError(
                "audio condition embeddings must have shape [batch, frames, hidden]."
            )
        if self.embeddings.shape[:2] != self.positions.shape:
            raise ValueError(
                "audio condition embeddings must align with positions on batch and frames."
            )
        if self.embeddings.shape[2] <= 0:
            raise ValueError("audio condition embedding hidden size must be positive.")
        if not torch.is_floating_point(self.embeddings):
            raise TypeError("audio condition embeddings must use a floating-point dtype.")
        if self.embeddings.device != self.positions.device:
            raise ValueError("audio condition tensors must share a device.")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> EmbeddedAudioCondition:
        return EmbeddedAudioCondition(
            positions=self.positions.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            embeddings=self.embeddings.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> EmbeddedAudioCondition:
        return EmbeddedAudioCondition(
            positions=self.positions.pin_memory(),
            mask=self.mask.pin_memory(),
            embeddings=self.embeddings.pin_memory(),
        )


AudioCondition: TypeAlias = TokenAudioCondition | EmbeddedAudioCondition


@dataclass(eq=False, frozen=True, slots=True)
class TokenInput:
    input_ids: Tensor
    attention_mask: Tensor
    audio: AudioCondition | None = None

    def __post_init__(self) -> None:
        _integer_tensor(self.input_ids, "token input_ids", ndim=2)
        _bool_tensor(
            self.attention_mask,
            "token attention_mask",
            shape=self.input_ids.shape,
        )
        if self.input_ids.device != self.attention_mask.device:
            raise ValueError("token input_ids and attention_mask must share a device.")
        if self.input_ids.shape[0] <= 0 or self.input_ids.shape[1] <= 0:
            raise ValueError("token input must have non-empty batch and sequence axes.")
        if _cpu(self.attention_mask) and not bool(self.attention_mask.any(dim=1).all()):
            raise ValueError("every token input row must contain at least one active token.")
        if self.audio is None:
            return
        if not isinstance(self.audio, (TokenAudioCondition, EmbeddedAudioCondition)):
            raise TypeError(
                "token audio must be TokenAudioCondition, EmbeddedAudioCondition, or None."
            )
        if self.audio.positions.shape[0] != self.input_ids.shape[0]:
            raise ValueError("token audio must align with the input batch size.")
        if self.audio.positions.device != self.input_ids.device:
            raise ValueError("token input and audio condition must share a device.")
        if _cpu(self.input_ids, self.audio.positions, self.audio.mask):
            active_positions = self.audio.positions.masked_select(self.audio.mask)
            if bool((active_positions < 0).any()) or bool(
                (active_positions >= self.input_ids.shape[1]).any()
            ):
                raise ValueError("active audio positions must index the token sequence.")
            rows = torch.arange(self.input_ids.shape[0], device=self.input_ids.device)
            rows = rows.unsqueeze(1).expand_as(self.audio.positions)
            if not bool(
                self.attention_mask[
                    rows.masked_select(self.audio.mask),
                    active_positions.to(dtype=torch.int64),
                ].all()
            ):
                raise ValueError("active audio positions must point to active input tokens.")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TokenInput:
        return TokenInput(
            input_ids=self.input_ids.to(device=device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(
                device=device,
                non_blocking=non_blocking,
            ),
            audio=(
                None
                if self.audio is None
                else self.audio.to(device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> TokenInput:
        return TokenInput(
            input_ids=self.input_ids.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            audio=None if self.audio is None else self.audio.pin_memory(),
        )


@dataclass(eq=False, frozen=True, slots=True)
class TokenTargets:
    labels: Tensor
    text_mask: Tensor | None = None
    audio_mask: Tensor | None = None

    def __post_init__(self) -> None:
        _integer_tensor(self.labels, "token labels", ndim=2)
        masks = tuple(mask for mask in (self.text_mask, self.audio_mask) if mask is not None)
        if not masks:
            raise ValueError("token targets require a text_mask, an audio_mask, or both.")
        for name, mask in (("text_mask", self.text_mask), ("audio_mask", self.audio_mask)):
            if mask is None:
                continue
            _bool_tensor(mask, f"token {name}", shape=self.labels.shape)
            if mask.device != self.labels.device:
                raise ValueError(f"token {name} and labels must share a device.")
            if _cpu(mask) and not bool(mask.any()):
                raise ValueError(f"token {name} must select at least one target.")
        if (
            self.text_mask is not None
            and self.audio_mask is not None
            and _cpu(self.text_mask, self.audio_mask)
            and bool((self.text_mask & self.audio_mask).any())
        ):
            raise ValueError("text and audio target masks must be disjoint.")
        if _cpu(self.labels, *masks):
            supervised = torch.zeros_like(self.labels, dtype=torch.bool)
            for mask in masks:
                supervised |= mask
            if not bool(supervised.any()):
                raise ValueError("token targets must supervise at least one token.")
            if bool(supervised[:, 0].any()):
                raise ValueError("causal token targets cannot supervise sequence position zero.")
            if not torch.equal(supervised, self.labels.ne(IGNORE_INDEX)):
                raise ValueError(
                    "text/audio target masks must exactly cover labels that are not IGNORE_INDEX."
                )

    @property
    def modalities(self) -> frozenset[Modality]:
        values: set[Modality] = set()
        if self.text_mask is not None:
            values.add(Modality.TEXT)
        if self.audio_mask is not None:
            values.add(Modality.AUDIO)
        return frozenset(values)

    def mask(self, modality: Modality) -> Tensor | None:
        if modality is Modality.TEXT:
            return self.text_mask
        if modality is Modality.AUDIO:
            return self.audio_mask
        raise TypeError("modality must be a Modality.")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TokenTargets:
        return TokenTargets(
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            text_mask=(
                None
                if self.text_mask is None
                else self.text_mask.to(device=device, non_blocking=non_blocking)
            ),
            audio_mask=(
                None
                if self.audio_mask is None
                else self.audio_mask.to(device=device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> TokenTargets:
        return TokenTargets(
            labels=self.labels.pin_memory(),
            text_mask=None if self.text_mask is None else self.text_mask.pin_memory(),
            audio_mask=None if self.audio_mask is None else self.audio_mask.pin_memory(),
        )


@dataclass(eq=False, frozen=True, slots=True)
class TokenBatch:
    input: TokenInput
    targets: TokenTargets
    prompt_lengths: Tensor
    routes: tuple[Route, ...]
    sample_ids: tuple[str, ...] | None = None
    audio_seconds: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input, TokenInput):
            raise TypeError("token batch input must be TokenInput.")
        if not isinstance(self.targets, TokenTargets):
            raise TypeError("token batch targets must be TokenTargets.")
        if self.targets.labels.shape != self.input.input_ids.shape:
            raise ValueError("token labels must align with input_ids.")
        if self.targets.labels.device != self.input.input_ids.device:
            raise ValueError("token inputs and targets must share a device.")
        _integer_tensor(self.prompt_lengths, "token prompt_lengths", ndim=1)
        if self.prompt_lengths.shape[0] != self.batch_size:
            raise ValueError("token prompt_lengths must have shape [batch].")
        if self.prompt_lengths.device != self.input.input_ids.device:
            raise ValueError("token prompt_lengths and inputs must share a device.")
        lengths = self.input.attention_mask.sum(dim=1)
        if _cpu(self.input.attention_mask, self.prompt_lengths):
            expected_mask = torch.arange(
                self.input.input_ids.shape[1],
                device=self.input.input_ids.device,
            ).unsqueeze(0) < lengths.unsqueeze(1)
            if not torch.equal(self.input.attention_mask, expected_mask):
                raise ValueError("token attention_mask must use contiguous right padding.")
            if bool((self.prompt_lengths < 1).any()) or bool(
                (self.prompt_lengths >= lengths).any()
            ):
                raise ValueError(
                    "token prompt_lengths must leave a non-empty response in every row."
                )
            target_mask = torch.zeros_like(self.input.attention_mask)
            if self.targets.text_mask is not None:
                target_mask |= self.targets.text_mask
            if self.targets.audio_mask is not None:
                target_mask |= self.targets.audio_mask
            if bool((target_mask & ~self.input.attention_mask).any()):
                raise ValueError("token targets cannot supervise padded input positions.")
            positions = torch.arange(
                self.input.input_ids.shape[1],
                device=self.input.input_ids.device,
            ).unsqueeze(0)
            before_response = positions < self.prompt_lengths.unsqueeze(1)
            if bool((target_mask & before_response).any()):
                raise ValueError("token targets must lie at or after the prompt boundary.")
        if self.input.audio is not None and _cpu(
            self.input.audio.positions,
            self.input.audio.mask,
            self.prompt_lengths,
        ):
            rows = torch.arange(self.batch_size, device=self.input.input_ids.device)
            rows = rows.unsqueeze(1).expand_as(self.input.audio.positions)
            active_rows = rows.masked_select(self.input.audio.mask)
            active_positions = self.input.audio.positions.masked_select(self.input.audio.mask)
            if not bool((active_positions < self.prompt_lengths[active_rows]).all()):
                raise ValueError("audio condition positions must lie inside the prompt.")
        if not isinstance(self.routes, tuple) or len(self.routes) != self.batch_size:
            raise ValueError("token routes must be a tuple aligned with batch rows.")
        if any(not isinstance(route, Route) for route in self.routes):
            raise TypeError("token routes must contain Route values.")
        if self.sample_ids is not None:
            if not isinstance(self.sample_ids, tuple) or len(self.sample_ids) != self.batch_size:
                raise ValueError("token sample_ids must be a tuple aligned with batch rows.")
            for index, sample_id in enumerate(self.sample_ids):
                _non_empty_string(sample_id, f"token sample_ids[{index}]")
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError("token sample_ids must be unique within a batch.")
        if self.audio_seconds is not None:
            if not isinstance(self.audio_seconds, Tensor):
                raise TypeError("token audio_seconds must be a Tensor or None.")
            if self.audio_seconds.ndim != 1 or self.audio_seconds.shape[0] != self.batch_size:
                raise ValueError("token audio_seconds must have shape [batch].")
            if not torch.is_floating_point(self.audio_seconds):
                raise TypeError("token audio_seconds must use a floating-point dtype.")
            if self.audio_seconds.device != self.input.input_ids.device:
                raise ValueError("token audio_seconds and inputs must share a device.")
            if _cpu(self.audio_seconds) and (
                not bool(torch.isfinite(self.audio_seconds).all())
                or bool((self.audio_seconds < 0).any())
            ):
                raise ValueError("token audio_seconds must be finite and non-negative.")

    @property
    def batch_size(self) -> int:
        return self.input.input_ids.shape[0]

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TokenBatch:
        return TokenBatch(
            input=self.input.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            prompt_lengths=self.prompt_lengths.to(
                device=device,
                non_blocking=non_blocking,
            ),
            routes=self.routes,
            sample_ids=self.sample_ids,
            audio_seconds=(
                None
                if self.audio_seconds is None
                else self.audio_seconds.to(device=device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> TokenBatch:
        return TokenBatch(
            input=self.input.pin_memory(),
            targets=self.targets.pin_memory(),
            prompt_lengths=self.prompt_lengths.pin_memory(),
            routes=self.routes,
            sample_ids=self.sample_ids,
            audio_seconds=(
                None if self.audio_seconds is None else self.audio_seconds.pin_memory()
            ),
        )


def _modalities(value: object, name: str) -> None:
    if not isinstance(value, frozenset):
        raise TypeError(f"{name} must be a frozenset of Modality values.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(modality, Modality) for modality in value):
        raise TypeError(f"{name} must contain only Modality values.")


def _audio_positions(positions: Tensor, mask: Tensor) -> None:
    _integer_tensor(positions, "audio condition positions", ndim=2)
    _bool_tensor(mask, "audio condition mask", shape=positions.shape)
    if positions.device != mask.device:
        raise ValueError("audio condition positions and mask must share a device.")
    if positions.shape[1] <= 0:
        raise ValueError("audio condition must have a non-empty frame axis.")
    if _cpu(mask) and not bool(mask.any()):
        raise ValueError("audio condition must contain at least one active position.")
    if _cpu(positions, mask) and not bool(positions.masked_select(~mask).eq(0).all()):
        raise ValueError("inactive audio condition positions must be zero.")


def _integer_tensor(value: object, name: str, *, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D.")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use a signed integer dtype.")


def _bool_tensor(value: object, name: str, *, shape: torch.Size) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}.")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must use bool dtype.")


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _tag_segment(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    if _TAG_SEGMENT.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_TAG_SEGMENT.pattern!r}.")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _cpu(*values: Tensor) -> bool:
    return all(value.device.type == "cpu" for value in values)


__all__ = [
    "IGNORE_INDEX",
    "Audio",
    "AudioCondition",
    "Content",
    "EmbeddedAudioCondition",
    "GenerationRequest",
    "GenerationResponse",
    "Modality",
    "Route",
    "Sample",
    "TokenBatch",
    "TokenAudioCondition",
    "TokenInput",
    "TokenTargets",
]
