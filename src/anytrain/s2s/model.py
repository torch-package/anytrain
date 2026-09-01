from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch
from torch import Tensor

from anytrain._compat import StrEnum

from .data import (
    AudioCondition,
    GenerationRequest,
    GenerationResponse,
    Modality,
    TokenBatch,
    TokenInput,
    TokenTargets,
)

_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)
_TAG_SEGMENT = re.compile(r"[a-z][a-z0-9_]*")
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class StageTiming:
    name: str
    seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _TAG_SEGMENT.fullmatch(self.name) is None:
            raise ValueError(f"stage timing name must match {_TAG_SEGMENT.pattern!r}.")
        _non_negative_float(self.seconds, "stage timing seconds")


@dataclass(frozen=True, slots=True)
class GenerationTiming:
    total_seconds: float
    stages: tuple[StageTiming, ...] = ()

    def __post_init__(self) -> None:
        _non_negative_float(self.total_seconds, "generation total_seconds")
        if not isinstance(self.stages, tuple):
            raise TypeError("generation stages must be a tuple of StageTiming values.")
        if any(not isinstance(stage, StageTiming) for stage in self.stages):
            raise TypeError("generation stages must contain only StageTiming values.")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("generation stage timing names must be unique.")


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int = 2_048
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens,
            int,
        ):
            raise TypeError("generation max_new_tokens must be an integer.")
        if self.max_new_tokens <= 0:
            raise ValueError("generation max_new_tokens must be positive.")
        if not isinstance(self.do_sample, bool):
            raise TypeError("generation do_sample must be a boolean.")
        _positive_float(self.temperature, "generation temperature")
        _probability(self.top_p, "generation top_p")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("generation seed must be an integer or None.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("generation seed must be non-negative.")


@dataclass(eq=False, frozen=True, slots=True)
class GenerationError:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or _TAG_SEGMENT.fullmatch(self.code) is None:
            raise ValueError(f"generation error code must match {_TAG_SEGMENT.pattern!r}.")
        if not isinstance(self.message, str):
            raise TypeError("generation error message must be a string.")
        if not self.message.strip():
            raise ValueError("generation error message must not be empty.")


class GenerationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(eq=False, frozen=True, slots=True)
class GenerationOutput:
    response: GenerationResponse | None
    timing: GenerationTiming | None = None
    error: GenerationError | None = None

    def __post_init__(self) -> None:
        if self.response is not None and not isinstance(self.response, GenerationResponse):
            raise TypeError("generation output response must be GenerationResponse or None.")
        if self.timing is not None and not isinstance(self.timing, GenerationTiming):
            raise TypeError("generation output timing must be GenerationTiming or None.")
        if self.error is not None and not isinstance(self.error, GenerationError):
            raise TypeError("generation output error must be GenerationError or None.")
        if self.error is None and self.response is None:
            raise ValueError("generation output without an error requires a response.")

    @property
    def status(self) -> GenerationStatus:
        if self.error is None:
            return GenerationStatus.SUCCEEDED
        if self.response is not None:
            return GenerationStatus.PARTIAL
        return GenerationStatus.FAILED


@runtime_checkable
class Generator(Protocol):
    """High-level capability shared by token-loop, Qwen-Omni, and UniSS providers."""

    def generate(
        self,
        requests: Sequence[GenerationRequest],
        options: GenerationOptions,
    ) -> Sequence[GenerationOutput]: ...


def generate(
    provider: Generator,
    requests: Sequence[GenerationRequest],
    options: GenerationOptions | None = None,
) -> tuple[GenerationOutput, ...]:
    """Run one provider and validate row alignment and response semantics."""

    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TypeError("generation requests must be a sequence of GenerationRequest values.")
    values = tuple(requests)
    if not values:
        raise ValueError("generation requests must not be empty.")
    if any(not isinstance(request, GenerationRequest) for request in values):
        raise TypeError("generation requests must contain only GenerationRequest values.")
    resolved_options = GenerationOptions() if options is None else options
    if not isinstance(resolved_options, GenerationOptions):
        raise TypeError("generation options must be GenerationOptions or None.")
    outputs = provider.generate(values, resolved_options)
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
        raise TypeError("Generator.generate() must return a sequence of GenerationOutput values.")
    resolved = tuple(outputs)
    if len(resolved) != len(values):
        raise ValueError("generation outputs must align with request rows.")
    for index, (request, output) in enumerate(zip(values, resolved, strict=True)):
        if not isinstance(output, GenerationOutput):
            raise TypeError(f"generation outputs[{index}] must be GenerationOutput.")
        response_modalities = (
            frozenset() if output.response is None else output.response.modalities
        )
        if output.error is None and response_modalities != request.response_modalities:
            raise ValueError(
                f"generation outputs[{index}] modalities must exactly match the request."
            )
        if output.error is not None and not response_modalities <= request.response_modalities:
            raise ValueError(
                f"generation outputs[{index}] partial modalities must be requested modalities."
            )
    return resolved


@dataclass(eq=False, frozen=True, slots=True)
class TokenStepInput(Generic[StateT]):
    """One cached or uncached autoregressive model call.

    ``selected_token_ids`` defines the output-logit columns in order. Provider-owned
    caches remain opaque in ``state`` so this contract does not expose HF or audio-head
    implementation types.
    """

    input_ids: Tensor
    attention_mask: Tensor
    selected_token_ids: Tensor
    audio: AudioCondition | None = None
    state: StateT | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        _integer_tensor(self.input_ids, "token step input_ids", ndim=2)
        if self.input_ids.shape[0] <= 0 or self.input_ids.shape[1] <= 0:
            raise ValueError("token step input_ids must have non-empty axes.")
        if not isinstance(self.attention_mask, Tensor):
            raise TypeError("token step attention_mask must be a Tensor.")
        if self.attention_mask.ndim != 2:
            raise ValueError("token step attention_mask must be 2D.")
        if self.attention_mask.dtype is not torch.bool:
            raise TypeError("token step attention_mask must use bool dtype.")
        if self.attention_mask.shape[0] != self.input_ids.shape[0]:
            raise ValueError("token step attention_mask must align with the input batch.")
        if self.attention_mask.shape[1] < self.input_ids.shape[1]:
            raise ValueError("token step attention_mask cannot be shorter than input_ids.")
        if self.attention_mask.device != self.input_ids.device:
            raise ValueError("token step input tensors must share a device.")
        if not isinstance(self.use_cache, bool):
            raise TypeError("token step use_cache must be a boolean.")
        _integer_tensor(self.selected_token_ids, "selected_token_ids", ndim=1)
        if self.selected_token_ids.numel() == 0:
            raise ValueError("selected_token_ids must not be empty.")
        if self.selected_token_ids.device != self.input_ids.device:
            raise ValueError("selected_token_ids and token step inputs must share a device.")
        if self.selected_token_ids.device.type == "cpu" and (
            self.selected_token_ids.unique().numel() != self.selected_token_ids.numel()
        ):
            raise ValueError("selected_token_ids must be unique.")
        if not self.use_cache and self.state is not None:
            raise ValueError("uncached token steps cannot carry provider state.")
        if not self.use_cache or self.state is None:
            if self.attention_mask.shape != self.input_ids.shape:
                raise ValueError(
                    "uncached and cached-prefill attention masks must align with input_ids."
                )
            TokenInput(self.input_ids, self.attention_mask, self.audio)
            return
        if self.attention_mask.shape[1] <= self.input_ids.shape[1]:
            raise ValueError(
                "cached decode attention masks must include context before the input suffix."
            )
        if self.audio is not None:
            raise ValueError("cached decode token steps cannot repeat audio condition data.")
        if self.attention_mask.device.type == "cpu" and not bool(
            self.attention_mask.any(dim=1).all()
        ):
            raise ValueError("every token step row must contain active context.")


@dataclass(eq=False, frozen=True, slots=True)
class TokenStepOutput(Generic[StateT]):
    logits: Tensor
    state: StateT | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logits, Tensor):
            raise TypeError("token step logits must be a Tensor.")
        if self.logits.ndim != 2:
            raise ValueError("token step logits must have shape [batch, selected tokens].")
        if not torch.is_floating_point(self.logits):
            raise TypeError("token step logits must use a floating-point dtype.")


@runtime_checkable
class TokenStepper(Protocol[StateT]):
    """Optional low-level capability implemented by token-loop providers only."""

    def step_tokens(self, input: TokenStepInput[StateT]) -> TokenStepOutput[StateT]: ...


def step_tokens(
    provider: TokenStepper[StateT],
    input: TokenStepInput[StateT],
) -> TokenStepOutput[StateT]:
    if not isinstance(input, TokenStepInput):
        raise TypeError("step_tokens input must be TokenStepInput.")
    output = provider.step_tokens(input)
    if not isinstance(output, TokenStepOutput):
        raise TypeError("TokenStepper.step_tokens() must return TokenStepOutput.")
    expected = (input.input_ids.shape[0], input.selected_token_ids.numel())
    if output.logits.shape != expected:
        raise ValueError(f"token step logits must have shape {expected}.")
    if output.logits.device != input.input_ids.device:
        raise ValueError("token step logits and inputs must share a device.")
    if input.use_cache and output.state is None:
        raise ValueError("cached token steps must return provider state.")
    if not input.use_cache and output.state is not None:
        raise ValueError("uncached token steps must not return provider state.")
    return output


@dataclass(eq=False, frozen=True, slots=True)
class LossStat:
    """A differentiable loss numerator and its exact observation count."""

    loss_sum: Tensor
    count: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.loss_sum, Tensor):
            raise TypeError("loss_sum must be a Tensor.")
        if self.loss_sum.ndim != 0:
            raise ValueError("loss_sum must be a scalar Tensor.")
        if not torch.is_floating_point(self.loss_sum):
            raise TypeError("loss_sum must use a floating-point dtype.")
        if not isinstance(self.count, Tensor):
            raise TypeError("loss count must be a Tensor.")
        if self.count.ndim != 0:
            raise ValueError("loss count must be a scalar Tensor.")
        if self.count.dtype is not torch.int64:
            raise TypeError("loss count must use int64 dtype.")
        if self.count.device != self.loss_sum.device:
            raise ValueError("loss_sum and count must share a device.")
        if self.count.requires_grad:
            raise ValueError("loss count must not require gradients.")
        if self.count.device.type == "cpu" and bool(self.count < 0):
            raise ValueError("loss count must be non-negative.")

    def mean(self) -> Tensor:
        denominator = self.count.clamp_min(1).to(dtype=self.loss_sum.dtype)
        return self.loss_sum / denominator

    def detached(self) -> LossStat:
        return LossStat(self.loss_sum.detach(), self.count.detach())

    def __add__(self, other: LossStat) -> LossStat:
        if not isinstance(other, LossStat):
            return NotImplemented
        if self.loss_sum.device != other.loss_sum.device:
            raise ValueError("loss stats must share a device before addition.")
        return LossStat(
            self.loss_sum + other.loss_sum.to(dtype=self.loss_sum.dtype),
            self.count + other.count,
        )


@dataclass(eq=False, frozen=True, slots=True)
class TokenLossStats:
    text: LossStat | None = None
    audio: LossStat | None = None

    def __post_init__(self) -> None:
        values = tuple(value for value in (self.text, self.audio) if value is not None)
        if not values:
            raise ValueError("token loss stats require text, audio, or both modality stats.")
        for name, value in (("text", self.text), ("audio", self.audio)):
            if value is None:
                continue
            if not isinstance(value, LossStat):
                raise TypeError(f"token {name} loss must be LossStat or None.")
            if value.loss_sum.device != values[0].loss_sum.device:
                raise ValueError("all token loss stats must share a device.")

    @property
    def total(self) -> LossStat:
        if self.text is None:
            if self.audio is None:  # pragma: no cover - guarded by __post_init__
                raise RuntimeError("token loss stats have no modality values.")
            return self.audio
        if self.audio is None:
            return self.text
        return self.text + self.audio

    @property
    def modalities(self) -> frozenset[Modality]:
        values: set[Modality] = set()
        if self.text is not None:
            values.add(Modality.TEXT)
        if self.audio is not None:
            values.add(Modality.AUDIO)
        return frozenset(values)

    def for_modality(self, modality: Modality) -> LossStat | None:
        if modality is Modality.TEXT:
            return self.text
        if modality is Modality.AUDIO:
            return self.audio
        raise TypeError("modality must be a Modality.")

    def detached(self) -> TokenLossStats:
        return TokenLossStats(
            text=None if self.text is None else self.text.detached(),
            audio=None if self.audio is None else self.audio.detached(),
        )

    def __add__(self, other: TokenLossStats) -> TokenLossStats:
        if not isinstance(other, TokenLossStats):
            return NotImplemented
        return TokenLossStats(
            text=_optional_sum(self.text, other.text),
            audio=_optional_sum(self.audio, other.audio),
        )


@runtime_checkable
class TokenScorer(Protocol):
    """Teacher-forced scoring capability; concrete providers call Liger directly."""

    def score_tokens(
        self,
        input: TokenInput,
        targets: TokenTargets,
    ) -> TokenLossStats: ...


def score_tokens(provider: TokenScorer, batch: TokenBatch) -> TokenLossStats:
    if not isinstance(batch, TokenBatch):
        raise TypeError("score_tokens batch must be TokenBatch.")
    stats = provider.score_tokens(batch.input, batch.targets)
    if not isinstance(stats, TokenLossStats):
        raise TypeError("TokenScorer.score_tokens() must return TokenLossStats.")
    if stats.modalities != batch.targets.modalities:
        raise ValueError("token loss modalities must exactly match batch target modalities.")
    if stats.total.count.device.type == "cpu" and bool(stats.total.count <= 0):
        raise ValueError("token total loss count must be positive.")
    for modality in stats.modalities:
        value = stats.for_modality(modality)
        if value is None or (
            value.count.device.type == "cpu" and bool(value.count <= 0)
        ):
            raise ValueError(f"token {modality.value} loss count must be positive.")
    return stats


def _optional_sum(first: LossStat | None, second: LossStat | None) -> LossStat | None:
    if first is None and second is None:
        return None
    if first is None:
        return second
    if second is None:
        return first
    return first + second


def _non_negative_float(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _positive_float(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _probability(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)) or not 0 < float(value) <= 1:
        raise ValueError(f"{name} must be in (0, 1].")


def _integer_tensor(value: object, name: str, *, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D.")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use a signed integer dtype.")


__all__ = [
    "GenerationError",
    "GenerationOptions",
    "GenerationOutput",
    "GenerationStatus",
    "GenerationTiming",
    "Generator",
    "LossStat",
    "StageTiming",
    "TokenLossStats",
    "TokenScorer",
    "TokenStepInput",
    "TokenStepOutput",
    "TokenStepper",
    "generate",
    "score_tokens",
    "step_tokens",
]
