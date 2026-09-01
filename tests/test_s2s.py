from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lightning import LightningModule, Trainer

from anytrain.s2s import (
    Audio,
    BatchUnits,
    Content,
    EmbeddedAudioCondition,
    GenerationError,
    GenerationOptions,
    GenerationOutput,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
    GenerationTiming,
    Generator,
    LossStat,
    Modality,
    Route,
    S2SMetricsCallback,
    StageTiming,
    TokenAudioCondition,
    TokenBatch,
    TokenInput,
    TokenLossStats,
    TokenScorer,
    TokenStepInput,
    TokenStepOutput,
    TokenStepper,
    TokenTargets,
    generate,
    generation_metrics,
    step_report,
    step_tokens,
    token_objective,
)


class _SemanticProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[GenerationRequest, ...], GenerationOptions]] = []

    def generate(
        self,
        requests: Sequence[GenerationRequest],
        options: GenerationOptions,
    ) -> Sequence[GenerationOutput]:
        self.calls.append((tuple(requests), options))
        return [
            GenerationOutput(
                GenerationResponse(text="translated", audio=_audio(24_000)),
                GenerationTiming(
                    total_seconds=0.5,
                    stages=(StageTiming("decoder", 0.2),),
                ),
            )
            for _ in requests
        ]


class S2SProvider(_SemanticProvider):
    pass


class QwenOmniProvider(_SemanticProvider):
    pass


class UniSSProvider(_SemanticProvider):
    pass


class _TokenScorer:
    def __init__(self, value: float = 2.0) -> None:
        self.value = torch.tensor(value, requires_grad=True)
        self.calls: list[tuple[TokenInput, TokenTargets]] = []

    def score_tokens(self, input: TokenInput, targets: TokenTargets) -> TokenLossStats:
        self.calls.append((input, targets))
        text_count = _count(targets.text_mask)
        audio_count = _count(targets.audio_mask)
        return TokenLossStats(
            text=LossStat(self.value * text_count, text_count.to(dtype=torch.int64)),
            audio=LossStat(
                self.value * 2 * audio_count,
                audio_count.to(dtype=torch.int64),
            ),
        )


class S2STokenScorer(_TokenScorer):
    pass


class UniSSTokenScorer(_TokenScorer):
    pass


class _TokenStepper:
    def __init__(self) -> None:
        self.calls: list[TokenStepInput[tuple[int, ...]]] = []

    def step_tokens(
        self,
        input: TokenStepInput[tuple[int, ...]],
    ) -> TokenStepOutput[tuple[int, ...]]:
        self.calls.append(input)
        logits = torch.arange(
            input.selected_token_ids.numel(),
            dtype=torch.float32,
            device=input.input_ids.device,
        ).expand(input.input_ids.shape[0], -1)
        return TokenStepOutput(logits, state=(*(input.state or ()), 1))


class S2STokenStepper(_TokenStepper):
    pass


class UniSSTokenStepper(_TokenStepper):
    pass


@pytest.mark.parametrize("provider_type", (S2SProvider, QwenOmniProvider, UniSSProvider))
def test_three_provider_families_use_one_semantic_generation_path(provider_type) -> None:
    provider = provider_type()
    request = _request()
    options = GenerationOptions(max_new_tokens=64, seed=7)

    outputs = generate(provider, (request,), options)

    assert provider.calls == [((request,), options)]
    assert outputs[0].response is not None
    assert outputs[0].response.modalities == frozenset((Modality.TEXT, Modality.AUDIO))
    assert outputs[0].status is GenerationStatus.SUCCEEDED
    assert isinstance(provider, Generator)


def test_generation_validates_row_alignment_and_response_modalities() -> None:
    request = _request()

    with pytest.raises(ValueError, match="align"):
        generate(
            cast(Generator, SimpleNamespace(generate=lambda _requests, _options: [])),
            (request,),
        )
    with pytest.raises(ValueError, match="exactly match"):
        generate(
            cast(
                Generator,
                SimpleNamespace(
                    generate=lambda _requests, _options: [
                        GenerationOutput(GenerationResponse(text="only text"))
                    ]
                ),
            ),
            (request,),
        )


def test_generation_options_validate_common_provider_controls() -> None:
    assert GenerationOptions() == GenerationOptions(
        max_new_tokens=2_048,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        seed=None,
    )
    with pytest.raises(ValueError, match="positive"):
        GenerationOptions(max_new_tokens=0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        GenerationOptions(top_p=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        GenerationOptions(seed=-1)
    assert GenerationOptions(seed=0).seed == 0


def test_generation_preserves_partial_and_failed_provider_results() -> None:
    request = _request()
    partial = GenerationOutput(
        GenerationResponse(text="partial text"),
        error=GenerationError("invalid_audio_payload", "missing end marker"),
    )
    failed = GenerationOutput(
        None,
        error=GenerationError("backend_failure", "provider failed"),
    )

    partial_result = generate(
        cast(
            Generator,
            SimpleNamespace(generate=lambda _requests, _options: [partial]),
        ),
        (request,),
    )
    failed_result = generate(
        cast(
            Generator,
            SimpleNamespace(generate=lambda _requests, _options: [failed]),
        ),
        (request,),
    )

    assert partial_result[0].status is GenerationStatus.PARTIAL
    assert failed_result[0].status is GenerationStatus.FAILED


@pytest.mark.parametrize("stepper_type", (S2STokenStepper, UniSSTokenStepper))
def test_token_loop_providers_share_one_optional_step_contract(stepper_type) -> None:
    provider = stepper_type()
    batch = _batch()
    input = TokenStepInput(
        input_ids=batch.input.input_ids,
        attention_mask=batch.input.attention_mask,
        selected_token_ids=torch.tensor([91, 7, 42]),
        audio=batch.input.audio,
    )

    output = step_tokens(provider, input)

    assert provider.calls == [input]
    assert output.logits.shape == (batch.batch_size, 3)
    assert output.logits[0].tolist() == [0.0, 1.0, 2.0]
    assert output.state == (1,)
    assert isinstance(provider, TokenStepper)
    assert not isinstance(QwenOmniProvider(), TokenStepper)


def test_token_step_validates_selected_order_alignment() -> None:
    input = TokenStepInput(
        input_ids=torch.ones(2, 1, dtype=torch.int64),
        attention_mask=torch.ones(2, 1, dtype=torch.bool),
        selected_token_ids=torch.tensor([30, 10]),
    )
    provider = cast(
        TokenStepper[object],
        SimpleNamespace(
            step_tokens=lambda _input: TokenStepOutput(torch.zeros(2, 3)),
        ),
    )

    with pytest.raises(ValueError, match="shape"):
        step_tokens(provider, input)


def test_cached_token_step_uses_suffix_and_opaque_provider_state() -> None:
    provider = S2STokenStepper()
    input: TokenStepInput[tuple[int, ...]] = TokenStepInput(
        input_ids=torch.tensor([[41], [32]]),
        attention_mask=torch.ones(2, 5, dtype=torch.bool),
        selected_token_ids=torch.tensor([91, 7]),
        state=(0,),
    )

    output = step_tokens(provider, input)

    assert output.state == (0, 1)
    with pytest.raises(ValueError, match="cannot repeat audio"):
        TokenStepInput(
            input_ids=input.input_ids,
            attention_mask=input.attention_mask,
            selected_token_ids=input.selected_token_ids,
            audio=_batch().input.audio,
            state=input.state,
        )


def test_semantic_data_requires_explicit_audio_shape_and_matching_target() -> None:
    with pytest.raises(ValueError, match="channels, time"):
        Audio(torch.zeros(8), 16_000)
    from anytrain.s2s import Sample

    assert Sample(_request(), Content(text="reference")).target.text == "reference"
    text_request = GenerationRequest(
        Content(text="source"),
        Route("mt"),
        frozenset((Modality.TEXT,)),
    )
    with pytest.raises(ValueError, match="requested"):
        Sample(text_request, Content(audio=_audio(16_000)))
    with pytest.raises(ValueError, match="match"):
        Route("S2ST")


@pytest.mark.parametrize("embedded", (False, True))
def test_audio_condition_has_one_mask_for_discrete_and_embedding_inputs(embedded: bool) -> None:
    batch = _batch(embedded=embedded)

    assert batch.input.audio is not None
    assert isinstance(batch.input.audio, EmbeddedAudioCondition) is embedded
    assert batch.prompt_lengths.tolist() == [2, 2]
    assert batch.to("cpu").routes == batch.routes

    positions = torch.tensor([[1, -1], [1, -1]])
    mask = torch.tensor([[True, False], [True, False]])
    with pytest.raises(ValueError, match="inactive"):
        TokenAudioCondition(positions, mask)


def test_token_batch_keeps_prompt_boundary_and_rejects_audio_after_prompt() -> None:
    batch = _batch()
    condition = TokenAudioCondition(
        positions=torch.tensor([[2, 0], [1, 0]]),
        mask=torch.tensor([[True, False], [True, False]]),
    )

    with pytest.raises(ValueError, match="inside the prompt"):
        TokenBatch(
            input=TokenInput(batch.input.input_ids, batch.input.attention_mask, condition),
            targets=batch.targets,
            prompt_lengths=batch.prompt_lengths,
            routes=batch.routes,
            audio_seconds=batch.audio_seconds,
        )


@pytest.mark.parametrize("empty_name", ("text_mask", "audio_mask"))
def test_token_targets_reject_an_empty_declared_modality(empty_name: str) -> None:
    labels = torch.full((2, 3), -100)
    active = torch.zeros_like(labels, dtype=torch.bool)
    active[:, 1] = True
    labels[active] = 1
    masks = {
        "text_mask": active if empty_name == "audio_mask" else torch.zeros_like(active),
        "audio_mask": active if empty_name == "text_mask" else torch.zeros_like(active),
    }
    with pytest.raises(ValueError, match="select at least one target"):
        TokenTargets(labels, **masks)


def test_token_contracts_reject_unsigned_ids_and_labels() -> None:
    with pytest.raises(TypeError, match="signed integer"):
        TokenInput(
            torch.ones(1, 2, dtype=torch.uint8),
            torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="signed integer"):
        TokenTargets(
            torch.zeros(1, 2, dtype=torch.uint8),
            text_mask=torch.tensor([[False, True]]),
        )
    with pytest.raises(TypeError, match="signed integer"):
        TokenStepInput(
            input_ids=torch.ones(1, 2, dtype=torch.uint8),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            selected_token_ids=torch.tensor([1]),
        )


def test_token_batch_supports_pinning_and_non_blocking_transfer(monkeypatch) -> None:
    batch = _batch(embedded=True)
    pin_calls: list[torch.Tensor] = []
    to_calls: list[bool] = []
    tensor_to = torch.Tensor.to

    def fake_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        pin_calls.append(tensor)
        return tensor

    def tracked_to(
        tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if "non_blocking" in kwargs:
            to_calls.append(bool(kwargs["non_blocking"]))
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin_memory)
    monkeypatch.setattr(torch.Tensor, "to", tracked_to)

    pinned = batch.pin_memory()
    moved = pinned.to("cpu", non_blocking=True)

    assert len(pin_calls) == 10
    assert to_calls and all(to_calls)
    assert moved.routes == batch.routes
    assert moved.sample_ids == batch.sample_ids


@pytest.mark.parametrize("scorer_type", (S2STokenScorer, UniSSTokenScorer))
def test_token_capability_uses_one_objective_and_derives_total_loss(scorer_type) -> None:
    scorer = scorer_type()
    batch = _batch()

    output = token_objective(scorer, batch)

    assert scorer.calls == [(batch.input, batch.targets)]
    # text: 2 * 2 tokens, audio: 4 * 1 token
    torch.testing.assert_close(output["loss"], torch.tensor(8.0 / 3.0))
    output["loss"].backward()
    torch.testing.assert_close(scorer.value.grad, torch.tensor(4.0 / 3.0))
    assert not output["s2s_report"].loss.total.loss_sum.requires_grad
    assert isinstance(scorer, TokenScorer)


def test_mixed_route_report_never_splits_one_batch_loss() -> None:
    output = token_objective(S2STokenScorer(), _batch())

    assert len(output["s2s_report"].routes) == 2
    assert all(route.loss is None for route in output["s2s_report"].routes)


def test_homogeneous_route_report_attributes_exact_sum_and_count() -> None:
    batch = _batch(routes=(Route("s2st", "target_cot"),) * 2)
    output = token_objective(UniSSTokenScorer(), batch)

    route = output["s2s_report"].routes[0]
    assert route.loss is not None
    assert int(route.loss.total.count) == 3
    assert int(route.units.input_tokens) == 7
    assert int(route.units.padded_tokens) == 3


def test_loss_and_unit_counts_require_int64() -> None:
    with pytest.raises(TypeError, match="loss count must use int64"):
        LossStat(torch.tensor(1.0), torch.tensor(1, dtype=torch.int32))
    scalar = torch.zeros((), dtype=torch.int64)
    with pytest.raises(TypeError, match="rows must use int64"):
        BatchUnits(
            rows=scalar.to(dtype=torch.int32),
            input_tokens=scalar,
            padded_tokens=scalar,
            text_tokens=scalar,
            audio_tokens=scalar,
            audio_seconds=torch.zeros((), dtype=torch.float64),
        )

    large = torch.tensor(2**31, dtype=torch.int64)
    stats = LossStat(torch.tensor(1.0), large) + LossStat(torch.tensor(2.0), large)
    units = BatchUnits(
        rows=large,
        input_tokens=large,
        padded_tokens=large,
        text_tokens=large,
        audio_tokens=large,
        audio_seconds=torch.tensor(1.0, dtype=torch.float64),
    )
    assert int(stats.count) == 2**32
    assert int((units + units).input_tokens) == 2**32


def test_step_report_rejects_loss_modalities_missing_from_targets() -> None:
    batch = _batch()
    text = LossStat(torch.tensor(1.0), torch.tensor(1, dtype=torch.int64))

    with pytest.raises(ValueError, match="exactly match"):
        step_report(batch, TokenLossStats(text=text))


def test_metrics_callback_logs_exact_window_without_model_dispatch() -> None:
    route = Route("s2st", "target_cot")
    batch = _batch(routes=(route,) * 2)
    output = token_objective(UniSSTokenScorer(), batch)
    writer = _Writer()
    trainer = _trainer(writer, step=1)
    callback = S2SMetricsCallback((route,), every_n_steps=1)

    callback.on_train_batch_end(trainer, _pl_module(), output, batch, 0)

    assert writer.values["loss/train"] == pytest.approx(8.0 / 3.0)
    assert writer.values["loss/train/s2st/target_cot"] == pytest.approx(8.0 / 3.0)
    assert writer.values["data/train/input_tokens/s2st/target_cot"] == 7.0
    assert writer.values["data/train/supervision_ratio/s2st/target_cot"] == pytest.approx(
        3 / 7
    )


def test_metrics_callback_omits_inexact_route_loss_and_requires_scalar_logger() -> None:
    routes = (Route("s2st"), Route("asr"))
    output = token_objective(S2STokenScorer(), _batch(routes=routes))
    writer = _Writer()
    callback = S2SMetricsCallback(routes, every_n_steps=1)

    callback.on_train_batch_end(_trainer(writer, step=1), _pl_module(), output, None, 0)

    assert "loss/train/s2st/direct" not in writer.values
    assert "loss/train/asr/direct" not in writer.values

    missing = S2SMetricsCallback(routes, every_n_steps=1)
    with pytest.raises(RuntimeError, match="add_scalar"):
        missing.on_train_batch_end(
            _trainer(None, step=1),
            _pl_module(),
            output,
            None,
            0,
        )


def test_metrics_callback_drops_prior_route_loss_after_a_mixed_batch() -> None:
    route = Route("s2st")
    other = Route("asr")
    writer = _Writer()
    callback = S2SMetricsCallback((route, other), every_n_steps=2)
    homogeneous = token_objective(
        S2STokenScorer(),
        _batch(routes=(route, route)),
    )
    mixed = token_objective(
        S2STokenScorer(),
        _batch(routes=(route, other)),
    )

    callback.on_train_batch_end(
        _trainer(writer, step=1),
        _pl_module(),
        homogeneous,
        None,
        0,
    )
    callback.on_train_batch_end(
        _trainer(writer, step=2),
        _pl_module(),
        mixed,
        None,
        1,
    )

    assert "loss/train" in writer.values
    assert "loss/train/s2st/direct" not in writer.values


def test_generation_metrics_use_named_stages_without_fixed_architecture_fields() -> None:
    output = GenerationOutput(
        GenerationResponse(text="partial"),
        GenerationTiming(
            total_seconds=2.0,
            stages=(StageTiming("thinker", 0.5), StageTiming("talker", 1.0)),
        ),
        GenerationError("invalid_audio_payload", "missing payload"),
    )

    values = generation_metrics(_request(), output)

    assert values["generation/s2st/direct/status/partial"] == 1.0
    assert values["generation/s2st/direct/error/invalid_audio_payload"] == 1.0
    assert values["generation/s2st/direct/rtf"] == 2.0
    assert values["generation/s2st/direct/stage/thinker_rtf"] == 0.5
    assert values["generation/s2st/direct/stage/talker_seconds"] == 1.0


def _request() -> GenerationRequest:
    return GenerationRequest(
        source=Content(audio=_audio(16_000)),
        route=Route("s2st"),
        response_modalities=frozenset((Modality.TEXT, Modality.AUDIO)),
        target_language="zh",
        sample_id="sample-0",
    )


def _audio(sample_rate: int) -> Audio:
    return Audio(torch.zeros(1, sample_rate), sample_rate)


def _batch(
    *,
    embedded: bool = False,
    routes: tuple[Route, ...] | None = None,
) -> TokenBatch:
    input_ids = torch.tensor(
        [
            [10, 20, 30, 40, 0],
            [11, 21, 31, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    condition_kwargs = {
        "positions": torch.tensor([[1, 0], [1, 0]]),
        "mask": torch.tensor([[True, False], [True, False]]),
    }
    audio = (
        EmbeddedAudioCondition(**condition_kwargs, embeddings=torch.zeros(2, 2, 4))
        if embedded
        else TokenAudioCondition(**condition_kwargs)
    )
    text_mask = torch.tensor(
        [
            [False, False, True, False, False],
            [False, False, True, False, False],
        ]
    )
    audio_mask = torch.tensor(
        [
            [False, False, False, True, False],
            [False, False, False, False, False],
        ]
    )
    labels = torch.where(text_mask | audio_mask, input_ids, -100)
    return TokenBatch(
        input=TokenInput(input_ids, attention_mask, audio),
        targets=TokenTargets(labels, text_mask=text_mask, audio_mask=audio_mask),
        prompt_lengths=torch.tensor([2, 2]),
        routes=routes or (Route("s2st"), Route("asr")),
        sample_ids=("sample-0", "sample-1"),
        audio_seconds=torch.tensor([1.0, 2.0]),
    )


def _count(mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.zeros((), dtype=torch.float32)
    return mask.sum(dtype=torch.float32)


class _Writer:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        assert step >= 0
        self.values[tag] = value


def _trainer(writer: _Writer | None, *, step: int) -> Trainer:
    logger = None if writer is None else SimpleNamespace(experiment=writer)
    return cast(
        Trainer,
        SimpleNamespace(
            global_step=step,
            world_size=1,
            is_global_zero=True,
            logger=logger,
            loggers=(),
        ),
    )


def _pl_module() -> LightningModule:
    return cast(LightningModule, SimpleNamespace())
