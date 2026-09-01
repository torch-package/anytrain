from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lightning import LightningModule, Trainer
from torch import nn

from anytrain.s2s import (
    Audio,
    BatchUnits,
    ConditionGainCallback,
    ConditionGainKey,
    ConditionGainProbe,
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
    condition_gain,
    condition_gain_metrics,
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

    def __call__(self, input: TokenInput, targets: TokenTargets) -> TokenLossStats:
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


class _ScriptedScorer:
    def __init__(
        self,
        means: Sequence[float],
        *,
        count_offset: int = 0,
        events: list[str] | None = None,
        advance_rng: bool = False,
    ) -> None:
        self.means = tuple(means)
        self.count_offset = count_offset
        self.events = events
        self.advance_rng = advance_rng
        self.calls: list[tuple[TokenInput, TokenTargets]] = []

    def __call__(self, input: TokenInput, targets: TokenTargets) -> TokenLossStats:
        index = len(self.calls)
        if index >= len(self.means):
            raise RuntimeError("scripted scorer has no value for this call.")
        self.calls.append((input, targets))
        if self.events is not None:
            self.events.append("score")
        if self.advance_rng:
            torch.rand(())
        values: dict[Modality, LossStat] = {}
        for modality in targets.modalities:
            mask = targets.mask(modality)
            if mask is None:
                raise RuntimeError("scripted scorer target mask is unavailable.")
            target_count = mask.sum(dtype=torch.int64)
            loss_sum = torch.tensor(
                self.means[index],
                dtype=torch.float32,
                device=mask.device,
            ) * target_count.to(dtype=torch.float32)
            values[modality] = LossStat(loss_sum, target_count + self.count_offset)
        return TokenLossStats(
            text=values.get(Modality.TEXT),
            audio=values.get(Modality.AUDIO),
        )


class _GainModule(LightningModule):
    def __init__(self, scorer: TokenScorer) -> None:
        super().__init__()
        self.scorer = scorer
        self.active = nn.Dropout()
        self.inactive = nn.Sequential(nn.Dropout())
        self.inactive.eval()

    def forward(self, input: TokenInput, targets: TokenTargets) -> TokenLossStats:
        return self.scorer(input, targets)


class _Precision:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        if self.events is not None:
            self.events.append("enter")
        try:
            yield
        finally:
            if self.events is not None:
                self.events.append("exit")


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


def test_condition_gain_accepts_moved_targets_with_the_same_ordered_stream() -> None:
    route = Route("s2st", "full_cot")
    key = ConditionGainKey(route, "source_audio", "source_cot")
    probe = ConditionGainProbe(
        "sample_0_source_to_cot",
        key,
        _gain_batch(route=route),
        _gain_batch(
            route=route,
            target_positions=(4, 5),
            sequence_length=6,
            prompt_length=4,
        ),
    )

    result = condition_gain(
        _ScriptedScorer((1.0, 2.0)),
        (probe,),
        bootstrap_samples=0,
    )[0]

    assert result.total is not None
    assert result.total.matched_loss == 1.0
    assert result.total.counterfactual_loss == 2.0
    assert result.total.gain == 1.0


@pytest.mark.parametrize("target_tokens", ((51, 99), (52, 51)))
def test_condition_gain_rejects_changed_target_values_or_order(
    target_tokens: tuple[int, ...],
) -> None:
    route = Route("s2st", "full_cot")
    key = ConditionGainKey(route, "source_audio", "source_cot")

    with pytest.raises(ValueError, match="ordered teacher-forced"):
        ConditionGainProbe(
            "changed_target",
            key,
            _gain_batch(route=route),
            _gain_batch(route=route, target_tokens=target_tokens),
        )


def test_condition_gain_rejects_changed_target_modality() -> None:
    route = Route("s2st", "full_cot")
    key = ConditionGainKey(route, "source_audio", "source_cot")

    with pytest.raises(ValueError, match="same modalities"):
        ConditionGainProbe(
            "changed_modality",
            key,
            _gain_batch(route=route),
            _gain_batch(
                route=route,
                target_modalities=(Modality.AUDIO, Modality.AUDIO),
            ),
        )


def test_condition_gain_rejects_changed_route() -> None:
    route = Route("s2st", "full_cot")
    key = ConditionGainKey(route, "source_audio", "source_cot")

    with pytest.raises(ValueError, match="route must match"):
        ConditionGainProbe(
            "changed_route",
            key,
            _gain_batch(route=route),
            _gain_batch(route=Route("asr")),
        )


def test_condition_gain_rejects_changed_sample_lineage() -> None:
    route = Route("s2st", "full_cot")
    key = ConditionGainKey(route, "source_audio", "source_cot")

    with pytest.raises(ValueError, match="sample_ids lineage"):
        ConditionGainProbe(
            "changed_sample",
            key,
            _gain_batch(route=route, sample_id="sample_0"),
            _gain_batch(route=route, sample_id="sample_1"),
        )


def test_condition_gain_rejects_changed_cross_modality_target_order() -> None:
    route = Route("s2st")
    key = ConditionGainKey(route, "source_audio", "final")

    with pytest.raises(ValueError, match="ordered teacher-forced"):
        ConditionGainProbe(
            "changed_joint_order",
            key,
            _gain_batch(
                route=route,
                target_tokens=(51, 52),
                target_modalities=(Modality.TEXT, Modality.AUDIO),
            ),
            _gain_batch(
                route=route,
                target_tokens=(52, 51),
                target_modalities=(Modality.AUDIO, Modality.TEXT),
            ),
        )


def test_condition_gain_rejects_changed_teacher_forced_target_history() -> None:
    route = Route("s2st", "full_cot")
    key = ConditionGainKey(route, "target_cot", "final")

    with pytest.raises(ValueError, match="ordered teacher-forced"):
        ConditionGainProbe(
            "changed_target_history",
            key,
            _gain_batch(route=route),
            _gain_batch(route=route, target_input_tokens=(91, 92)),
        )


def test_condition_gain_rejects_scorer_counts_that_do_not_match_target_masks() -> None:
    probe = _gain_probe()

    with pytest.raises(ValueError, match="scorer counts must exactly match target masks"):
        condition_gain(
            _ScriptedScorer((1.0, 2.0), count_offset=1),
            (probe,),
            bootstrap_samples=0,
        )


@pytest.mark.parametrize("loss", (float("nan"), float("inf")))
def test_condition_gain_rejects_non_finite_loss_sums(loss: float) -> None:
    with pytest.raises(ValueError, match="loss sums must be finite"):
        condition_gain(
            _ScriptedScorer((loss, loss)),
            (_gain_probe(),),
            bootstrap_samples=0,
        )


@pytest.mark.parametrize(
    ("means", "expected_matched", "expected_counterfactual", "expected_gain"),
    (
        ((1.0, 5.0, 3.0, 4.0), 2.5, 4.25, 1.75),
        ((5.0, 1.0, 4.0, 2.0), 4.25, 1.75, -2.5),
    ),
)
def test_condition_gain_is_exactly_token_weighted(
    means: tuple[float, ...],
    expected_matched: float,
    expected_counterfactual: float,
    expected_gain: float,
) -> None:
    key = ConditionGainKey(Route("tts"), "source_text", "final")
    probes = (
        _gain_probe(
            name="short",
            key=key,
            target_positions=(2,),
            target_tokens=(51,),
            sequence_length=3,
        ),
        _gain_probe(
            name="long",
            key=key,
            target_positions=(2, 3, 4),
            target_tokens=(51, 52, 53),
            sequence_length=5,
        ),
    )

    stat = condition_gain(
        _ScriptedScorer(means),
        probes,
        bootstrap_samples=0,
    )[0].total

    assert stat is not None
    assert stat.matched_loss == pytest.approx(expected_matched)
    assert stat.counterfactual_loss == pytest.approx(expected_counterfactual)
    assert stat.gain == pytest.approx(expected_gain)
    assert stat.target_count == 4


def test_condition_gain_bootstrap_uses_a_private_rng() -> None:
    key = ConditionGainKey(Route("asr"), "source_audio", "final")
    probes = (
        _gain_probe(name="first", key=key),
        _gain_probe(name="second", key=key),
    )
    torch.manual_seed(91)
    before = torch.random.get_rng_state()

    result = condition_gain(
        _ScriptedScorer((1.0, 2.0, 3.0, 5.0)),
        probes,
        bootstrap_samples=32,
        bootstrap_seed=7,
    )[0]

    assert torch.equal(before, torch.random.get_rng_state())
    assert result.total is not None
    assert result.total.bootstrap_low is not None
    assert result.total.bootstrap_high is not None


def test_condition_gain_keeps_source_to_cot_and_cot_to_final_separate() -> None:
    route = Route("s2st", "full_cot")
    source_to_cot = ConditionGainKey(route, "source_audio", "source_cot")
    cot_to_final = ConditionGainKey(route, "target_cot", "final")
    probes = (
        _gain_probe(name="source_to_cot", key=source_to_cot),
        _gain_probe(
            name="cot_to_final",
            key=cot_to_final,
            target_positions=(3,),
            target_tokens=(71,),
            sequence_length=4,
            prompt_length=3,
        ),
    )

    results = {
        result.key: result
        for result in condition_gain(
            _ScriptedScorer((1.0, 3.0, 2.0, 6.0)),
            probes,
            bootstrap_samples=0,
        )
    }

    assert set(results) == {source_to_cot, cot_to_final}
    source_stat = results[source_to_cot].total
    final_stat = results[cot_to_final].total
    assert source_stat is not None
    assert final_stat is not None
    assert source_stat.target_count == 2
    assert final_stat.target_count == 1
    assert source_stat.gain == 2.0
    assert final_stat.gain == 4.0


@pytest.mark.parametrize("scorer_type", (S2STokenScorer, UniSSTokenScorer))
def test_condition_gain_uses_one_scorer_path_without_model_dispatch(scorer_type) -> None:
    scorer = scorer_type()

    result = condition_gain(
        scorer,
        (
            _gain_probe(
                target_modalities=(Modality.TEXT, Modality.AUDIO),
            ),
        ),
        bootstrap_samples=0,
    )[0]

    assert len(scorer.calls) == 2
    assert result.total is not None
    assert result.total.gain == 0.0


def test_condition_gain_metrics_include_explicit_probe_set_identity() -> None:
    result = condition_gain(
        _ScriptedScorer((1.0, 2.0)),
        (_gain_probe(),),
        bootstrap_samples=0,
    )[0]

    metrics = condition_gain_metrics(result, probe_set_id="core_v1")

    assert "condition_gain/core_v1/tts/direct/source_text/final/gain" in metrics


def test_condition_gain_reports_explicit_exclusions_as_coverage() -> None:
    eligible = _gain_probe(name="eligible")
    excluded = ConditionGainProbe.excluded(
        "excluded",
        eligible.key,
        "missing_aligned_donor",
    )

    result = condition_gain(
        _ScriptedScorer((1.0, 2.0)),
        (eligible, excluded),
        bootstrap_samples=0,
    )[0]
    metrics = condition_gain_metrics(result)

    assert result.eligible_count == 1
    assert result.excluded_count == 1
    assert metrics["condition_gain/tts/direct/source_text/final/coverage"] == 0.5


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


def test_condition_gain_callback_scores_on_every_rank_and_logs_only_global_zero() -> None:
    calls: list[int] = []
    writer = _Writer()
    for is_global_zero in (False, True):
        scorer = _ScriptedScorer((1.0, 2.0))
        module = _GainModule(scorer)
        callback = ConditionGainCallback(
            (_gain_probe(),),
            probe_set_id="core_v1",
            every_n_steps=1,
            bootstrap_samples=0,
        )
        callback.on_train_batch_end(
            _trainer(
                writer if is_global_zero else None,
                step=1,
                world_size=2,
                is_global_zero=is_global_zero,
                strategy_model=module,
            ),
            module,
            None,
            None,
            0,
        )
        calls.append(len(scorer.calls))

    assert calls == [2, 2]
    assert writer.values[
        "condition_gain/core_v1/tts/direct/source_text/final/gain"
    ] == pytest.approx(1.0)


def test_condition_gain_callback_requires_a_scalar_logger_on_global_zero() -> None:
    scorer = _ScriptedScorer((1.0, 2.0))
    module = _GainModule(scorer)
    callback = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=1,
        bootstrap_samples=0,
    )

    with pytest.raises(RuntimeError, match="add_scalar"):
        callback.on_train_batch_end(
            _trainer(None, step=1, strategy_model=module),
            module,
            None,
            None,
            0,
        )

    assert len(scorer.calls) == 2


def test_condition_gain_callback_requires_a_strategy_model() -> None:
    module = _GainModule(_ScriptedScorer((1.0, 2.0)))
    callback = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=1,
    )

    with pytest.raises(RuntimeError, match="trainer.strategy.model"):
        callback.on_train_batch_end(
            _trainer(_Writer(), step=1, strategy_model=object()),
            module,
            None,
            None,
            0,
        )


def test_condition_gain_callback_rejects_an_invalid_forward_result() -> None:
    invalid = cast(TokenScorer, lambda _input, _targets: object())
    module = _GainModule(invalid)
    callback = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=1,
        bootstrap_samples=0,
    )

    with pytest.raises(TypeError, match="forward must return TokenLossStats"):
        callback.on_train_batch_end(
            _trainer(_Writer(), step=1, strategy_model=module),
            module,
            None,
            None,
            0,
        )


def test_condition_gain_callback_restores_cadence_state() -> None:
    first_scorer = _ScriptedScorer((1.0, 2.0))
    first_module = _GainModule(first_scorer)
    first = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=2,
        bootstrap_samples=0,
    )
    first.on_train_batch_end(
        _trainer(_Writer(), step=2, strategy_model=first_module),
        first_module,
        None,
        None,
        0,
    )

    second_scorer = _ScriptedScorer((3.0, 5.0))
    second = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=2,
        bootstrap_samples=0,
    )
    second.load_state_dict(first.state_dict())
    module = _GainModule(second_scorer)
    for step in (2, 3, 4):
        second.on_train_batch_end(
            _trainer(_Writer(), step=step, strategy_model=module),
            module,
            None,
            None,
            0,
        )

    assert len(first_scorer.calls) == 2
    assert len(second_scorer.calls) == 2


def test_condition_gain_callback_preserves_modes_rng_and_precision_context() -> None:
    events: list[str] = []
    precision = _Precision(events)
    scorer = _ScriptedScorer(
        (1.0, 2.0),
        events=events,
        advance_rng=True,
    )
    module = _GainModule(scorer)
    modes = tuple(child.training for child in module.modules())
    callback = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=1,
        bootstrap_samples=0,
    )
    torch.manual_seed(29)
    rng_state = torch.random.get_rng_state()

    callback.on_train_batch_end(
        _trainer(
            _Writer(),
            step=1,
            precision_plugin=precision,
            strategy_model=module,
        ),
        module,
        None,
        None,
        0,
    )

    assert tuple(child.training for child in module.modules()) == modes
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert events == ["enter", "score", "score", "exit"]


def test_condition_gain_callback_calls_the_strategy_wrapped_model() -> None:
    wrapped_scorer = _ScriptedScorer((1.0, 2.0))
    wrapped = _GainModule(wrapped_scorer)
    pure_scorer = _ScriptedScorer(())
    pure = _GainModule(pure_scorer)
    callback = ConditionGainCallback(
        (_gain_probe(),),
        probe_set_id="core_v1",
        every_n_steps=1,
        bootstrap_samples=0,
    )
    callback.on_train_batch_end(
        _trainer(_Writer(), step=1, strategy_model=wrapped),
        pure,
        None,
        None,
        0,
    )

    assert len(wrapped_scorer.calls) == 2
    assert pure_scorer.calls == []


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


def _gain_batch(
    *,
    route: Route | None = None,
    sample_id: str | None = "sample_0",
    target_positions: tuple[int, ...] = (2, 3),
    target_tokens: tuple[int, ...] = (51, 52),
    target_input_tokens: tuple[int, ...] | None = None,
    target_modalities: tuple[Modality, ...] | None = None,
    sequence_length: int = 4,
    prompt_length: int = 2,
    condition_token: int = 11,
) -> TokenBatch:
    input_tokens = target_tokens if target_input_tokens is None else target_input_tokens
    modalities = (
        (Modality.TEXT,) * len(target_tokens) if target_modalities is None else target_modalities
    )
    if not (len(target_positions) == len(target_tokens) == len(input_tokens) == len(modalities)):
        raise ValueError("gain test target fields must align.")
    input_ids = torch.arange(10, 10 + sequence_length, dtype=torch.int64).unsqueeze(0)
    input_ids[0, 0] = condition_token
    labels = torch.full_like(input_ids, -100)
    text_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    audio_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for position, input_token, target_token, modality in zip(
        target_positions,
        input_tokens,
        target_tokens,
        modalities,
        strict=True,
    ):
        input_ids[0, position] = input_token
        labels[0, position] = target_token
        target_mask = text_mask if modality is Modality.TEXT else audio_mask
        target_mask[0, position] = True
    return TokenBatch(
        input=TokenInput(input_ids, torch.ones_like(input_ids, dtype=torch.bool)),
        targets=TokenTargets(
            labels,
            text_mask=text_mask if bool(text_mask.any()) else None,
            audio_mask=audio_mask if bool(audio_mask.any()) else None,
        ),
        prompt_lengths=torch.tensor([prompt_length]),
        routes=(Route("tts") if route is None else route,),
        sample_ids=None if sample_id is None else (sample_id,),
    )


def _gain_probe(
    *,
    name: str = "sample_0",
    key: ConditionGainKey | None = None,
    target_positions: tuple[int, ...] = (2, 3),
    target_tokens: tuple[int, ...] = (51, 52),
    target_modalities: tuple[Modality, ...] | None = None,
    sequence_length: int = 4,
    prompt_length: int = 2,
) -> ConditionGainProbe:
    resolved_key = (
        ConditionGainKey(Route("tts"), "source_text", "final")
        if key is None
        else key
    )
    common = {
        "route": resolved_key.route,
        "target_positions": target_positions,
        "target_tokens": target_tokens,
        "target_modalities": target_modalities,
        "sequence_length": sequence_length,
        "prompt_length": prompt_length,
    }
    return ConditionGainProbe(
        name,
        resolved_key,
        _gain_batch(**common, condition_token=11),
        _gain_batch(**common, condition_token=19),
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


def _trainer(
    writer: _Writer | None,
    *,
    step: int,
    world_size: int = 1,
    is_global_zero: bool = True,
    precision_plugin: _Precision | None = None,
    strategy_model: object | None = None,
) -> Trainer:
    logger = None if writer is None else SimpleNamespace(experiment=writer)
    return cast(
        Trainer,
        SimpleNamespace(
            global_step=step,
            world_size=world_size,
            is_global_zero=is_global_zero,
            logger=logger,
            loggers=(),
            precision_plugin=(
                _Precision() if precision_plugin is None else precision_plugin
            ),
            strategy=SimpleNamespace(model=strategy_model),
        ),
    )


def _pl_module() -> LightningModule:
    return cast(LightningModule, SimpleNamespace())
