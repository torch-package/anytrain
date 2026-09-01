from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from anytrain.lightning import StepInterval, _capability, experiment
from anytrain.lightning.oom import annotate

from .data import Modality, Route, TokenBatch
from .model import LossStat, TokenScorer, score_tokens

_TAG_SEGMENT = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, slots=True)
class ConditionGainKey:
    """One measured dependency, independent of a downstream task enum."""

    route: Route
    intervention: str
    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.route, Route):
            raise TypeError("condition gain route must be Route.")
        _tag_segment(self.intervention, "condition gain intervention")
        _tag_segment(self.target, "condition gain target")

    @property
    def tag(self) -> str:
        return f"{self.route.tag}/{self.intervention}/{self.target}"


@dataclass(eq=False, frozen=True, slots=True)
class ConditionGainProbe:
    """One row-aligned matched/counterfactual pair, or one explicit exclusion."""

    name: str
    key: ConditionGainKey
    matched: TokenBatch | None
    counterfactual: TokenBatch | None
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "condition gain probe name")
        if not isinstance(self.key, ConditionGainKey):
            raise TypeError("condition gain probe key must be ConditionGainKey.")
        paired = self.matched is not None and self.counterfactual is not None
        excluded = self.matched is None and self.counterfactual is None
        if not paired and not excluded:
            raise ValueError("condition gain probe must contain both cases or exclude both cases.")
        if excluded:
            if self.exclusion_reason is None:
                raise ValueError("excluded condition gain probes require a reason.")
            _non_empty_string(
                self.exclusion_reason,
                "condition gain exclusion_reason",
            )
            return
        if self.exclusion_reason is not None:
            raise ValueError("eligible condition gain probes cannot have an exclusion reason.")
        if not isinstance(self.matched, TokenBatch) or not isinstance(
            self.counterfactual,
            TokenBatch,
        ):
            raise TypeError("condition gain cases must be TokenBatch values.")
        _case(self.matched, self.key, "matched")
        _case(self.counterfactual, self.key, "counterfactual")
        if self.matched.sample_ids != self.counterfactual.sample_ids:
            raise ValueError("condition gain cases must preserve identical sample_ids lineage.")
        if self.matched.targets.modalities != self.counterfactual.targets.modalities:
            raise ValueError("condition gain cases must supervise the same modalities.")
        if _cpu_targets(self.matched, self.counterfactual) and not torch.equal(
            _target_stream(self.matched),
            _target_stream(self.counterfactual),
        ):
            raise ValueError(
                "condition gain cases must preserve the ordered teacher-forced "
                "target token/modality stream."
            )

    @classmethod
    def excluded(
        cls,
        name: str,
        key: ConditionGainKey,
        reason: str,
    ) -> ConditionGainProbe:
        return cls(name, key, None, None, reason)

    @property
    def eligible(self) -> bool:
        return self.matched is not None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ConditionGainProbe:
        if not self.eligible:
            return self
        if self.matched is None or self.counterfactual is None:
            raise RuntimeError("eligible condition gain probe is missing a case.")
        return ConditionGainProbe(
            self.name,
            self.key,
            self.matched.to(device, non_blocking=non_blocking),
            self.counterfactual.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class ConditionGainStat:
    matched_loss: float
    counterfactual_loss: float
    target_count: int
    bootstrap_low: float | None = None
    bootstrap_high: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("matched_loss", self.matched_loss),
            ("counterfactual_loss", self.counterfactual_loss),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"condition gain {name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"condition gain {name} must be finite.")
        if isinstance(self.target_count, bool) or not isinstance(self.target_count, int):
            raise TypeError("condition gain target_count must be an integer.")
        if self.target_count <= 0:
            raise ValueError("condition gain target_count must be positive.")
        if (self.bootstrap_low is None) != (self.bootstrap_high is None):
            raise ValueError("condition gain bootstrap bounds must both be present or absent.")
        for name, value in (
            ("bootstrap_low", self.bootstrap_low),
            ("bootstrap_high", self.bootstrap_high),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"condition gain {name} must be a real number or None.")
            if not math.isfinite(float(value)):
                raise ValueError(f"condition gain {name} must be finite or None.")
        if (
            self.bootstrap_low is not None
            and self.bootstrap_high is not None
            and self.bootstrap_low > self.bootstrap_high
        ):
            raise ValueError("condition gain bootstrap bounds must be ordered.")

    @property
    def gain(self) -> float:
        return self.counterfactual_loss - self.matched_loss


@dataclass(frozen=True, slots=True)
class ConditionGainResult:
    key: ConditionGainKey
    eligible_count: int
    excluded_count: int
    total: ConditionGainStat | None = None
    text: ConditionGainStat | None = None
    audio: ConditionGainStat | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ConditionGainKey):
            raise TypeError("condition gain result key must be ConditionGainKey.")
        for name, value in (
            ("eligible_count", self.eligible_count),
            ("excluded_count", self.excluded_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"condition gain {name} must be an integer.")
            if value < 0:
                raise ValueError(f"condition gain {name} must be non-negative.")
        if self.eligible_count + self.excluded_count <= 0:
            raise ValueError("condition gain result must cover at least one probe.")
        for name, value in (
            ("total", self.total),
            ("text", self.text),
            ("audio", self.audio),
        ):
            if value is not None and not isinstance(value, ConditionGainStat):
                raise TypeError(f"condition gain {name} must be ConditionGainStat or None.")
        if self.eligible_count > 0 and self.total is None:
            raise ValueError("eligible condition gain results require total statistics.")
        if self.eligible_count == 0 and any(
            value is not None for value in (self.total, self.text, self.audio)
        ):
            raise ValueError("fully excluded condition gain results cannot contain statistics.")

    def for_modality(self, modality: Modality) -> ConditionGainStat | None:
        if modality is Modality.TEXT:
            return self.text
        if modality is Modality.AUDIO:
            return self.audio
        raise TypeError("modality must be a Modality.")


@dataclass(frozen=True, slots=True)
class _Observation:
    matched_sum: float
    counterfactual_sum: float
    count: int


@dataclass(slots=True)
class _Group:
    eligible_count: int = 0
    excluded_count: int = 0
    modalities: frozenset[Modality] | None = None
    values: dict[Modality | None, list[_Observation]] = field(default_factory=dict)


@torch.no_grad()
def condition_gain(
    provider: TokenScorer,
    probes: Sequence[ConditionGainProbe],
    *,
    bootstrap_samples: int = 1_000,
    bootstrap_seed: int = 0,
) -> tuple[ConditionGainResult, ...]:
    """Measure counterfactual-minus-matched loss for explicit fixed probes."""

    values = _probes(probes)
    _non_negative_int(bootstrap_samples, "condition gain bootstrap_samples")
    _non_negative_int(bootstrap_seed, "condition gain bootstrap_seed")
    if not isinstance(provider, TokenScorer):
        raise TypeError("condition gain provider must implement TokenScorer.")

    groups: dict[ConditionGainKey, _Group] = {}
    for probe in values:
        group = groups.setdefault(probe.key, _Group())
        if not probe.eligible:
            group.excluded_count += 1
            continue
        if probe.matched is None or probe.counterfactual is None:
            raise RuntimeError("eligible condition gain probe is missing a case.")
        matched = score_tokens(provider, probe.matched)
        counterfactual = score_tokens(provider, probe.counterfactual)
        if matched.modalities != counterfactual.modalities:
            raise ValueError("condition gain scorer returned different case modalities.")
        if group.modalities is None:
            group.modalities = matched.modalities
        elif group.modalities != matched.modalities:
            raise ValueError("condition gain probes sharing a key must return the same modalities.")
        group.eligible_count += 1
        group.values.setdefault(None, []).append(
            _observation(
                matched.total,
                counterfactual.total,
                matched_target_count=_target_count(probe.matched),
                counterfactual_target_count=_target_count(probe.counterfactual),
                name="total",
            )
        )
        for modality in matched.modalities:
            matched_stat = matched.for_modality(modality)
            counterfactual_stat = counterfactual.for_modality(modality)
            if matched_stat is None or counterfactual_stat is None:
                raise RuntimeError("condition gain modality statistics are unavailable.")
            group.values.setdefault(modality, []).append(
                _observation(
                    matched_stat,
                    counterfactual_stat,
                    matched_target_count=_target_count(probe.matched, modality),
                    counterfactual_target_count=_target_count(
                        probe.counterfactual,
                        modality,
                    ),
                    name=modality.value,
                )
            )

    results: list[ConditionGainResult] = []
    for key in sorted(groups, key=lambda value: value.tag):
        group = groups[key]
        results.append(
            ConditionGainResult(
                key=key,
                eligible_count=group.eligible_count,
                excluded_count=group.excluded_count,
                total=_gain_stat(
                    group.values.get(None, ()),
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=_derived_seed(bootstrap_seed, key.tag, "total"),
                ),
                text=_gain_stat(
                    group.values.get(Modality.TEXT, ()),
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=_derived_seed(bootstrap_seed, key.tag, "text"),
                ),
                audio=_gain_stat(
                    group.values.get(Modality.AUDIO, ()),
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=_derived_seed(bootstrap_seed, key.tag, "audio"),
                ),
            )
        )
    return tuple(results)


def condition_gain_metrics(
    result: ConditionGainResult,
    *,
    probe_set_id: str | None = None,
) -> dict[str, float]:
    if not isinstance(result, ConditionGainResult):
        raise TypeError("condition gain metrics require ConditionGainResult.")
    if probe_set_id is not None:
        _tag_segment(probe_set_id, "condition gain probe_set_id")
    set_tag = "" if probe_set_id is None else f"/{probe_set_id}"
    prefix = f"condition_gain{set_tag}/{result.key.tag}"
    attempted = result.eligible_count + result.excluded_count
    metrics = {
        f"{prefix}/eligible_count": float(result.eligible_count),
        f"{prefix}/excluded_count": float(result.excluded_count),
        f"{prefix}/coverage": result.eligible_count / attempted,
    }
    for name, stat in (
        (None, result.total),
        (Modality.TEXT.value, result.text),
        (Modality.AUDIO.value, result.audio),
    ):
        if stat is None:
            continue
        tag = prefix if name is None else f"{prefix}/{name}"
        metrics[f"{tag}/matched_loss"] = stat.matched_loss
        metrics[f"{tag}/counterfactual_loss"] = stat.counterfactual_loss
        metrics[f"{tag}/gain"] = stat.gain
        metrics[f"{tag}/target_tokens"] = float(stat.target_count)
        if stat.bootstrap_low is not None and stat.bootstrap_high is not None:
            metrics[f"{tag}/bootstrap_low"] = stat.bootstrap_low
            metrics[f"{tag}/bootstrap_high"] = stat.bootstrap_high
    return metrics


class ConditionGainCallback(Callback):
    """Periodically score fixed interventions through the strategy-wrapped model."""

    def __init__(
        self,
        probes: Sequence[ConditionGainProbe],
        *,
        probe_set_id: str,
        every_n_steps: int,
        bootstrap_samples: int = 1_000,
        bootstrap_seed: int = 0,
    ) -> None:
        super().__init__()
        self.probes = _probes(probes)
        _tag_segment(probe_set_id, "condition gain probe_set_id")
        _non_negative_int(bootstrap_samples, "condition gain bootstrap_samples")
        _non_negative_int(bootstrap_seed, "condition gain bootstrap_seed")
        self.probe_set_id = probe_set_id
        self.interval = StepInterval(every_n_steps=every_n_steps)
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            probe_set_id=self.probe_set_id,
            every_n_steps=self.interval.every_n_steps,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_if_due(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        self._run_if_due(trainer, pl_module)

    def state_dict(self) -> dict[str, dict[str, int | None]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if set(state_dict) != {"interval"}:
            raise ValueError("condition gain callback state must contain only interval.")
        interval = state_dict["interval"]
        if not isinstance(interval, Mapping):
            raise TypeError("condition gain interval state must be a mapping.")
        self.interval.load_state_dict(interval)

    def _run_if_due(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = _capability.global_step(trainer)
        if not self.interval.should_run(step):
            return
        device = getattr(pl_module, "device", None)
        if not isinstance(device, torch.device):
            raise TypeError("condition gain callback requires pl_module.device.")
        model = trainer.strategy.model
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("condition gain callback requires trainer.strategy.model.")
        provider = cast(TokenScorer, model)
        module_modes = tuple((module, module.training) for module in model.modules())
        rng_devices = [] if device.type == "cpu" else [device]
        rng_device_type = "cuda" if device.type == "cpu" else device.type
        try:
            with torch.random.fork_rng(
                devices=rng_devices,
                device_type=rng_device_type,
            ):
                model.eval()
                probes = tuple(probe.to(device) for probe in self.probes)
                with trainer.precision_plugin.forward_context():
                    results = condition_gain(
                        provider,
                        probes,
                        bootstrap_samples=self.bootstrap_samples,
                        bootstrap_seed=self.bootstrap_seed,
                    )
        except torch.OutOfMemoryError as error:
            annotate(
                error,
                phase="condition_gain",
                inputs={
                    "probe_set_id": self.probe_set_id,
                    "probe_count": len(self.probes),
                },
            )
            raise
        finally:
            for module, training in module_modes:
                module.training = training
        if not _capability.is_global_zero(trainer):
            return
        writer = experiment.scalar(trainer)
        if writer is None:
            raise RuntimeError(
                "ConditionGainCallback requires a logger with add_scalar capability."
            )
        for result in results:
            for tag, value in condition_gain_metrics(
                result,
                probe_set_id=self.probe_set_id,
            ).items():
                writer.add_scalar(tag, value, step)


def _case(batch: TokenBatch, key: ConditionGainKey, name: str) -> None:
    if batch.batch_size != 1:
        raise ValueError(f"condition gain {name} case must contain exactly one row.")
    if batch.routes != (key.route,):
        raise ValueError(f"condition gain {name} route must match the probe key.")


def _cpu_targets(first: TokenBatch, second: TokenBatch) -> bool:
    tensors = [first.targets.labels, second.targets.labels]
    for batch in (first, second):
        tensors.extend(
            mask for mask in (batch.targets.text_mask, batch.targets.audio_mask) if mask is not None
        )
    return all(value.device.type == "cpu" for value in tensors)


def _target_stream(batch: TokenBatch) -> torch.Tensor:
    selected = torch.zeros_like(batch.targets.labels, dtype=torch.bool)
    modalities = torch.full_like(batch.targets.labels, -1)
    for index, modality in enumerate(Modality):
        mask = batch.targets.mask(modality)
        if mask is None:
            continue
        selected |= mask
        modalities.masked_fill_(mask, index)
    return torch.stack(
        (
            modalities.masked_select(selected),
            batch.input.input_ids.masked_select(selected),
            batch.targets.labels.masked_select(selected),
        ),
        dim=1,
    )


def _probes(probes: Sequence[ConditionGainProbe]) -> tuple[ConditionGainProbe, ...]:
    if isinstance(probes, (str, bytes)) or not isinstance(probes, Sequence):
        raise TypeError("condition gain probes must be a sequence.")
    values = tuple(probes)
    if not values:
        raise ValueError("condition gain probes must not be empty.")
    if any(not isinstance(probe, ConditionGainProbe) for probe in values):
        raise TypeError("condition gain probes must contain ConditionGainProbe values.")
    names = tuple(probe.name for probe in values)
    if len(names) != len(set(names)):
        raise ValueError("condition gain probe names must be unique.")
    return values


def _target_count(batch: TokenBatch, modality: Modality | None = None) -> int:
    if modality is not None:
        mask = batch.targets.mask(modality)
        if mask is None:
            raise RuntimeError("condition gain target mask is unavailable.")
        return int(mask.detach().sum().item())
    return sum(_target_count(batch, value) for value in batch.targets.modalities)


def _observation(
    matched: LossStat,
    counterfactual: LossStat,
    *,
    matched_target_count: int,
    counterfactual_target_count: int,
    name: str,
) -> _Observation:
    matched_count = int(matched.count.detach().item())
    counterfactual_count = int(counterfactual.count.detach().item())
    if matched_target_count != counterfactual_target_count:
        raise ValueError(f"condition gain {name} target masks must select identical token counts.")
    if matched_count != matched_target_count or counterfactual_count != counterfactual_target_count:
        raise ValueError(f"condition gain {name} scorer counts must exactly match target masks.")
    matched_sum = float(matched.loss_sum.detach().item())
    counterfactual_sum = float(counterfactual.loss_sum.detach().item())
    if not math.isfinite(matched_sum) or not math.isfinite(counterfactual_sum):
        raise ValueError("condition gain case loss sums must be finite.")
    return _Observation(matched_sum, counterfactual_sum, matched_count)


def _gain_stat(
    values: Sequence[_Observation],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> ConditionGainStat | None:
    if not values:
        return None
    matched_sum = sum(value.matched_sum for value in values)
    counterfactual_sum = sum(value.counterfactual_sum for value in values)
    count = sum(value.count for value in values)
    matched_loss = matched_sum / count
    counterfactual_loss = counterfactual_sum / count
    low, high = _bootstrap_interval(
        values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return ConditionGainStat(
        matched_loss=matched_loss,
        counterfactual_loss=counterfactual_loss,
        target_count=count,
        bootstrap_low=low,
        bootstrap_high=high,
    )


def _bootstrap_interval(
    values: Sequence[_Observation],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if samples == 0:
        return None, None
    if len(values) == 1:
        value = values[0]
        gain = (value.counterfactual_sum - value.matched_sum) / value.count
        return gain, gain
    matched = torch.tensor(
        [value.matched_sum for value in values],
        dtype=torch.float64,
    )
    counterfactual = torch.tensor(
        [value.counterfactual_sum for value in values],
        dtype=torch.float64,
    )
    counts = torch.tensor([value.count for value in values], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0,
        len(values),
        (samples, len(values)),
        generator=generator,
    )
    selected_counts = counts[indices].sum(dim=1)
    gains = (counterfactual[indices].sum(dim=1) - matched[indices].sum(dim=1)) / selected_counts
    quantiles = torch.quantile(
        gains,
        torch.tensor([0.025, 0.975], dtype=gains.dtype, device=gains.device),
    )
    return float(quantiles[0]), float(quantiles[1])


def _derived_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2**63)


def _tag_segment(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if _TAG_SEGMENT.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_TAG_SEGMENT.pattern!r}.")


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "ConditionGainCallback",
    "ConditionGainKey",
    "ConditionGainProbe",
    "ConditionGainResult",
    "ConditionGainStat",
    "condition_gain",
    "condition_gain_metrics",
]
