from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from anytrain import observation
from anytrain._buffer import register_buffer

DirichletStrategy = Literal["std", "var", "range"]


@dataclass(frozen=True)
class ADTConfig:
    num_experts: int
    dispersion_strategy: DirichletStrategy = "std"
    stat_ema_decays: tuple[float, float, float] = (0.9, 0.999, 0.9)
    prior_strength: float | None = None
    minka_refinement_iters: int | None = None
    use_gumbel_softmax: bool = False
    temperature_groups: int = 1
    warmup_steps: int = 0
    min_temperature: float | None = None
    max_temperature: float | None = None
    temperature_smoothing_decay: float | None = None
    sync_distributed_stats: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stat_ema_decays, tuple):
            raise TypeError("stat_ema_decays must be a tuple.")
        for name, value in (
            ("use_gumbel_softmax", self.use_gumbel_softmax),
            ("sync_distributed_stats", self.sync_distributed_stats),
        ):
            _require_bool(name, value)
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}.")
        if self.temperature_groups <= 0:
            raise ValueError(
                f"temperature_groups must be positive, got {self.temperature_groups}."
            )
        if self.num_experts % self.temperature_groups != 0:
            raise ValueError(
                "num_experts must be divisible by temperature_groups: "
                f"got num_experts={self.num_experts}, "
                f"temperature_groups={self.temperature_groups}."
            )
        if len(self.stat_ema_decays) != 3:
            raise ValueError(
                "stat_ema_decays must contain exactly 3 values, "
                f"got {len(self.stat_ema_decays)}."
            )
        for decay in self.stat_ema_decays:
            if not 0 <= decay < 1:
                raise ValueError(f"each stat EMA decay must be in [0, 1), got {decay}.")
        if self.prior_strength is not None and self.prior_strength < 0:
            raise ValueError(f"prior_strength must be non-negative, got {self.prior_strength}.")
        if self.minka_refinement_iters is not None and self.minka_refinement_iters <= 0:
            raise ValueError(
                "minka_refinement_iters must be positive, "
                f"got {self.minka_refinement_iters}."
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}."
            )
        if self.min_temperature is not None and self.min_temperature <= 0:
            raise ValueError(
                f"min_temperature must be positive, got {self.min_temperature}."
            )
        if self.max_temperature is not None and self.max_temperature <= 0:
            raise ValueError(
                f"max_temperature must be positive, got {self.max_temperature}."
            )
        if (
            self.min_temperature is not None
            and self.max_temperature is not None
            and self.min_temperature > self.max_temperature
        ):
            raise ValueError(
                "min_temperature must be less than or equal to max_temperature: "
                f"got min_temperature={self.min_temperature}, "
                f"max_temperature={self.max_temperature}."
            )
        if (
            self.temperature_smoothing_decay is not None
            and not 0 <= self.temperature_smoothing_decay < 1
        ):
            raise ValueError(
                "temperature_smoothing_decay must be in [0, 1), "
                f"got {self.temperature_smoothing_decay}."
            )
        if self.dispersion_strategy not in {"std", "var", "range"}:
            raise ValueError(
                f"Unsupported ADT dispersion strategy {self.dispersion_strategy!r}."
            )


@dataclass(eq=False)
class ADTStats:
    """Runtime utilization statistics for Adaptive Dirichlet Tempering."""

    expert_utilization: Tensor
    expert_vars: Tensor
    alpha: Tensor
    temperature: Tensor

    @property
    def expert_means(self) -> Tensor:
        """Backward-compatible alias for running expert utilization."""

        return self.expert_utilization

    @property
    def effective_experts(self) -> Tensor:
        """Entropy perplexity of the running expert utilization distribution."""

        utilization = self.expert_utilization.to(dtype=torch.float32)
        safe_utilization = utilization.clamp_min(torch.finfo(utilization.dtype).tiny)
        terms = torch.where(
            utilization > 0,
            utilization * safe_utilization.log(),
            torch.zeros_like(utilization),
        )
        entropy = -terms.sum()
        return entropy.exp()


def adt_stats_metrics(
    stats: ADTStats,
    *,
    metrics: Iterable[str] = ("variance", "alpha", "temperature"),
) -> dict[str, Tensor]:
    """Flatten ADT-only stats into compact scalar summaries.

    ADT-specific per-expert vectors are summarized as {group}_mean,
    {group}_std, {group}_min and {group}_max so scalar dashboards stay stable as
    the number of experts grows. Gate/routing utilization belongs to MoE/gate
    logging, not ADT stats metrics.
    """

    enabled = _adt_metric_groups(metrics)
    values: dict[str, Tensor] = {}
    if "variance" in enabled:
        values.update(_summary_metrics(stats.expert_vars, "variance"))
    if "alpha" in enabled:
        values.update(_summary_metrics(stats.alpha, "alpha"))
    if "temperature" in enabled:
        values.update(_summary_metrics(stats.temperature, "temperature"))
    return values


class AdaptiveDirichletTempering(nn.Module):
    """Adaptive temperature scaling for mixture-of-experts router logits.

    ``forward`` accepts router logits with ``num_experts`` on the last dimension
    and returns same-shaped expert weights.  ADT is intentionally a narrow
    logits calibration and utilization-statistics module: it does not select
    experts, compute balancing losses, or perform task/framework logging.
    """

    _eps = 1e-6
    # Implementation knobs below are intentionally not constructor arguments.
    # Keep them as class defaults for now; promote to public ``__init__`` kwargs
    # only when a caller needs to tune them in production.
    _dispersion_strategy: DirichletStrategy = "std"
    _stat_ema_decays: tuple[float, float, float] = (0.9, 0.999, 0.9)
    _prior_strength: float | None = None
    _minka_refinement_iters: int | None = None
    _use_gumbel_softmax: bool = False
    _temperature_groups: int = 1
    _temperature_smoothing_decay: float | None = None
    # Local stats only, like ``nn.BatchNorm*``. Cross-rank sync stays opt-in via
    # a future SyncBatchNorm-style API rather than a silent auto switch.
    _sync_distributed_stats: bool = False

    _disabled: bool
    _stats_frozen: bool
    _expert_means: Tensor
    _expert_means_square: Tensor
    _expert_logs: Tensor
    _temperature_ema: Tensor
    _num_updates: Tensor
    _num_updates_value: int
    prior_mean: Tensor
    prior_var: Tensor
    prior_log: Tensor

    def __init__(
        self,
        num_experts: int,
        *,
        warmup_steps: int = 0,
        min_temperature: float | None = None,
        max_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.config = ADTConfig(
            num_experts=num_experts,
            dispersion_strategy=self._dispersion_strategy,
            stat_ema_decays=self._stat_ema_decays,
            prior_strength=self._prior_strength,
            minka_refinement_iters=self._minka_refinement_iters,
            use_gumbel_softmax=self._use_gumbel_softmax,
            temperature_groups=self._temperature_groups,
            warmup_steps=warmup_steps,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            temperature_smoothing_decay=self._temperature_smoothing_decay,
            sync_distributed_stats=self._sync_distributed_stats,
        )

        register_buffer(self, "_expert_means", self._initial_expert_means())
        register_buffer(self, "_expert_means_square", self._initial_expert_means().pow(2))
        register_buffer(self, "_expert_logs", torch.zeros(self.config.num_experts))

        prior_mean, prior_var, prior_log = self._initial_priors()
        register_buffer(self, "prior_mean", prior_mean)
        register_buffer(self, "prior_var", prior_var)
        register_buffer(self, "prior_log", prior_log)

        register_buffer(self, "_temperature_ema", torch.ones(self.config.num_experts))
        register_buffer(self, "_num_updates", torch.zeros((), dtype=torch.long))
        self._num_updates_value = 0

        self._disabled = False
        self._stats_frozen = False
        observation.registry.recommend(self)

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def disable(self) -> AdaptiveDirichletTempering:
        self._disabled = True
        return self

    def enable(self) -> AdaptiveDirichletTempering:
        self._disabled = False
        return self

    def freeze_stats(self) -> AdaptiveDirichletTempering:
        self._stats_frozen = True
        return self

    def unfreeze_stats(self) -> AdaptiveDirichletTempering:
        self._stats_frozen = False
        return self

    @torch.no_grad()
    def reset_stats(self) -> AdaptiveDirichletTempering:
        means = self._initial_expert_means(device=self._expert_means.device, dtype=self._expert_means.dtype)
        self._expert_means.copy_(means)
        self._expert_means_square.copy_(means.pow(2))
        self._expert_logs.zero_()
        self._temperature_ema.fill_(1)
        self._num_updates.zero_()
        self._num_updates_value = 0
        return self

    @property
    def num_updates(self) -> int:
        return self._num_updates_value

    @property
    def expert_means(self) -> Tensor:
        with torch.no_grad():
            return self._with_prior(self._n_eff(0), self._expert_means, self.prior_mean)

    @property
    def expert_utilization(self) -> Tensor:
        """Running mean router probability for each expert."""
        return self.expert_means

    @property
    def expert_vars(self) -> Tensor:
        with torch.no_grad():
            variance = self._expert_means_square - self._expert_means.pow(2)
            variance = variance.clamp_min(0)
            return self._with_prior(self._n_eff(1), variance, self.prior_var)

    @property
    def expert_logs(self) -> Tensor:
        with torch.no_grad():
            return self._with_prior(self._n_eff(2), self._expert_logs, self.prior_log)

    @property
    def alpha(self) -> Tensor:
        with torch.no_grad():
            numerator = self.expert_means * (1 - self.expert_means)
            denominator = self.expert_vars + self._eps
            alpha = numerator / denominator - 1
            if self.config.minka_refinement_iters is None:
                return alpha

            grouped_alpha = alpha.view(self.config.temperature_groups, -1)
            grouped_logs = self.expert_logs.view(self.config.temperature_groups, -1)
            refined = refine_alpha_minka(
                grouped_alpha,
                grouped_logs,
                self.config.minka_refinement_iters,
            )
            return refined.reshape(-1)

    @property
    def temperature(self) -> Tensor:
        if self.config.temperature_smoothing_decay is not None:
            return self._temperature_ema.clone()
        return self._raw_temperature()

    @property
    def effective_experts(self) -> Tensor:
        """Entropy perplexity of the running expert utilization distribution."""

        return self.stats().effective_experts

    def stats(self) -> ADTStats:
        return ADTStats(
            expert_utilization=self.expert_utilization,
            expert_vars=self.expert_vars,
            alpha=self.alpha,
            temperature=self.temperature,
        )

    def diagnostics(self) -> dict[str, Tensor]:
        return adt_stats_metrics(self.stats())

    @torch.no_grad()
    def update_stats(self, logits: Tensor, *, mask: Tensor | None = None) -> None:
        if self._stats_frozen:
            return

        self._validate_logits(logits)
        sums, square_sums, log_sums, count, local_count = self._router_stat_sums(
            logits,
            mask=mask,
        )
        if self.config.sync_distributed_stats:
            sums, square_sums, log_sums, count = self._sync_stat_sums(
                sums,
                square_sums,
                log_sums,
                count,
            )
            if count.item() == 0:
                return
        elif local_count == 0:
            return

        means = sums / count
        means_square = square_sums / count

        self._expert_means.copy_(
            ema_update(self._expert_means, means, decay=self.config.stat_ema_decays[0])
        )
        self._expert_means_square.copy_(
            ema_update(
                self._expert_means_square,
                means_square,
                decay=self.config.stat_ema_decays[1],
            )
        )

        if self.config.minka_refinement_iters is not None:
            log_means = log_sums / count
            self._expert_logs.copy_(
                ema_update(
                    self._expert_logs,
                    log_means,
                    decay=self.config.stat_ema_decays[2],
                )
            )

        self._num_updates.add_(1)
        self._num_updates_value += 1

    def forward(
        self,
        logits: Tensor,
        *,
        mask: Tensor | None = None,
        collect_stats: bool | None = None,
    ) -> Tensor:
        """Convert router logits to calibrated expert weights.

        Args:
            logits: Tensor shaped ``(..., num_experts)``. The last dimension is
                interpreted as expert logits; all leading dimensions are reduced
                when collecting utilization statistics.
            mask: Optional bool tensor broadcastable to ``logits.shape[:-1]``.
                Masked-out positions do not update running statistics.
            collect_stats: Overrides the default stats collection mode. When
                ``None``, stats are collected only while the module is in
                training mode.

        Returns:
            Tensor with the same shape and dtype as ``logits``. Values are
            probabilities over the expert dimension and sum to 1 along the last
            dimension.
        """
        if collect_stats is not None:
            _require_bool("collect_stats", collect_stats)
        self._validate_logits(logits)

        should_update = self.training if collect_stats is None else collect_stats
        in_warmup = self._in_warmup()
        if should_update:
            self.update_stats(logits, mask=mask)

        temperature = self._temperature_for_forward(force_unit=in_warmup)
        temperature = temperature.view(*([1] * (logits.ndim - 1)), -1)
        if self.training and self.config.use_gumbel_softmax:
            weights = tensor_temperature_gumbel_softmax(logits, temperature=temperature, dim=-1)
        else:
            weights = (logits / temperature).softmax(dim=-1)
        return weights.to(dtype=logits.dtype)

    def _initial_expert_means(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return torch.ones(self.config.num_experts, device=device, dtype=dtype) / self.config.num_experts

    def _initial_priors(self) -> tuple[Tensor, Tensor, Tensor]:
        prior_mean = self._initial_expert_means()
        prior_variance = (self.config.num_experts - 1) / (
            (1 + self.config.num_experts) * self.config.num_experts**2
        )
        prior_var = torch.full((self.config.num_experts,), prior_variance)

        concentration = torch.ones(self.config.num_experts)
        concentration_sum = concentration.sum()
        prior_log = torch.digamma(concentration) - torch.digamma(concentration_sum)
        return prior_mean, prior_var, prior_log

    def _with_prior(self, n_eff: float, stats: Tensor, prior: Tensor) -> Tensor:
        if self.config.prior_strength is None:
            return stats
        numerator = stats * n_eff + self.config.prior_strength * prior
        denominator = n_eff + self.config.prior_strength
        return numerator / denominator

    def _n_eff(self, index: int) -> float:
        decay = self.config.stat_ema_decays[index]
        return 2 / (1 - decay) - 1

    def _raw_temperature(self) -> Tensor:
        if self._disabled:
            return torch.ones_like(self._temperature_ema)

        with torch.no_grad():
            grouped_alpha = self.alpha.view(self.config.temperature_groups, -1)
            dispersion = self._strategy_dispersion(grouped_alpha)
            temperature = self._transform_temperature(dispersion)
            experts_per_group = self.config.num_experts // self.config.temperature_groups
            temperature = temperature.repeat_interleave(experts_per_group)
            return self._clamp_temperature(temperature)

    def _temperature_for_forward(self, *, force_unit: bool) -> Tensor:
        if force_unit or self._disabled:
            return torch.ones_like(self._temperature_ema)

        temperature = self._raw_temperature()
        if self.config.temperature_smoothing_decay is None:
            return temperature

        if self.training:
            self._temperature_ema.copy_(
                ema_update(
                    self._temperature_ema,
                    temperature,
                    decay=self.config.temperature_smoothing_decay,
                )
            )
        return self._temperature_ema

    def _strategy_dispersion(self, alpha: Tensor) -> Tensor:
        if self.config.dispersion_strategy == "var":
            return alpha.var(dim=-1, unbiased=False)
        if self.config.dispersion_strategy == "std":
            return alpha.std(dim=-1, unbiased=False)
        if self.config.dispersion_strategy == "range":
            return alpha.max(dim=-1).values - alpha.min(dim=-1).values
        raise ValueError(
            f"Unsupported ADT dispersion strategy {self.config.dispersion_strategy!r}."
        )

    def _transform_temperature(self, dispersion: Tensor) -> Tensor:
        return 1 + torch.log1p(dispersion)

    def _clamp_temperature(self, temperature: Tensor) -> Tensor:
        if self.config.min_temperature is not None:
            temperature = temperature.clamp_min(self.config.min_temperature)
        if self.config.max_temperature is not None:
            temperature = temperature.clamp_max(self.config.max_temperature)
        return temperature

    def _in_warmup(self) -> bool:
        return self._num_updates_value < self.config.warmup_steps

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._num_updates_value = int(self._num_updates.item())

    def _router_stat_sums(
        self,
        logits: Tensor,
        *,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        flat_logits = logits.reshape(-1, self.config.num_experts)
        if mask is not None:
            flat_mask = self._flatten_mask(mask, logits=logits)
            flat_logits = flat_logits[flat_mask]

        accumulation_dtype = (
            torch.float32
            if flat_logits.dtype in {torch.float16, torch.bfloat16}
            else flat_logits.dtype
        )
        flat_logits = flat_logits.to(dtype=accumulation_dtype)
        count_value = flat_logits.size(0)
        count = torch.tensor(
            count_value,
            device=logits.device,
            dtype=accumulation_dtype,
        )
        zeros = torch.zeros(
            self.config.num_experts,
            device=logits.device,
            dtype=accumulation_dtype,
        )
        if count_value == 0:
            return zeros, zeros.clone(), zeros.clone(), count, count_value

        expert_weights = flat_logits.softmax(dim=-1)
        sums = expert_weights.sum(dim=0)
        square_sums = expert_weights.pow(2).sum(dim=0)
        if self.config.minka_refinement_iters is None:
            log_sums = zeros
        else:
            log_sums = expert_weights.clamp_min(self._eps).log().sum(dim=0)
        return sums, square_sums, log_sums, count, count_value

    def _flatten_mask(self, mask: Tensor, *, logits: Tensor) -> Tensor:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be a bool tensor, got dtype={mask.dtype}.")
        if mask.device != logits.device:
            raise ValueError(
                f"mask must be on the same device as logits: got mask={mask.device}, "
                f"logits={logits.device}."
            )

        try:
            broadcast_mask = torch.broadcast_to(mask, logits.shape[:-1])
        except RuntimeError as exc:
            raise ValueError(
                "mask must be broadcastable to logits.shape[:-1]: "
                f"got mask shape {tuple(mask.shape)}, logits shape {tuple(logits.shape)}."
            ) from exc
        return broadcast_mask.reshape(-1)

    def _sync_stat_sums(
        self,
        sums: Tensor,
        square_sums: Tensor,
        log_sums: Tensor,
        count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return sums, square_sums, log_sums, count

        synced = torch.cat([sums, square_sums, log_sums, count.reshape(1)])
        torch.distributed.all_reduce(synced, op=torch.distributed.ReduceOp.SUM)
        num_experts = self.config.num_experts
        sums = synced[:num_experts]
        square_sums = synced[num_experts : 2 * num_experts]
        log_sums = synced[2 * num_experts : 3 * num_experts]
        count = synced[-1]
        return sums, square_sums, log_sums, count

    def _validate_logits(self, logits: Tensor) -> None:
        if logits.ndim < 2:
            raise ValueError(f"logits must have at least 2 dimensions, got shape {logits.shape}.")
        if logits.size(-1) != self.config.num_experts:
            raise ValueError(
                "last logits dimension must match num_experts: "
                f"got {logits.size(-1)}, expected {self.config.num_experts}."
            )

ADT = AdaptiveDirichletTempering


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def refine_alpha_minka(
    alpha: Tensor,
    log_mean: Tensor,
    num_iters: int = 2,
    eps: float = 1e-3,
) -> Tensor:
    alpha = alpha.clamp_min(eps)

    for _ in range(num_iters):
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        expected_log_theta = torch.digamma(alpha) - torch.digamma(alpha_0)

        trigamma_alpha = torch.polygamma(1, alpha)
        trigamma_alpha_0 = torch.polygamma(1, alpha_0)

        numerator = log_mean - expected_log_theta
        denominator = (trigamma_alpha - trigamma_alpha_0).clamp_min(eps)

        alpha = (alpha + numerator / denominator).clamp_min(eps)

    return alpha


def tensor_temperature_gumbel_softmax(
    logits: Tensor,
    *,
    temperature: Tensor,
    dim: int,
    eps: float = 1e-10,
) -> Tensor:
    uniform = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(uniform.clamp(min=eps, max=1 - eps)))
    return ((logits + gumbel) / temperature).softmax(dim=dim)


def ema_update(
    x: Tensor,
    y: Tensor,
    *,
    momentum: float | None = None,
    decay: float | None = None,
) -> Tensor:
    if (momentum is None) == (decay is None):
        raise ValueError("Exactly one of momentum or decay must be specified.")
    if momentum is not None:
        return x * (1 - momentum) + y * momentum
    if decay is None:
        raise ValueError("decay must be specified.")
    return x * decay + y * (1 - decay)


def _adt_metric_groups(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values}
    allowed = {"variance", "alpha", "temperature"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown ADT-only metrics: {sorted(unknown)!r}.")
    return result


def _summary_metrics(values: Tensor, name: str) -> dict[str, Tensor]:
    if values.ndim != 1:
        raise ValueError(f"ADT stat {name} must be a 1D tensor, got shape {tuple(values.shape)}.")
    return {
        f"{name}_mean": values.mean(),
        f"{name}_std": values.std(unbiased=False),
        f"{name}_min": values.min(),
        f"{name}_max": values.max(),
    }


def _collect_observation(module: nn.Module) -> dict[str, observation.Observation]:
    if not isinstance(module, AdaptiveDirichletTempering):
        raise TypeError("ADT observation requires AdaptiveDirichletTempering.")
    return {
        name: observation.Curve(value)
        for name, value in adt_stats_metrics(module.stats()).items()
    }


observation.registry.register(
    AdaptiveDirichletTempering,
    (
        observation.StateObservation(
            name="adt",
            collect=_collect_observation,
            recommended=True,
        ),
    ),
)
