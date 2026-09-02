from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anytrain._buffer import register_buffer

from . import _checks
from ._observation import recommend as recommend_observation
from ._observation import register as register_observation
from .lookup import nearest_codebook_indices
from .output import QuantizationLoss, QuantizeOutput
from .projection import make_projection


@dataclass(frozen=True)
class VQConfig:
    input_dim: int
    codebook_size: int
    codebook_dim: int | None = None
    normalize_lookup: bool = True
    lookup_in_float32: bool = True
    weight_norm: bool = False
    scale_grad_by_freq: bool = False
    use_ema: bool = False
    decay: float = 0.99
    eps: float = 1e-5
    projection_bias: bool = True
    kmeans_init: bool = False
    kmeans_iters: int = 10
    dead_code_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}.")
        codebook_dim = self.resolved_codebook_dim
        if codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {codebook_dim}.")
        if not 0 <= self.decay < 1:
            raise ValueError(f"decay must be in [0, 1), got {self.decay}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        if self.kmeans_iters <= 0:
            raise ValueError(f"kmeans_iters must be positive, got {self.kmeans_iters}.")
        if self.dead_code_threshold < 0:
            raise ValueError(
                "dead_code_threshold must be non-negative, "
                f"got {self.dead_code_threshold}."
            )
        if not self.use_ema and (self.kmeans_init or self.dead_code_threshold > 0):
            raise ValueError(
                "kmeans_init and dead-code recycling require use_ema=True."
            )

    @property
    def resolved_codebook_dim(self) -> int:
        return self.input_dim if self.codebook_dim is None else self.codebook_dim


@register_observation
class EmbeddingVectorQuantizer(nn.Module):
    config: VQConfig
    project_in: nn.Module
    project_out: nn.Module
    codebook: nn.Embedding
    _ema_counts: Tensor
    _ema_sums: Tensor

    def __init__(self, config: VQConfig) -> None:
        super().__init__()
        codebook_dim = config.resolved_codebook_dim
        self.config = config
        self.input_dim = config.input_dim
        self.codebook_size = config.codebook_size
        self.codebook_dim = codebook_dim
        self.num_codebooks = 1

        self.project_in = make_projection(
            config.input_dim,
            codebook_dim,
            bias=config.projection_bias,
            weight_norm=config.weight_norm,
        )
        self.project_out = make_projection(
            codebook_dim,
            config.input_dim,
            bias=config.projection_bias,
            weight_norm=config.weight_norm,
        )
        self.codebook = nn.Embedding(
            config.codebook_size,
            codebook_dim,
            scale_grad_by_freq=config.scale_grad_by_freq,
        )
        if config.use_ema:
            register_buffer(self, "_ema_counts", torch.ones(config.codebook_size))
            register_buffer(self, "_ema_sums", torch.empty(config.codebook_size, codebook_dim))
            self.codebook.weight.requires_grad_(False)
        # Zero EMA counts are the checkpoint-compatible pending k-means sentinel.
        self._ema_initialized = not config.kmeans_init
        self.reset_parameters()
        recommend_observation(self)

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse)
        if self.use_ema:
            self._ema_counts.data = self._ema_counts.data.float()
            self._ema_sums.data = self._ema_sums.data.float()
        return module

    @property
    def use_ema(self) -> bool:
        return self.config.use_ema

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,)

    @property
    def ema_counts(self) -> Tensor:
        if not self.use_ema:
            raise AttributeError("ema_counts is only available when use_ema=True.")
        return self._ema_counts

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
            if self.use_ema:
                if self.config.kmeans_init:
                    self._ema_sums.zero_()
                    self._ema_counts.zero_()
                    self._ema_initialized = False
                else:
                    self._ema_sums.copy_(self.codebook.weight)
                    self._ema_counts.fill_(1)
                    self._ema_initialized = True

    def forward(
        self,
        latents: Tensor,
        *,
        _allow_empty: bool = False,
        _on_inactive: Callable[[nn.Module], None] | None = None,
    ) -> QuantizeOutput:
        # Keep the long-standing ``forward -> quantize(latents)`` delegation
        # for subclasses that override the public one-argument method.  The
        # private empty-stage knobs are only used by callers that explicitly
        # opt into that extension; passing them on the ordinary path would
        # turn a backwards-compatible override into a TypeError.
        if not _allow_empty and _on_inactive is None:
            return self.quantize(latents)
        return self.quantize(
            latents,
            _allow_empty=_allow_empty,
            _on_inactive=_on_inactive,
        )

    def quantize(
        self,
        latents: Tensor,
        *,
        _allow_empty: bool = False,
        _on_inactive: Callable[[nn.Module], None] | None = None,
    ) -> QuantizeOutput:
        _checks.input_latents(
            latents,
            self.input_dim,
            require_non_empty=not _allow_empty,
        )
        if _allow_empty and latents.numel() == 0:
            # Residual dropout uses empty batches for stages that have no local
            # assignments.  Keep the public (non-empty) input contract while
            # still allowing those stages to participate in distributed EMA
            # collectives without constructing NaN losses from empty tensors.
            if _on_inactive is not None:
                _on_inactive(self)
            elif self.training and self.use_ema:
                self._update_ema_without_assignments()
            leading_shape = latents.shape[:-1]
            return QuantizeOutput(
                quantized_latents=latents.new_empty(*leading_shape, self.input_dim),
                indices=torch.empty(
                    leading_shape,
                    dtype=torch.long,
                    device=latents.device,
                ),
                codebook_vectors=latents.new_empty(*leading_shape, self.codebook_dim),
                latents=latents.new_empty(*leading_shape, self.codebook_dim),
                loss=None,
            )
        projected_latents = self.project_in(latents)
        if self.training and self.use_ema:
            self._prepare_ema(projected_latents)
        codebook_vectors, indices = self._nearest_codebook_vectors(projected_latents)

        loss = None
        if self.training:
            if self.use_ema:
                self._update_ema(projected_latents, indices)
                commitment = F.mse_loss(projected_latents, codebook_vectors.detach())
                loss = QuantizationLoss(
                    commitment=commitment,
                    codebook=torch.zeros_like(commitment),
                )
            else:
                loss = QuantizationLoss(
                    commitment=F.mse_loss(projected_latents, codebook_vectors.detach()),
                    codebook=F.mse_loss(codebook_vectors, projected_latents.detach()),
                )

        straight_through = projected_latents + (codebook_vectors - projected_latents).detach()
        quantized_latents = self.project_out(straight_through)
        return QuantizeOutput(
            quantized_latents=quantized_latents,
            indices=indices,
            codebook_vectors=codebook_vectors,
            latents=projected_latents,
            loss=loss,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        _checks.input_latents(latents, self.input_dim)
        codebook_vectors, _ = self._nearest_codebook_vectors(self.project_in(latents))
        return codebook_vectors

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(codebook_vectors, self.codebook_dim)
        _, indices = self._nearest_codebook_vectors(codebook_vectors)
        return indices

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        _checks.indices(indices, self.codebook_size)
        return self._indices_to_codebook_vectors_unchecked(indices)

    def _indices_to_codebook_vectors_unchecked(self, indices: Tensor) -> Tensor:
        """Lookup already-validated indices without repeating range checks."""

        return self.codebook(indices)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.codebook_vectors(codebook_vectors, self.codebook_dim)
        return self.project_out(codebook_vectors)

    def _nearest_codebook_vectors(self, projected_latents: Tensor) -> tuple[Tensor, Tensor]:
        _checks.codebook_vectors(
            projected_latents,
            self.codebook_dim,
            name="projected_latents",
        )
        leading_shape = projected_latents.shape[:-1]
        indices = nearest_codebook_indices(
            projected_latents,
            self.codebook.weight,
            normalize=self.config.normalize_lookup,
            lookup_in_float32=self.config.lookup_in_float32,
        )
        codebook_vectors = self.codebook(indices).reshape(*leading_shape, self.codebook_dim)
        return codebook_vectors, indices

    @torch.no_grad()
    def _update_ema(self, projected_latents: Tensor, indices: Tensor) -> None:
        flat_latents = projected_latents.detach().reshape(-1, self.codebook_dim)
        flat_latents = flat_latents.float()
        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.codebook_size).float()
        sums = torch.zeros(
            self.codebook_size,
            self.codebook_dim,
            device=flat_latents.device,
            dtype=torch.float32,
        )
        sums.index_add_(0, flat_indices, flat_latents)
        self._sync_ema_stats(counts, sums)
        # A public quantize call always supplies at least one latent, so this
        # path is guaranteed to have a non-zero global count after the
        # collective.  Keep the normal EMA update on the in-place hot path;
        # only the explicitly empty-stage helper needs the predicate-based
        # no-op handling below.
        self._apply_ema_stats(counts, sums, assume_non_empty=True)

    @torch.no_grad()
    def _update_ema_without_assignments(self) -> None:
        if not self.use_ema:
            raise RuntimeError("EMA participation requires use_ema=True.")
        if not self._distributed_ready():
            return

        empty_latents = self._ema_sums.new_empty(0, self.codebook_dim)
        self._prepare_ema(empty_latents)
        counts = torch.zeros_like(self._ema_counts)
        sums = torch.zeros_like(self._ema_sums)
        self._sync_ema_stats(counts, sums)
        self._apply_ema_stats(counts, sums)

    @torch.no_grad()
    def _prepare_ema(self, projected_latents: Tensor) -> None:
        flat_latents = projected_latents.detach().reshape(-1, self.codebook_dim).float()
        if not self._maybe_initialize_codebook(flat_latents):
            return
        self._recycle_dead_codes(flat_latents)

    def _maybe_initialize_codebook(self, flat_latents: Tensor) -> bool:
        if self._ema_initialized:
            return True
        if bool(self._ema_counts.sum().item() > 0):
            self._ema_initialized = True
            return True

        centers = self._distributed_sample_vectors(flat_latents, self.codebook_size)
        if centers is None:
            return False

        for _ in range(self.config.kmeans_iters):
            counts, sums = self._codebook_stats(flat_latents, centers)
            self._sync_ema_stats(counts, sums)
            active = counts > 0
            centers[active] = sums[active] / counts[active, None]

        counts, sums = self._codebook_stats(flat_latents, centers)
        self._sync_ema_stats(counts, sums)
        initial_count = max(1.0, self.config.dead_code_threshold)
        sparse = counts < initial_count
        counts[sparse] = initial_count
        sums[sparse] = centers[sparse] * initial_count

        self._ema_counts.copy_(counts)
        self._ema_sums.copy_(sums)
        self.codebook.weight.copy_(centers)
        self._ema_initialized = True
        return True

    def _recycle_dead_codes(self, flat_latents: Tensor) -> None:
        threshold = self.config.dead_code_threshold
        if threshold <= 0:
            return
        dead = self._ema_counts < threshold
        dead_count = int(dead.sum().item())
        if dead_count == 0:
            return

        replacements = self._distributed_sample_vectors(flat_latents, dead_count)
        if replacements is None:
            return
        self.codebook.weight[dead] = replacements.to(dtype=self.codebook.weight.dtype)
        self._ema_counts[dead] = threshold
        self._ema_sums[dead] = replacements * threshold

    def _distributed_sample_vectors(
        self,
        flat_latents: Tensor,
        count: int,
    ) -> Tensor | None:
        if count <= 0:
            return flat_latents.new_empty(0, self.codebook_dim)
        if not self._distributed_ready():
            if flat_latents.shape[0] == 0:
                return None
            return _sample_vectors(flat_latents, count)

        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            if flat_latents.shape[0] == 0:
                return None
            return _sample_vectors(flat_latents, count)

        rank = torch.distributed.get_rank()
        available = torch.zeros(world_size, dtype=torch.long, device=flat_latents.device)
        available[rank] = int(flat_latents.shape[0] > 0)
        from anytrain.lightning.audit import audit_collective

        audit_collective(
            "torch.distributed.all_reduce",
            lambda: torch.distributed.all_reduce(
                available, op=torch.distributed.ReduceOp.SUM
            ),
            reduce_op="sum",
            source="EmbeddingVectorQuantizer._distributed_sample_vectors.available",
            tensor_shape=tuple(available.shape),
            tensor_dtype=str(available.dtype),
            tensor_device=str(available.device),
        )
        active_ranks = torch.where(available > 0)[0]
        if active_ranks.numel() == 0:
            return None

        positions = torch.arange(count, device=flat_latents.device)
        # Give every replacement a single owner, then SUM to share the sampled pool.
        owners = active_ranks[positions.remainder(active_ranks.numel())]
        local_positions = positions[owners == rank]
        sampled = flat_latents.new_zeros(count, self.codebook_dim)
        if local_positions.numel() > 0:
            sampled[local_positions] = _sample_vectors(
                flat_latents,
                int(local_positions.numel()),
            )
        audit_collective(
            "torch.distributed.all_reduce",
            lambda: torch.distributed.all_reduce(
                sampled, op=torch.distributed.ReduceOp.SUM
            ),
            reduce_op="sum",
            source="EmbeddingVectorQuantizer._distributed_sample_vectors.sampled",
            tensor_shape=tuple(sampled.shape),
            tensor_dtype=str(sampled.dtype),
            tensor_device=str(sampled.device),
        )
        return sampled

    def _codebook_stats(self, flat_latents: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
        indices = nearest_codebook_indices(
            flat_latents,
            codebook,
            normalize=self.config.normalize_lookup,
            lookup_in_float32=self.config.lookup_in_float32,
        ).reshape(-1)
        counts = torch.bincount(indices, minlength=self.codebook_size).float()
        sums = torch.zeros(
            self.codebook_size,
            self.codebook_dim,
            device=flat_latents.device,
            dtype=torch.float32,
        )
        sums.index_add_(0, indices, flat_latents)
        return counts, sums

    def _apply_ema_stats(
        self,
        counts: Tensor,
        sums: Tensor,
        *,
        assume_non_empty: bool = False,
    ) -> None:
        if assume_non_empty:
            # This is the regular quantization path.  Preserve the original
            # in-place operation order both for throughput and bitwise EMA
            # compatibility; ``counts`` is guaranteed non-zero there.
            self._ema_counts.mul_(self.config.decay).add_(
                counts,
                alpha=1 - self.config.decay,
            )
            self._ema_sums.mul_(self.config.decay).add_(
                sums,
                alpha=1 - self.config.decay,
            )
            total_count = self._ema_counts.sum()
            smoothed_counts = (
                (self._ema_counts + self.config.eps)
                / (total_count + self.codebook_size * self.config.eps)
                * total_count
            )
            self.codebook.weight.copy_(
                self._ema_sums / smoothed_counts.clamp_min(self.config.eps)[:, None]
            )
            return

        # Empty residual stages must still participate in DDP collectives.
        # Keep the all-zero (globally inactive) path entirely on device.  A
        # Python scalar check here would synchronize every inactive stage;
        # selecting the old buffers with a tensor predicate preserves exact
        # no-op EMA semantics when every rank is empty.
        # Counts are normally non-negative bincount results.  ``ne(0)`` is
        # deliberately used instead of ``gt(0)`` so malformed NaN/negative
        # stats retain the legacy propagation behavior; only an exact global
        # zero is treated as the inactive no-op case.
        has_assignments = counts.sum().ne(0)
        # Keep the legacy in-place update order (and its rounding behavior)
        # while computing into temporaries so the no-op branch can select the
        # original buffers below.  This also avoids materializing separate
        # multiply results for both operands.
        updated_counts = self._ema_counts.clone()
        updated_counts.mul_(self.config.decay).add_(
            counts,
            alpha=1 - self.config.decay,
        )
        updated_sums = self._ema_sums.clone()
        updated_sums.mul_(self.config.decay).add_(
            sums,
            alpha=1 - self.config.decay,
        )

        total_count = updated_counts.sum()
        smoothed_counts = (
            (updated_counts + self.config.eps)
            / (total_count + self.codebook_size * self.config.eps)
            * total_count
        )
        updated_weight = updated_sums / smoothed_counts.clamp_min(self.config.eps)[:, None]

        self._ema_counts.copy_(
            torch.where(has_assignments, updated_counts, self._ema_counts)
        )
        self._ema_sums.copy_(torch.where(has_assignments, updated_sums, self._ema_sums))
        self.codebook.weight.copy_(
            torch.where(
                has_assignments,
                updated_weight.to(dtype=self.codebook.weight.dtype),
                self.codebook.weight,
            )
        )

    def _sync_ema_stats(self, counts: Tensor, sums: Tensor) -> None:
        if not self._distributed_ready():
            return
        from anytrain.lightning.audit import audit_collective

        for tensor, source in (
            (counts, "EmbeddingVectorQuantizer._sync_ema_stats.counts"),
            (sums, "EmbeddingVectorQuantizer._sync_ema_stats.sums"),
        ):
            audit_collective(
                "torch.distributed.all_reduce",
                lambda tensor=tensor: torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.SUM
                ),
                reduce_op="sum",
                source=source,
                tensor_shape=tuple(tensor.shape),
                tensor_dtype=str(tensor.dtype),
                tensor_device=str(tensor.device),
            )

    def _distributed_ready(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()


def _sample_vectors(samples: Tensor, count: int) -> Tensor:
    if samples.shape[0] >= count:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:count]
    else:
        indices = torch.randint(samples.shape[0], (count,), device=samples.device)
    return samples.index_select(0, indices)
