from __future__ import annotations

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

    def forward(self, latents: Tensor) -> QuantizeOutput:
        return self.quantize(latents)

    def quantize(self, latents: Tensor) -> QuantizeOutput:
        _checks.input_latents(latents, self.input_dim)
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
        self._apply_ema_stats(counts, sums)

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
        if not bool(counts.any().item()):
            return
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
        torch.distributed.all_reduce(available, op=torch.distributed.ReduceOp.SUM)
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
        torch.distributed.all_reduce(sampled, op=torch.distributed.ReduceOp.SUM)
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

    def _apply_ema_stats(self, counts: Tensor, sums: Tensor) -> None:
        self._ema_counts.mul_(self.config.decay).add_(counts, alpha=1 - self.config.decay)
        self._ema_sums.mul_(self.config.decay).add_(sums, alpha=1 - self.config.decay)

        total_count = self._ema_counts.sum()
        smoothed_counts = (
            (self._ema_counts + self.config.eps)
            / (total_count + self.codebook_size * self.config.eps)
            * total_count
        )
        self.codebook.weight.copy_(
            self._ema_sums / smoothed_counts.clamp_min(self.config.eps)[:, None]
        )

    def _sync_ema_stats(self, counts: Tensor, sums: Tensor) -> None:
        if not self._distributed_ready():
            return
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)

    def _distributed_ready(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()


def _sample_vectors(samples: Tensor, count: int) -> Tensor:
    if samples.shape[0] >= count:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:count]
    else:
        indices = torch.randint(samples.shape[0], (count,), device=samples.device)
    return samples.index_select(0, indices)
