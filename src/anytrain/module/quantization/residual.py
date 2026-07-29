from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import _checks
from .embedding import EmbeddingVectorQuantizer, VQConfig
from .output import QuantizationLoss, QuantizeOutput


@dataclass
class RVQConfig:
    vq_configs: list[VQConfig]
    dropout: float | None = None

    def __post_init__(self) -> None:
        if not self.vq_configs:
            raise ValueError("vq_configs must contain at least one VQConfig.")
        if self.dropout is not None and not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}.")

        first = self.vq_configs[0]
        for config in self.vq_configs:
            if config.input_dim != first.input_dim:
                raise ValueError("RVQ first version requires uniform input_dim.")
            if config.codebook_dim != first.codebook_dim:
                raise ValueError("RVQ first version requires uniform codebook_dim.")
            if config.codebook_size != first.codebook_size:
                raise ValueError("RVQ first version requires uniform codebook_size.")

    @property
    def num_codebooks(self) -> int:
        return len(self.vq_configs)

    @classmethod
    def from_kwargs(
        cls,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        normalize_latents: bool = True,
        weight_norm: bool = False,
        scale_grad_by_freq: bool = False,
        use_ema: bool = False,
        decay: float = 0.99,
        eps: float = 1e-5,
        projection_bias: bool = True,
        dropout: float | None = None,
    ) -> RVQConfig:
        if num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {num_codebooks}.")
        return cls(
            vq_configs=[
                VQConfig(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    normalize_latents=normalize_latents,
                    weight_norm=weight_norm,
                    scale_grad_by_freq=scale_grad_by_freq,
                    use_ema=use_ema,
                    decay=decay,
                    eps=eps,
                    projection_bias=projection_bias,
                )
                for _ in range(num_codebooks)
            ],
            dropout=dropout,
        )


class ResidualVectorQuantizer(nn.Module):
    config: RVQConfig
    quantizers: nn.ModuleList

    def __init__(self, config: RVQConfig) -> None:
        super().__init__()
        codebook_dim = config.vq_configs[0].codebook_dim
        if codebook_dim is None:
            raise RuntimeError("VQConfig.codebook_dim should be resolved in __post_init__.")

        self.config = config
        self.num_codebooks = config.num_codebooks
        self.input_dim = config.vq_configs[0].input_dim
        self.codebook_dim = codebook_dim
        self.codebook_size = config.vq_configs[0].codebook_size
        self.quantizers = nn.ModuleList(
            [EmbeddingVectorQuantizer(vq_config) for vq_config in config.vq_configs]
        )

    def forward(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> QuantizeOutput:
        return self.quantize(latents, num_active_codebooks=num_active_codebooks)

    def quantize(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> QuantizeOutput:
        _checks.input_latents(latents, self.input_dim)
        active_count = self._active_codebooks(num_active_codebooks)
        leading_shape = latents.shape[:-1]
        flat_latents = latents.reshape(-1, self.input_dim)
        dropout_enabled = (
            self.training
            and self.config.dropout is not None
            and self.config.dropout > 0
            and active_count > 1
        )
        if not dropout_enabled:
            return self._quantize_all_active(flat_latents, leading_shape, active_count)

        flat_size = flat_latents.size(0)

        active_mask = self._sample_active_mask(flat_size, active_count, flat_latents.device)
        residual = flat_latents.clone()
        quantized_sum = torch.zeros_like(flat_latents)
        indices_list: list[Tensor] = []
        vector_list: list[Tensor] = []
        latent_list: list[Tensor] = []
        commitment_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
        codebook_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
        loss_weight_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
        has_loss = False
        # `_sample_active_mask` always returns a prefix mask: each vector keeps
        # the first N codebooks.  Reduce that prefix depth once so the loop
        # does not synchronize the device once per codebook.
        active_depth = (
            0 if active_mask.numel() == 0 else int(active_mask.sum(dim=1).max().item())
        )

        for index, quantizer in enumerate(self.quantizers[:active_count]):
            mask = active_mask[:, index]
            if index < active_depth:
                output = quantizer(residual[mask])
                selected_quantized = output.quantized_latents
                residual[mask] = residual[mask] - selected_quantized
                quantized_sum[mask] = quantized_sum[mask] + selected_quantized

                indices_i = torch.full(
                    (flat_size,),
                    -1,
                    dtype=output.indices.dtype,
                    device=flat_latents.device,
                )
                indices_i[mask] = output.indices

                vectors_i = torch.zeros(
                    flat_size,
                    self.codebook_dim,
                    dtype=flat_latents.dtype,
                    device=flat_latents.device,
                )
                latents_i = torch.zeros_like(vectors_i)
                if output.codebook_vectors is not None:
                    vectors_i[mask] = output.codebook_vectors
                if output.latents is not None:
                    latents_i[mask] = output.latents

                if output.loss is not None:
                    loss_weight = mask.sum().to(dtype=flat_latents.dtype)
                    commitment_loss_sum = commitment_loss_sum + output.loss.commitment * loss_weight
                    codebook_loss_sum = codebook_loss_sum + output.loss.codebook * loss_weight
                    loss_weight_sum = loss_weight_sum + loss_weight
                    has_loss = True
            else:
                if quantizer.training and quantizer.use_ema:
                    # Match peer EMA collectives when this rank has no assignments.
                    quantizer._update_ema_without_assignments()
                indices_i = torch.full(
                    (flat_size,),
                    -1,
                    dtype=torch.long,
                    device=flat_latents.device,
                )
                vectors_i = torch.zeros(
                    flat_size,
                    self.codebook_dim,
                    dtype=flat_latents.dtype,
                    device=flat_latents.device,
                )
                latents_i = torch.zeros_like(vectors_i)

            indices_list.append(indices_i)
            vector_list.append(vectors_i)
            latent_list.append(latents_i)

        loss = None
        if has_loss:
            loss = QuantizationLoss(
                commitment=commitment_loss_sum / loss_weight_sum,
                codebook=codebook_loss_sum / loss_weight_sum,
            )

        return QuantizeOutput(
            quantized_latents=quantized_sum.reshape(*leading_shape, self.input_dim),
            indices=torch.stack(indices_list, dim=-1).reshape(*leading_shape, active_count),
            codebook_vectors=torch.stack(vector_list, dim=1).reshape(
                *leading_shape,
                active_count,
                self.codebook_dim,
            ),
            latents=torch.stack(latent_list, dim=1).reshape(
                *leading_shape,
                active_count,
                self.codebook_dim,
            ),
            loss=loss,
            active_codebook_mask=active_mask.reshape(*leading_shape, active_count),
        )

    def _quantize_all_active(
        self,
        flat_latents: Tensor,
        leading_shape: torch.Size,
        active_count: int,
    ) -> QuantizeOutput:
        residual = flat_latents
        quantized_sum = torch.zeros_like(flat_latents)
        indices_list: list[Tensor] = []
        vector_list: list[Tensor] = []
        latent_list: list[Tensor] = []
        commitment_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
        codebook_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
        loss_count = 0

        for quantizer in self.quantizers[:active_count]:
            output = quantizer(residual)
            quantized_latents = output.quantized_latents
            residual = residual - quantized_latents
            quantized_sum = quantized_sum + quantized_latents

            indices_list.append(output.indices.reshape(-1))
            vector_list.append(
                self._require_output_tensor(output.codebook_vectors, "codebook_vectors").reshape(
                    -1,
                    self.codebook_dim,
                )
            )
            latent_list.append(
                self._require_output_tensor(output.latents, "latents").reshape(
                    -1,
                    self.codebook_dim,
                )
            )
            if output.loss is not None:
                commitment_loss_sum = commitment_loss_sum + output.loss.commitment
                codebook_loss_sum = codebook_loss_sum + output.loss.codebook
                loss_count += 1

        loss = None
        if loss_count > 0:
            loss = QuantizationLoss(
                commitment=commitment_loss_sum / loss_count,
                codebook=codebook_loss_sum / loss_count,
            )

        flat_size = flat_latents.size(0)
        return QuantizeOutput(
            quantized_latents=quantized_sum.reshape(*leading_shape, self.input_dim),
            indices=torch.stack(indices_list, dim=-1).reshape(*leading_shape, active_count),
            codebook_vectors=torch.stack(vector_list, dim=1).reshape(
                *leading_shape,
                active_count,
                self.codebook_dim,
            ),
            latents=torch.stack(latent_list, dim=1).reshape(
                *leading_shape,
                active_count,
                self.codebook_dim,
            ),
            loss=loss,
            active_codebook_mask=torch.ones(
                flat_size,
                active_count,
                dtype=torch.bool,
                device=flat_latents.device,
            ).reshape(*leading_shape, active_count),
        )

    def latents_to_codebook_vectors(
        self,
        latents: Tensor,
        *,
        num_active_codebooks: int | None = None,
    ) -> Tensor:
        _checks.input_latents(latents, self.input_dim)
        active_count = self._active_codebooks(num_active_codebooks)
        leading_shape = latents.shape[:-1]
        residual = latents.reshape(-1, self.input_dim)

        vector_list = []
        for quantizer in self.quantizers[:active_count]:
            codebook_vectors = quantizer.latents_to_codebook_vectors(residual)
            vector_list.append(codebook_vectors)
            residual = residual - quantizer.project_codebook_vectors(codebook_vectors)

        return torch.stack(vector_list, dim=1).reshape(
            *leading_shape,
            active_count,
            self.codebook_dim,
        )

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        _checks.active_codebook_vectors(
            codebook_vectors,
            codebook_dim=self.codebook_dim,
            num_codebooks=self.num_codebooks,
        )
        indices = []
        for index, quantizer in enumerate(self.quantizers[: codebook_vectors.shape[-2]]):
            indices.append(quantizer.codebook_vectors_to_indices(codebook_vectors[..., index, :]))
        return torch.stack(indices, dim=-1)

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        _checks.active_indices(
            indices,
            codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            allow_inactive=True,
        )
        vectors = []
        for index, quantizer in enumerate(self.quantizers[: indices.shape[-1]]):
            indices_i = indices[..., index]
            mask = indices_i >= 0
            vectors_i = torch.zeros(
                *indices_i.shape,
                self.codebook_dim,
                dtype=quantizer.codebook.weight.dtype,
                device=indices.device,
            )
            if bool(mask.any().item()):
                vectors_i[mask] = quantizer.indices_to_codebook_vectors(indices_i[mask])
            vectors.append(vectors_i)
        return torch.stack(vectors, dim=-2)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        _checks.active_codebook_vectors(
            codebook_vectors,
            codebook_dim=self.codebook_dim,
            num_codebooks=self.num_codebooks,
        )
        projected = []
        for index, quantizer in enumerate(self.quantizers[: codebook_vectors.shape[-2]]):
            projected.append(quantizer.project_codebook_vectors(codebook_vectors[..., index, :]))
        return torch.stack(projected, dim=-2).sum(dim=-2)

    def _sample_active_mask(
        self,
        flat_size: int,
        active_count: int,
        device: torch.device,
    ) -> Tensor:
        if not self.training or self.config.dropout is None or self.config.dropout == 0:
            return torch.ones(flat_size, active_count, dtype=torch.bool, device=device)

        active_per_vector = torch.full((flat_size,), active_count, dtype=torch.long, device=device)
        dropout_mask = torch.rand(flat_size, device=device) < self.config.dropout
        sampled_active = torch.randint(
            1,
            active_count + 1,
            size=(flat_size,),
            device=device,
        )
        active_per_vector = torch.where(dropout_mask, sampled_active, active_per_vector)
        positions = torch.arange(active_count, device=device).unsqueeze(0)
        return positions < active_per_vector.unsqueeze(1)

    @staticmethod
    def _require_output_tensor(value: Tensor | None, name: str) -> Tensor:
        if value is None:
            raise RuntimeError(f"RVQ quantizer output unexpectedly omitted {name}.")
        return value

    def _active_codebooks(self, num_active_codebooks: int | None) -> int:
        if num_active_codebooks is None:
            return self.num_codebooks
        if num_active_codebooks <= 0 or num_active_codebooks > self.num_codebooks:
            raise ValueError(
                "num_active_codebooks must be in [1, num_codebooks]: "
                f"got {num_active_codebooks}, num_codebooks={self.num_codebooks}."
            )
        return num_active_codebooks
