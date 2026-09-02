from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .output import QuantizationLoss, QuantizeOutput


def sample_active_mask(
    flat_size: int,
    active_count: int,
    device: torch.device,
    *,
    training: bool,
    dropout: float | None,
) -> Tensor:
    if not training or dropout is None or dropout == 0:
        return torch.ones(flat_size, active_count, dtype=torch.bool, device=device)

    active_per_vector = torch.full((flat_size,), active_count, dtype=torch.long, device=device)
    dropout_mask = torch.rand(flat_size, device=device) < dropout
    sampled_active = torch.randint(
        1,
        active_count + 1,
        size=(flat_size,),
        device=device,
    )
    active_per_vector = torch.where(dropout_mask, sampled_active, active_per_vector)
    positions = torch.arange(active_count, device=device).unsqueeze(0)
    return positions < active_per_vector.unsqueeze(1)


def run_residual_quantization(
    *,
    flat_latents: Tensor,
    leading_shape: torch.Size,
    active_count: int,
    quantizers: nn.ModuleList,
    input_dim: int,
    codebook_dim: int,
    training: bool,
    dropout: float | None,
    sample_mask: Callable[[int, int, torch.device], Tensor],
    on_inactive: Callable[[nn.Module], None] | None = None,
) -> QuantizeOutput:
    """Run residual stages while keeping dropout control flow device-local.

    ``on_inactive`` is invoked for a stage with no local assignments.  This
    preserves distributed EMA participation without reducing a CUDA mask to a
    Python integer.
    """
    dropout_enabled = training and dropout is not None and dropout > 0 and active_count > 1
    if not dropout_enabled:
        return _run_all_active(
            flat_latents=flat_latents,
            leading_shape=leading_shape,
            active_count=active_count,
            quantizers=quantizers,
            input_dim=input_dim,
            codebook_dim=codebook_dim,
        )

    flat_size = flat_latents.size(0)
    active_mask = sample_mask(flat_size, active_count, flat_latents.device)
    residual = flat_latents.clone()
    quantized_sum = torch.zeros_like(flat_latents)
    indices_list: list[Tensor] = []
    vector_list: list[Tensor] = []
    latent_list: list[Tensor] = []
    commitment_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
    codebook_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
    loss_weight_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
    has_loss = False

    # CPU scalar conversion is synchronous only with respect to host memory,
    # and retaining the old active-depth shortcut avoids launching/probing an
    # empty quantizer for every trailing stage.  Accelerator paths deliberately
    # leave this ``None``: reducing a CUDA mask to a Python integer would add a
    # host/device synchronization to every residual forward.
    cpu_active_depth: int | None = None
    if flat_latents.device.type == "cpu":
        active_columns = torch.where(active_mask.any(dim=0))[0]
        cpu_active_depth = 0 if active_columns.numel() == 0 else int(active_columns[-1]) + 1

    for index, quantizer in enumerate(quantizers[:active_count]):
        mask = active_mask[:, index]
        stage_is_empty = cpu_active_depth is not None and index >= cpu_active_depth
        if stage_is_empty:
            # No local position reaches this trailing stage.  Invoke the
            # callback directly for DDP EMA participation, then synthesize the
            # known empty shape; this also keeps custom stage modules from
            # needing private kwargs.
            selected_residual = residual[:0]
            if on_inactive is not None:
                on_inactive(quantizer)
            output = _empty_stage_output(
                selected_residual,
                input_dim=input_dim,
                codebook_dim=codebook_dim,
            )
        else:
            selected_residual = residual[mask]
            if selected_residual.numel() == 0:
                # Non-prefix masks (or accelerator masks whose stage happens
                # to be empty) use the same callback/synthetic path without a
                # scalar ``any().item()`` check.
                if on_inactive is not None:
                    on_inactive(quantizer)
                output = _empty_stage_output(
                    selected_residual,
                    input_dim=input_dim,
                    codebook_dim=codebook_dim,
                )
            else:
                # Keep the ordinary public call for active batches.  Besides
                # avoiding needless private-kwarg plumbing, this preserves the
                # residual runner's compatibility with custom quantizer stages.
                output = quantizer(selected_residual)
        selected_quantized = output.quantized_latents
        residual[mask] = selected_residual - selected_quantized
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
            codebook_dim,
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
        quantized_latents=quantized_sum.reshape(*leading_shape, input_dim),
        indices=torch.stack(indices_list, dim=-1).reshape(*leading_shape, active_count),
        codebook_vectors=torch.stack(vector_list, dim=1).reshape(
            *leading_shape,
            active_count,
            codebook_dim,
        ),
        latents=torch.stack(latent_list, dim=1).reshape(
            *leading_shape,
            active_count,
            codebook_dim,
        ),
        loss=loss,
        active_codebook_mask=active_mask.reshape(*leading_shape, active_count),
    )


def _run_all_active(
    *,
    flat_latents: Tensor,
    leading_shape: torch.Size,
    active_count: int,
    quantizers: nn.ModuleList,
    input_dim: int,
    codebook_dim: int,
) -> QuantizeOutput:
    residual = flat_latents
    quantized_sum = torch.zeros_like(flat_latents)
    indices_list: list[Tensor] = []
    vector_list: list[Tensor] = []
    latent_list: list[Tensor] = []
    commitment_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
    codebook_loss_sum = torch.zeros((), dtype=flat_latents.dtype, device=flat_latents.device)
    loss_count = 0

    for quantizer in quantizers[:active_count]:
        output = quantizer(residual)
        quantized_latents = output.quantized_latents
        residual = residual - quantized_latents
        quantized_sum = quantized_sum + quantized_latents

        indices_list.append(output.indices.reshape(-1))
        vector_list.append(_require_output_tensor(output.codebook_vectors, "codebook_vectors").reshape(-1, codebook_dim))
        latent_list.append(_require_output_tensor(output.latents, "latents").reshape(-1, codebook_dim))
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
        quantized_latents=quantized_sum.reshape(*leading_shape, input_dim),
        indices=torch.stack(indices_list, dim=-1).reshape(*leading_shape, active_count),
        codebook_vectors=torch.stack(vector_list, dim=1).reshape(
            *leading_shape,
            active_count,
            codebook_dim,
        ),
        latents=torch.stack(latent_list, dim=1).reshape(
            *leading_shape,
            active_count,
            codebook_dim,
        ),
        loss=loss,
        active_codebook_mask=torch.ones(
            flat_size,
            active_count,
            dtype=torch.bool,
            device=flat_latents.device,
        ).reshape(*leading_shape, active_count),
    )


def _require_output_tensor(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise RuntimeError(f"Residual quantizer output unexpectedly omitted {name}.")
    return value


def _empty_stage_output(
    selected_residual: Tensor,
    *,
    input_dim: int,
    codebook_dim: int,
) -> QuantizeOutput:
    """Build the shape-only result for a stage with no local assignments."""

    return QuantizeOutput(
        quantized_latents=selected_residual.new_empty(0, input_dim),
        indices=torch.empty(0, dtype=torch.long, device=selected_residual.device),
        codebook_vectors=selected_residual.new_empty(0, codebook_dim),
        latents=selected_residual.new_empty(0, codebook_dim),
        loss=None,
    )
