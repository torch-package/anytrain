from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from anytrain.utils import is_signed_integer_dtype

from ._validation import scalar
from .abc import LossABC
from .observation import emit_diagnostics, loss_with_diagnostics


@dataclass(frozen=True)
class PackedCodebookLogits:
    """Valid-frame logits and labels for scalar codebook losses."""

    # Keep this as a generic sequence at the boundary: callers commonly build
    # packed codebooks incrementally and hand us a list.  The dense path still
    # materializes a tuple, but validation should not reject an equivalent list.
    logits: Sequence[Tensor]
    labels: Tensor


@loss_with_diagnostics
class MaskedCodebookCrossEntropyLoss(LossABC):
    """Codebook cross entropy with an optional probability-weighted geometry term.

    With ``geometry_weight > 0``, the loss adds

    ``geometry_weight * mean_frame(sum_i p_i ||e_anchor - e_i||_2^2 / D)``

    for every valid frame and codebook, where ``p`` is the softmax probability
    distribution and ``e_anchor`` is either the top-1 embedding (the requested
    ``e_x``) or the target embedding. The target anchor is usually the more
    useful choice because it preserves a direct geometric signal toward the
    correct token; the top-1 anchor is provided for compatibility with the
    original formulation and acts mainly as a confidence regularizer.

    ``include_top1`` enables optional per-codebook top-1 observations.
    ``validate=False`` is intended for trusted batches in a training hot path
    and skips explicit input validation, but the reduction still rejects zero
    valid frames because their mean is undefined. Unpacked inputs still compact
    valid frames with ``nonzero()``; callers that also need to avoid dynamic
    index materialization should provide ``PackedCodebookLogits`` directly.
    """

    def __init__(
        self,
        *,
        geometry_weight: float = 0.0,
        geometry_anchor: str = "top1",
        geometry_temperature: float = 1.0,
        detach_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.geometry_weight = _finite_nonnegative(geometry_weight, "geometry_weight")
        if not isinstance(geometry_anchor, str):
            raise TypeError("geometry_anchor must be a string.")
        if geometry_anchor not in {"top1", "target"}:
            raise ValueError(
                "geometry_anchor must be either 'top1' or 'target', "
                f"got {geometry_anchor!r}."
            )
        self.geometry_anchor = geometry_anchor
        self.geometry_temperature = _finite_positive(
            geometry_temperature, "geometry_temperature"
        )
        if not isinstance(detach_embeddings, bool):
            raise TypeError("detach_embeddings must be a bool.")
        self.detach_embeddings = detach_embeddings

    def compute_loss(
        self,
        logits: Sequence[Tensor] | PackedCodebookLogits,
        labels: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
        validate: bool = True,
        include_top1: bool = False,
    ) -> Tensor:
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool.")
        if not isinstance(include_top1, bool):
            raise TypeError("include_top1 must be a bool.")
        if isinstance(logits, PackedCodebookLogits):
            if labels is not None or mask is not None:
                raise TypeError(
                    "labels and mask must be omitted when logits is PackedCodebookLogits."
                )
            return self._compute_packed(
                logits,
                validate=validate,
                include_top1=include_top1,
                codebook_embeddings=codebook_embeddings,
            )
        if not isinstance(logits, Sequence):
            raise TypeError("codebook logits must be a sequence of Tensor values.")
        if labels is None or mask is None:
            raise TypeError("labels and mask are required with unpacked codebook logits.")
        if validate:
            _validate_inputs(logits, labels, mask)
            if self.geometry_weight > 0 and codebook_embeddings is not None:
                _validate_embeddings(codebook_embeddings, logits, labels)

        frame_indices = mask.flatten().nonzero().flatten()
        packed = PackedCodebookLogits(
            logits=tuple(
                value.flatten(0, 1).index_select(0, frame_indices)
                for value in logits
            ),
            labels=labels.flatten(0, 1).index_select(0, frame_indices),
        )
        return self._compute_packed(
            packed,
            validate=False,
            include_top1=include_top1,
            codebook_embeddings=codebook_embeddings,
        )

    def _compute_packed(
        self,
        packed: PackedCodebookLogits,
        *,
        validate: bool,
        include_top1: bool,
        codebook_embeddings: Sequence[Tensor] | None,
    ) -> Tensor:
        if len(packed.logits) == 0:
            raise ValueError("codebook logits must provide at least one tensor.")
        if validate:
            _validate_packed_inputs(packed)
        if packed.labels.ndim == 0 or packed.labels.size(0) == 0:
            raise ValueError("codebook loss requires at least one valid frame.")
        if self.geometry_weight > 0 and codebook_embeddings is None:
            raise ValueError(
                "codebook_embeddings are required when geometry_weight is positive."
            )
        if self.geometry_weight > 0 and codebook_embeddings is not None and validate:
            _validate_embeddings(codebook_embeddings, packed.logits, packed.labels)

        labels = packed.labels.to(dtype=torch.long)
        losses: list[Tensor] = []
        geometry_losses: list[Tensor] = []
        diagnostics: dict[str, Tensor] = {}
        for codebook, value in enumerate(packed.logits):
            target = labels[:, codebook]
            loss = F.cross_entropy(value, target)
            losses.append(loss)
            diagnostics[f"codebook_{codebook}"] = loss
            if include_top1:
                diagnostics[f"codebook_{codebook}_top1"] = (
                    value.argmax(dim=-1).eq(target).to(dtype=loss.dtype).mean()
                )
            if self.geometry_weight > 0:
                if codebook_embeddings is None:  # guarded above; keeps type checkers precise
                    raise RuntimeError("codebook embeddings unexpectedly missing.")
                geometry = _geometry_loss(
                    value,
                    target,
                    codebook_embeddings[codebook],
                    anchor=self.geometry_anchor,
                    temperature=self.geometry_temperature,
                    detach_embeddings=self.detach_embeddings,
                )
                geometry_losses.append(geometry)
                diagnostics[f"codebook_{codebook}_geometry"] = geometry

        cross_entropy = torch.stack(losses).mean()
        if geometry_losses:
            geometry = torch.stack(geometry_losses).mean()
            total = cross_entropy + self.geometry_weight * geometry
            diagnostics["geometry"] = geometry
            diagnostics["geometry_weighted"] = self.geometry_weight * geometry
        else:
            total = cross_entropy
        diagnostics["valid_frames"] = total.new_tensor(labels.size(0))
        diagnostics["cross_entropy"] = cross_entropy
        emit_diagnostics(self, diagnostics)
        return scalar(total)


def _validate_inputs(logits: Sequence[Tensor], labels: Tensor, mask: Tensor) -> None:
    if not isinstance(labels, Tensor):
        raise TypeError("codebook labels must be a Tensor.")
    if not isinstance(mask, Tensor):
        raise TypeError("codebook target mask must be a Tensor.")
    if labels.ndim != 3 or mask.shape != labels.shape[:2]:
        raise ValueError(
            "codebook labels and mask must have shapes [B, F, Q] and [B, F]."
        )
    if not is_signed_integer_dtype(labels.dtype):
        raise TypeError("codebook labels must use a signed integer dtype.")
    if mask.dtype != torch.bool:
        raise TypeError("codebook target mask must be boolean.")
    if len(logits) != labels.size(-1):
        raise ValueError("codebook logits must provide one tensor per codebook.")
    if len(logits) == 0:
        raise ValueError("codebook logits must provide at least one tensor.")
    for value in logits:
        if not isinstance(value, Tensor):
            raise TypeError("codebook logits must contain Tensor values.")
        if value.ndim != 3 or value.shape[:2] != labels.shape[:2]:
            raise ValueError(
                "codebook logits must have shape [B, F, vocab] and align with labels."
            )
    if mask.device != labels.device or any(value.device != labels.device for value in logits):
        raise ValueError("codebook logits, labels, and mask must share a device.")
    valid_targets = labels[mask].to(dtype=torch.long)
    limits = torch.tensor(
        [value.size(-1) for value in logits],
        device=labels.device,
        dtype=torch.long,
    )
    if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
        raise ValueError("valid codebook target is outside its logits vocabulary.")


def _validate_embeddings(
    embeddings: Sequence[Tensor],
    logits: Sequence[Tensor],
    labels: Tensor,
) -> None:
    if len(embeddings) != len(logits):
        raise ValueError(
            "codebook_embeddings must provide one [vocab, dimension] tensor per codebook."
        )
    for codebook, (embedding, value) in enumerate(zip(embeddings, logits, strict=True)):
        if not isinstance(embedding, Tensor):
            raise TypeError(f"codebook embedding {codebook} must be a tensor.")
        if embedding.ndim != 2 or embedding.size(0) != value.size(-1):
            raise ValueError(
                f"codebook embedding {codebook} must have shape ["
                f"{value.size(-1)}, dimension]."
            )
        if embedding.size(1) == 0:
            raise ValueError(f"codebook embedding {codebook} dimension must be positive.")
        if not embedding.is_floating_point():
            raise TypeError(f"codebook embedding {codebook} must use a floating dtype.")
        if embedding.device != labels.device:
            raise ValueError("codebook embeddings, logits, and labels must share a device.")


def _geometry_loss(
    logits: Tensor,
    labels: Tensor,
    embeddings: Tensor,
    *,
    anchor: str,
    temperature: float,
    detach_embeddings: bool,
) -> Tensor:
    table = embeddings.float()
    if detach_embeddings:
        table = table.detach()
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    if anchor == "top1":
        anchor_indices = logits.argmax(dim=-1)
    elif anchor == "target":
        anchor_indices = labels
    else:  # pragma: no cover - constructor validates this option
        raise RuntimeError(f"unsupported geometry anchor {anchor!r}.")
    anchor_embedding = table.index_select(0, anchor_indices)
    # Expand sum_i p_i ||e_anchor - e_i||^2 algebraically to avoid a
    # temporary [N, vocab, dimension] tensor for large codebooks.
    expected_embedding = probabilities @ table
    expected_squared_norm = probabilities @ table.square().sum(dim=-1)
    anchor_squared_norm = anchor_embedding.square().sum(dim=-1)
    distances = (
        anchor_squared_norm
        - 2 * (anchor_embedding * expected_embedding).sum(dim=-1)
        + expected_squared_norm
    )
    return distances.clamp_min(0).div(table.size(-1)).mean()


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _finite_positive(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_packed_inputs(packed: PackedCodebookLogits) -> None:
    labels = packed.labels
    if not isinstance(labels, Tensor):
        raise TypeError("packed codebook labels must be a Tensor.")
    if isinstance(packed.logits, (str, bytes)) or not isinstance(
        packed.logits, Sequence
    ):
        raise TypeError("packed codebook logits must be a sequence of Tensor values.")
    if labels.ndim != 2:
        raise ValueError("packed codebook labels must have shape [N, Q].")
    if not is_signed_integer_dtype(labels.dtype):
        raise TypeError("packed codebook labels must use a signed integer dtype.")
    if len(packed.logits) != labels.size(-1):
        raise ValueError("packed codebook logits must provide one tensor per codebook.")
    for value in packed.logits:
        if not isinstance(value, Tensor):
            raise TypeError("packed codebook logits must contain Tensor values.")
        if value.ndim != 2 or value.size(0) != labels.size(0):
            raise ValueError(
                "packed codebook logits must have shape [N, vocab] and align with labels."
            )
    if any(value.device != labels.device for value in packed.logits):
        raise ValueError("packed codebook logits and labels must share a device.")
    limits = torch.tensor(
        [value.size(-1) for value in packed.logits],
        device=labels.device,
        dtype=torch.long,
    )
    valid_targets = labels.to(dtype=torch.long)
    if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
        raise ValueError("packed codebook target is outside its logits vocabulary.")

__all__ = ["MaskedCodebookCrossEntropyLoss", "PackedCodebookLogits"]
