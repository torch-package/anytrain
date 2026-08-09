from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from anytrain.utils import is_signed_integer_dtype

from ._observation import emit_diagnostics, loss_with_diagnostics
from ._validation import scalar
from .abc import LossABC


@dataclass(frozen=True)
class PackedCodebookLogits:
    """Valid-frame logits and labels for scalar codebook cross entropy."""

    logits: tuple[Tensor, ...]
    labels: Tensor


@loss_with_diagnostics
class MaskedCodebookCrossEntropyLoss(LossABC):
    """Mean cross entropy over every valid frame and codebook.

    ``include_top1`` enables optional per-codebook top-1 observations.
    ``validate=False`` is intended for trusted batches in a training hot path
    and skips input validation, but the reduction still rejects zero valid
    frames because their mean is undefined.
    """

    def compute_loss(
        self,
        logits: Sequence[Tensor] | PackedCodebookLogits,
        labels: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        validate: bool = True,
        include_top1: bool = False,
    ) -> Tensor:
        if isinstance(logits, PackedCodebookLogits):
            if labels is not None or mask is not None:
                raise TypeError(
                    "labels and mask must be omitted when logits is PackedCodebookLogits."
                )
            return self._compute_packed(
                logits,
                validate=validate,
                include_top1=include_top1,
            )
        if labels is None or mask is None:
            raise TypeError("labels and mask are required with unpacked codebook logits.")
        if validate:
            _validate_inputs(logits, labels, mask)

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
        )

    def _compute_packed(
        self,
        packed: PackedCodebookLogits,
        *,
        validate: bool,
        include_top1: bool,
    ) -> Tensor:
        if len(packed.logits) == 0:
            raise ValueError("codebook logits must provide at least one tensor.")
        if validate:
            _validate_packed_inputs(packed)
        if packed.labels.ndim == 0 or packed.labels.size(0) == 0:
            raise ValueError("codebook loss requires at least one valid frame.")

        labels = packed.labels.to(dtype=torch.long)
        losses: list[Tensor] = []
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

        total = torch.stack(losses).mean()
        diagnostics["valid_frames"] = total.new_tensor(labels.size(0))
        emit_diagnostics(self, diagnostics)
        return scalar(total)


def _validate_inputs(logits: Sequence[Tensor], labels: Tensor, mask: Tensor) -> None:
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


def _validate_packed_inputs(packed: PackedCodebookLogits) -> None:
    labels = packed.labels
    if labels.ndim != 2:
        raise ValueError("packed codebook labels must have shape [N, Q].")
    if not is_signed_integer_dtype(labels.dtype):
        raise TypeError("packed codebook labels must use a signed integer dtype.")
    if len(packed.logits) != labels.size(-1):
        raise ValueError("packed codebook logits must provide one tensor per codebook.")
    if any(value.device != labels.device for value in packed.logits):
        raise ValueError("packed codebook logits and labels must share a device.")
    for value in packed.logits:
        if value.ndim != 2 or value.size(0) != labels.size(0):
            raise ValueError(
                "packed codebook logits must have shape [N, vocab] and align with labels."
            )
    limits = torch.tensor(
        [value.size(-1) for value in packed.logits],
        device=labels.device,
        dtype=torch.long,
    )
    valid_targets = labels.to(dtype=torch.long)
    if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
        raise ValueError("packed codebook target is outside its logits vocabulary.")

__all__ = ["MaskedCodebookCrossEntropyLoss", "PackedCodebookLogits"]
