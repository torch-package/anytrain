from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._validation import finite_real, positive_real, strict_bool
from ._observation import emit, recommend, register


class DPOLoss(nn.Module):
    """Direct Preference Optimization over sequence log-probabilities."""

    def __init__(
        self,
        *,
        beta: float = 0.1,
        reference_free: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = positive_real(beta, "beta")
        self.reference_free = strict_bool(reference_free, "reference_free")
        self.label_smoothing = finite_real(label_smoothing, "label_smoothing")
        if self.label_smoothing < 0 or self.label_smoothing >= 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5).")
        recommend(self)

    def forward(
        self,
        *,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor | None = None,
        ref_rejected_logps: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        """Compute DPO loss.

        Set ``validate=False`` only when the caller has already established the
        tensor shape, device, dtype, and finiteness contracts for this batch.
        """

        validate = strict_bool(validate, "validate")
        if validate:
            _validate_pair(policy_chosen_logps, policy_rejected_logps, name="policy")
        policy_margin = _logprob_margin(policy_chosen_logps, policy_rejected_logps)
        if self.reference_free:
            reference_margin = torch.zeros_like(policy_margin)
        else:
            if ref_chosen_logps is None or ref_rejected_logps is None:
                raise ValueError("reference log-probs are required unless reference_free=True.")
            if validate:
                _validate_pair(ref_chosen_logps, ref_rejected_logps, name="reference")
                if ref_chosen_logps.shape != policy_chosen_logps.shape:
                    raise ValueError("reference and policy log-probs must have the same shape.")
            reference_margin = _logprob_margin(ref_chosen_logps, ref_rejected_logps).detach()

        logits = self.beta * (policy_margin - reference_margin)
        losses = (
            -F.logsigmoid(logits) * (1.0 - self.label_smoothing)
            - F.logsigmoid(-logits) * self.label_smoothing
        )
        loss = losses.mean()
        emit(
            self,
            {
                "policy_margin": policy_margin.mean(),
                "reference_margin": reference_margin.mean(),
                "preference_logit": logits.mean(),
                "accuracy": (logits > 0).to(dtype=loss.dtype).mean(),
            },
        )
        return loss


def _logprob_margin(chosen: Tensor, rejected: Tensor) -> Tensor:
    dtype = torch.promote_types(torch.promote_types(chosen.dtype, rejected.dtype), torch.float32)
    return chosen.to(dtype=dtype) - rejected.to(dtype=dtype)


def _validate_pair(chosen: Tensor, rejected: Tensor, *, name: str) -> None:
    if not isinstance(chosen, Tensor) or not isinstance(rejected, Tensor):
        raise TypeError(f"{name} log-probs must be tensors.")
    if chosen.shape != rejected.shape:
        raise ValueError(f"{name} chosen and rejected log-probs must have the same shape.")
    if chosen.ndim < 1:
        raise ValueError(f"{name} log-probs must include a batch dimension.")
    if chosen.is_complex() or rejected.is_complex():
        raise TypeError(f"{name} log-probs must be real-valued.")
    if not bool(torch.isfinite(chosen).all()) or not bool(torch.isfinite(rejected).all()):
        raise ValueError(f"{name} log-probs must be finite.")


register(DPOLoss)


__all__ = ["DPOLoss"]
