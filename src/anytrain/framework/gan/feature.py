from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anytrain._compat import strict_zip

from ._output import Features, validate_matching_features
from .types import Reduction


class _FeatureMatching(nn.Module):
    def __init__(
        self,
        *,
        reduction: Reduction | str = Reduction.Mean,
        detach_real: bool = True,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.l1_loss,
    ) -> None:
        super().__init__()
        self.reduction = Reduction(reduction)
        self.detach_real = detach_real
        self.loss_fn = loss_fn

    def forward(
        self,
        fake: Features,
        real: Features,
    ) -> Tensor:
        validate_matching_features(fake, real, require_features=True)
        losses: list[Tensor] = []
        for fake_branch, real_branch in strict_zip(fake, real):
            for fake_feature, real_feature in strict_zip(fake_branch, real_branch):
                target = real_feature.detach() if self.detach_real else real_feature
                loss = self.loss_fn(fake_feature, target)
                if not isinstance(loss, Tensor):
                    raise TypeError("feature matching loss_fn must return a Tensor.")
                if loss.ndim != 0:
                    raise ValueError("feature matching loss_fn must return a scalar tensor.")
                losses.append(loss)

        return self._reduce(losses)

    def _reduce(self, losses: list[Tensor]) -> Tensor:
        if not losses:
            raise ValueError("feature matching requires at least one feature map.")
        stacked = torch.stack(losses)
        if self.reduction == Reduction.Mean:
            return stacked.mean()
        if self.reduction == Reduction.Sum:
            return stacked.sum()
        raise ValueError(f"Unsupported GAN reduction {self.reduction!r}.")
