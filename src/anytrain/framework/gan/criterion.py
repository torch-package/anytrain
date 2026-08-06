from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import GAN, Reduction, _gan


class _LogitCriterion(nn.Module):
    def __init__(
        self,
        gan: GAN | str = GAN.Hinge,
        *,
        reduction: Reduction | str = Reduction.Mean,
    ) -> None:
        super().__init__()
        self.gan = _gan(gan)
        self.reduction = Reduction(reduction)

    def discriminator_loss(
        self,
        real: Sequence[Tensor],
        fake: Sequence[Tensor],
    ) -> Tensor:
        real_loss, fake_loss = self.discriminator_components(real, fake)
        return real_loss + fake_loss

    def discriminator_components(
        self,
        real: Sequence[Tensor],
        fake: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if len(real) != len(fake):
            raise ValueError("real and fake logits must have the same branch count.")
        return self.real_loss(real), self.fake_loss(fake)

    def generator_loss(self, fake: Sequence[Tensor]) -> Tensor:
        return self._reduce([self._adv_loss(logits) for logits in fake])

    def real_loss(self, real: Sequence[Tensor]) -> Tensor:
        return self._reduce([self._real_loss(logits) for logits in real])

    def fake_loss(self, fake: Sequence[Tensor]) -> Tensor:
        return self._reduce([self._fake_loss(logits) for logits in fake])

    def _real_loss(self, real_logits: Tensor) -> Tensor:
        if self.gan == GAN.Hinge:
            return F.relu(1 - real_logits).mean()
        if self.gan == GAN.LSGAN:
            return torch.mean((1 - real_logits) ** 2)
        if self.gan == GAN.WGAN:
            return -real_logits.mean()
        raise ValueError(f"Unsupported GAN type {self.gan!r}.")

    def _fake_loss(self, fake_logits: Tensor) -> Tensor:
        if self.gan == GAN.Hinge:
            return F.relu(1 + fake_logits).mean()
        if self.gan == GAN.LSGAN:
            return (fake_logits**2).mean()
        if self.gan == GAN.WGAN:
            return fake_logits.mean()
        raise ValueError(f"Unsupported GAN type {self.gan!r}.")

    def _adv_loss(self, fake_logits: Tensor) -> Tensor:
        if self.gan in {GAN.Hinge, GAN.WGAN}:
            return -fake_logits.mean()
        if self.gan == GAN.LSGAN:
            return torch.mean((1 - fake_logits) ** 2)
        raise ValueError(f"Unsupported GAN type {self.gan!r}.")

    def _reduce(self, losses: list[Tensor]) -> Tensor:
        if not losses:
            raise ValueError("at least one discriminator branch is required.")
        stacked = torch.stack(losses)
        if self.reduction == Reduction.Mean:
            return stacked.mean()
        if self.reduction == Reduction.Sum:
            return stacked.sum()
        raise ValueError(f"Unsupported GAN reduction {self.reduction!r}.")
