from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from anytrain import observation

from ._output import Features, split
from .criterion import _LogitCriterion
from .feature import _FeatureMatching
from .penalty import _GradientPenalty
from .types import GAN, Preset, Reduction, _gan, _preset


class Loss(nn.Module):
    def __init__(
        self,
        discriminator: nn.Module,
        *,
        gan: GAN | str = GAN.Hinge,
        reduction: Reduction | str = Reduction.Mean,
        feature_weight: float = 0.0,
        gp_weight: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(discriminator, nn.Module):
            raise TypeError("discriminator must be a torch.nn.Module.")
        if feature_weight < 0:
            raise ValueError("feature_weight must be non-negative.")
        if gp_weight is not None and gp_weight < 0:
            raise ValueError("gp_weight must be non-negative.")

        self.gan = _gan(gan)
        self.reduction = Reduction(reduction)
        self.discriminator = discriminator
        self._criterion = _LogitCriterion(self.gan, reduction=self.reduction)
        self.feature_weight = float(feature_weight)
        self._feature = (
            _FeatureMatching(reduction=self.reduction) if self.feature_weight > 0 else None
        )
        self.gp_weight = (
            10.0 if gp_weight is None and self.gan == GAN.WGAN else float(gp_weight or 0.0)
        )
        self._gp = _GradientPenalty(reduction=self.reduction) if self.gp_weight > 0 else None
        observation.registry.recommend(self)

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        gan: GAN | str = GAN.Hinge,
        reduction: Reduction | str = Reduction.Mean,
        feature_weight: float = 0.0,
        gp_weight: float | None = None,
        **kwargs,
    ) -> Loss:
        resolved = _preset(preset)
        if resolved == Preset.DAC:
            from .audio import DACDiscriminator

            discriminator = DACDiscriminator(**kwargs)
        else:
            raise NotImplementedError(f"Unsupported GAN preset: {resolved.value}")
        return cls(
            discriminator,
            gan=gan,
            reduction=reduction,
            feature_weight=feature_weight,
            gp_weight=gp_weight,
        )

    def forward(
        self,
        fake: Tensor,
        real: Tensor | None = None,
        *,
        objective: Literal["discriminator", "generator"],
    ) -> Tensor:
        if objective == "discriminator":
            if real is None:
                raise ValueError("real must be provided for the discriminator objective.")
            return self._discriminator_loss(fake, real)
        if objective == "generator":
            return self._generator_loss(fake, real)
        raise ValueError("objective must be 'discriminator' or 'generator'.")

    def discriminator_loss(self, fake: Tensor, real: Tensor) -> Tensor:
        return self(fake, real, objective="discriminator")

    def generator_loss(self, fake: Tensor, real: Tensor | None = None) -> Tensor:
        return self(fake, real, objective="generator")

    def _discriminator_loss(self, fake: Tensor, real: Tensor) -> Tensor:
        _, fake_logits = self._discriminate(fake.detach())
        _, real_logits = self._discriminate(real)
        real_loss, fake_loss = self._criterion.discriminator_components(
            real_logits,
            fake_logits,
        )
        total = real_loss + fake_loss
        observation.emit(
            self,
            "discriminator/components",
            {
                "real": observation.Curve(real_loss.detach()),
                "fake": observation.Curve(fake_loss.detach()),
            },
        )

        if self._gp is not None and self.gp_weight > 0:
            gp = self._gp(self.discriminator, fake.detach(), real.detach())
            total = total + total.new_tensor(self.gp_weight) * gp
            observation.emit(
                self,
                "discriminator/gradient_penalty",
                {"gp": observation.Curve(gp.detach())},
            )
        return total

    def _generator_loss(self, fake: Tensor, real: Tensor | None) -> Tensor:
        fake_features, fake_logits = self._discriminate(fake)
        adversarial = self._criterion.generator_loss(fake_logits)
        total = adversarial
        observation.emit(
            self,
            "generator/components",
            {"adv": observation.Curve(adversarial.detach())},
        )

        if self.feature_weight > 0:
            if real is None:
                raise ValueError("real must be provided when feature matching is enabled.")
            if self._feature is None:
                raise RuntimeError("feature_loss is not configured.")
            with torch.no_grad():
                real_features, _ = self._discriminate(real)
            feature = self._feature(fake_features, real_features)
            total = total + total.new_tensor(self.feature_weight) * feature
            observation.emit(
                self,
                "generator/feature_matching",
                {"feature": observation.Curve(feature.detach())},
            )
        return total

    def _discriminate(self, x: Tensor) -> tuple[Features, list[Tensor]]:
        return split(self.discriminator(x))


def _collect_discriminator_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> observation.ObservationResult:
    del module, inputs
    return _collect_loss(kwargs, output, objective="discriminator")


def _collect_generator_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> observation.ObservationResult:
    del module, inputs
    return _collect_loss(kwargs, output, objective="generator")


def _collect_loss(
    kwargs: Mapping[str, Any],
    output: object,
    *,
    objective: str,
) -> observation.Curve | None:
    if kwargs.get("objective") != objective:
        return None
    if not isinstance(output, Tensor):
        raise TypeError("GAN loss observation requires a Tensor module output.")
    return observation.Curve(output)


def _collect_weights(module: nn.Module) -> dict[str, observation.Observation]:
    loss = cast(Loss, module)
    reference = next(loss.parameters(), None)
    if reference is None:
        reference = next(loss.buffers(), None)
    device = torch.device("cpu") if reference is None else reference.device
    return {
        "feature_weight": observation.Curve(
            torch.tensor(loss.feature_weight, dtype=torch.float32, device=device)
        ),
        "gp_weight": observation.Curve(
            torch.tensor(loss.gp_weight, dtype=torch.float32, device=device)
        ),
    }


observation.registry.register(
    Loss,
    (
        observation.OutputObservation(
            "discriminator/loss",
            _collect_discriminator_loss,
            recommended=True,
        ),
        observation.ForwardEvent("discriminator/components"),
        observation.ForwardEvent("discriminator/gradient_penalty"),
        observation.OutputObservation(
            "generator/loss",
            _collect_generator_loss,
            recommended=True,
        ),
        observation.ForwardEvent("generator/components"),
        observation.ForwardEvent("generator/feature_matching"),
        observation.StateObservation(
            "weights",
            _collect_weights,
            recommended=True,
        ),
    ),
)
