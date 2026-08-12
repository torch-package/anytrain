from __future__ import annotations

import torch
from torch import Tensor, nn

from .._validation import positive_int
from ._deps import DiscretePathSample, MixtureDiscreteProbPath, PolynomialConvexScheduler
from ._runtime import require_batch, require_pair
from .sampler import DiscreteEulerSampler
from .source import MaskTokenSource, UniformTokenSource
from .time import DEFAULT_TIME_EPS, UniformTimeSampler
from .types import FlowSampleOutput, ModelCaller, Source, TimeSampler, default_call_model


class DiscreteFlowRuntime:
    def __init__(
        self,
        vocab_size: int,
        *,
        path: MixtureDiscreteProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        sampler: DiscreteEulerSampler | None = None,
        call_model: ModelCaller = default_call_model,
    ):
        self.vocab_size = positive_int(vocab_size, "vocab_size")
        self.path = (
            MixtureDiscreteProbPath(PolynomialConvexScheduler(n=2.0)) if path is None else path
        )
        self.source = UniformTokenSource(self.vocab_size) if source is None else source
        if isinstance(self.source, MaskTokenSource) and self.source.mask_id >= self.vocab_size:
            raise ValueError(
                "MaskTokenSource.mask_id must be in [0, vocab_size) because the wrapped "
                f"Facebook discrete backends gather/one-hot over vocab_size={self.vocab_size}; "
                f"got mask_id={self.source.mask_id}. Include the mask token in vocab_size "
                "(for example vocab_size=1025 with mask_id=1024)."
            )
        self.time_sampler = (
            UniformTimeSampler(t_max=1.0 - DEFAULT_TIME_EPS)
            if time_sampler is None
            else time_sampler
        )
        self.sampler = DiscreteEulerSampler() if sampler is None else sampler
        self.call_model = call_model

    def source_like(self, x_1: Tensor) -> Tensor:
        return self.source.sample_like(x_1)

    def training_sample(
        self,
        x_1: Tensor,
        *,
        x_0: Tensor | None = None,
    ) -> DiscretePathSample:
        require_batch(x_1, "x_1")
        if x_1.dtype != torch.long:
            raise TypeError(f"x_1 must have dtype torch.long, got {x_1.dtype}.")
        if x_0 is None:
            x_0 = self.source_like(x_1)
        elif x_0.dtype != torch.long:
            raise TypeError(f"x_0 must have dtype torch.long, got {x_0.dtype}.")
        require_pair(x_0, x_1)
        t = self.time_sampler.sample(x_1.shape[0], x_1.device)
        return self.path.sample(x_0=x_0, x_1=x_1, t=t)

    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        **model_extras: object,
    ) -> FlowSampleOutput:
        return self.sampler.sample(
            model,
            x_0,
            vocab_size=self.vocab_size,
            path=self.path,
            call_model=self.call_model,
            **model_extras,
        )


__all__ = ["DiscreteFlowRuntime"]
