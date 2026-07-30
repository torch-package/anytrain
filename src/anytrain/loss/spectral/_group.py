from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import nn

from ..group import LossGroup
from .transform import SpectralTransformCache


@runtime_checkable
class SpectralCacheAware(Protocol):
    def _forward_with_spectral_cache(
        self,
        *args: Any,
        spectral_cache: SpectralTransformCache,
        **kwargs: Any,
    ) -> Any: ...


class SpectralLossGroup(LossGroup):
    def _call_loss(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        spectral_cache = kwargs.pop("_spectral_cache", None)
        if spectral_cache is not None and isinstance(module, SpectralCacheAware):
            return module._forward_with_spectral_cache(
                *args,
                spectral_cache=spectral_cache,
                **kwargs,
            )
        return module(*args, **kwargs)

    def _forward_with_spectral_cache(
        self,
        *args: Any,
        spectral_cache: SpectralTransformCache,
        **kwargs: Any,
    ):
        return self(*args, _spectral_cache=spectral_cache, **kwargs)
