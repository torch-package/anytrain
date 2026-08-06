from __future__ import annotations

from typing import Any

from torch import nn

from ..group import LossGroup


class SpectralLossGroup(LossGroup):
    def _call_loss(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        spectral_cache = kwargs.pop("_spectral_cache", None)
        if spectral_cache is not None and isinstance(module, SpectralLossGroup):
            return module(
                *args,
                _spectral_cache=spectral_cache,
                **kwargs,
            )
        return module(*args, **kwargs)
