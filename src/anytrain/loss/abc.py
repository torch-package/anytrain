from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from ._observation import recommend
from ._validation import scalar


class LossABC(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        recommend(self)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return scalar(self.compute_loss(*args, **kwargs))

    @abstractmethod
    def compute_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError
