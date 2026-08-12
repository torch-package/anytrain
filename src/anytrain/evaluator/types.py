from __future__ import annotations

import torch

MetricValue = float | torch.Tensor
MetricDict = dict[str, MetricValue]

__all__ = ["MetricDict", "MetricValue"]
