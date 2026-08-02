from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch

from ._callback import UnitScheduleCallback
from ._spec import LRSchedule, StepClock, UnitClock


class ScheduleRuntime:
    """Compile an LR schedule into Lightning optimizer returns and callbacks."""

    def __init__(self, schedule: LRSchedule) -> None:
        if not isinstance(schedule, LRSchedule):
            raise TypeError("schedule runtime requires an LRSchedule.")
        self.schedule = schedule
        self._callbacks: list[pl.Callback] | None = None

    def configure_optimizers(self, optimizer: torch.optim.Optimizer) -> Any:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer.")
        if isinstance(self.schedule.clock, StepClock):
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: self.schedule.lr_ratio(float(step)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        if isinstance(self.schedule.clock, UnitClock):
            return optimizer
        raise TypeError("schedule clock must be StepClock or UnitClock.")

    def callbacks(self) -> list[pl.Callback]:
        if isinstance(self.schedule.clock, StepClock):
            return []
        if self._callbacks is None:
            self._callbacks = [UnitScheduleCallback(self.schedule)]
        return list(self._callbacks)


__all__ = ["ScheduleRuntime"]
