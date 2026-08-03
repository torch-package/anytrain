from __future__ import annotations

from dataclasses import dataclass

import lightning.pytorch as pl

from .. import _capability


@dataclass
class StepCadence:
    every_n_steps: int | None
    _last_logged_step: int | None = None

    def __post_init__(self) -> None:
        value = self.every_n_steps
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("every_n_steps must be an integer.")
        if value <= 0:
            raise ValueError("every_n_steps must be positive.")

    def should_log(self, trainer: pl.Trainer) -> bool:
        if self.every_n_steps is None:
            return True
        step = _capability.global_step(trainer)
        if step % self.every_n_steps != 0 or self._last_logged_step == step:
            return False
        self._last_logged_step = step
        return True


__all__ = ["StepCadence"]
