from __future__ import annotations

from collections.abc import Mapping


class StepInterval:
    """Run at most once for each matching monotonically increasing step."""

    def __init__(self, *, every_n_steps: int) -> None:
        if isinstance(every_n_steps, bool) or not isinstance(every_n_steps, int):
            raise TypeError("every_n_steps must be an integer.")
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive.")
        self.every_n_steps = every_n_steps
        self._last_step: int | None = None

    def should_run(self, step: int) -> bool:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer.")
        if step < 0:
            raise ValueError("step must be non-negative.")
        if self._last_step == step:
            return False
        if self._last_step is not None and step < self._last_step:
            raise RuntimeError("step moved backwards.")
        self._last_step = step
        return step % self.every_n_steps == 0

    def state_dict(self) -> dict[str, int | None]:
        return {"last_step": self._last_step}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"last_step"}:
            raise ValueError("step interval state must contain only last_step.")
        last_step = state["last_step"]
        if last_step is None:
            self._last_step = None
            return
        if isinstance(last_step, bool) or not isinstance(last_step, int):
            raise TypeError("interval last_step must be an integer or None.")
        if last_step < 0:
            raise ValueError("interval last_step must be non-negative.")
        self._last_step = last_step


__all__ = ["StepInterval"]
