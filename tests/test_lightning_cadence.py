from __future__ import annotations

import pytest

from anytrain.lightning import StepInterval


def test_step_interval_deduplicates_and_restores() -> None:
    interval = StepInterval(every_n_steps=2)

    assert interval.should_run(0)
    assert not interval.should_run(0)
    assert not interval.should_run(1)
    assert interval.should_run(2)

    restored = StepInterval(every_n_steps=2)
    restored.load_state_dict(interval.state_dict())
    assert not restored.should_run(2)
    assert not restored.should_run(3)
    assert restored.should_run(4)


def test_step_interval_rejects_invalid_steps_and_state() -> None:
    interval = StepInterval(every_n_steps=1)

    with pytest.raises(ValueError, match="non-negative"):
        interval.should_run(-1)
    interval.should_run(2)
    with pytest.raises(RuntimeError, match="backwards"):
        interval.should_run(1)
    with pytest.raises(ValueError, match="only last_step"):
        interval.load_state_dict({})
    with pytest.raises(TypeError, match="integer or None"):
        interval.load_state_dict({"last_step": 1.0})
