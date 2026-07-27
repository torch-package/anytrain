from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

from lightning import Trainer

from anytrain.lightning import experiment


class ExperimentTest(unittest.TestCase):
    def test_returns_supported_capabilities(self) -> None:
        writer = SimpleNamespace(
            add_text=lambda *args: None,
            add_scalar=lambda *args: None,
            add_audio=lambda *args, **kwargs: None,
            add_histogram=lambda *args: None,
        )
        trainer = _trainer(writer)

        self.assertIs(experiment.text(trainer), writer)
        self.assertIs(experiment.scalar(trainer), writer)
        self.assertIs(experiment.audio(trainer), writer)
        self.assertIs(experiment.histogram(trainer), writer)

    def test_returns_none_without_a_logger(self) -> None:
        trainer = cast(Trainer, SimpleNamespace(logger=None))

        self.assertIsNone(experiment.text(trainer))
        self.assertIsNone(experiment.scalar(trainer))
        self.assertIsNone(experiment.audio(trainer))
        self.assertIsNone(experiment.histogram(trainer))

    def test_returns_none_for_unsupported_capabilities(self) -> None:
        trainer = _trainer(SimpleNamespace(add_text=None))

        self.assertIsNone(experiment.text(trainer))
        self.assertIsNone(experiment.scalar(trainer))
        self.assertIsNone(experiment.audio(trainer))
        self.assertIsNone(experiment.histogram(trainer))


def _trainer(writer: object) -> Trainer:
    return cast(
        Trainer,
        SimpleNamespace(logger=SimpleNamespace(experiment=writer)),
    )


if __name__ == "__main__":
    unittest.main()
