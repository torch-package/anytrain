from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anytrain.lightning import ModelCheckpoint, validation
from anytrain.lightning.schedule import (
    Constant,
    LRSchedule,
    Phase,
    ScheduleRuntime,
    UnitBatch,
    UnitClock,
)


class _UnitProvider:
    def __call__(self, *, batch: Any, **kwargs: Any) -> UnitBatch:
        del kwargs
        inputs, _ = batch
        return UnitBatch(valid=float(inputs.numel()), unit="tokens")


class _Module(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.validation_steps: list[int] = []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        inputs, targets = batch
        return (self.layer(inputs) - targets).square().mean()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        self.validation_steps.append(self.global_step)
        self.log("val/loss", 1.0, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _runtime(*, sync_distributed: bool = True) -> ScheduleRuntime:
    return ScheduleRuntime(
        LRSchedule(
            clock=UnitClock(
                unit="tokens",
                provider=_UnitProvider(),
                units_per_epoch=2.0,
                sync_cuda=False,
                sync_distributed=sync_distributed,
            ),
            phases=[Phase("main", duration=100.0, lr=Constant(1.0))],
        )
    )


def test_unit_epochs_trigger_validation_and_checkpoint(tmp_path: Path) -> None:
    runtime = _runtime()
    schedule = runtime.unit_callback()
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        filename="step-{step}",
        save_last=False,
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
        async_save=False,
        unit_schedule=schedule,
        every_n_unit_epochs=1,
    )
    module = _Module()
    data = TensorDataset(torch.ones(8, 1), torch.zeros(8, 1))
    validation_data = TensorDataset(torch.ones(1, 1), torch.zeros(1, 1))
    trainer = Trainer(
        accelerator="cpu",
        # Consumers register with the schedule; their Trainer callback-list order
        # must not affect the shared unit-boundary decision.
        callbacks=[checkpoint, validation.UnitEpochInterval(schedule), schedule],
        check_val_every_n_epoch=None,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_epochs=-1,
        max_steps=4,
        num_sanity_val_steps=0,
        val_check_interval=1,
    )

    trainer.fit(
        module,
        train_dataloaders=DataLoader(data, batch_size=1),
        val_dataloaders=DataLoader(validation_data, batch_size=1),
    )

    assert module.validation_steps == [2, 4]
    assert schedule.total_valid == 4.0
    assert schedule.completed_unit_epochs == 2
    assert sorted(path.name for path in tmp_path.glob("*.ckpt")) == [
        "step-2.ckpt",
        "step-4.ckpt",
    ]


def test_distributed_unit_epoch_checkpoint_requires_global_clock(
    tmp_path: Path,
) -> None:
    runtime = _runtime(sync_distributed=False)
    schedule = runtime.unit_callback()
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        save_top_k=-1,
        async_save=False,
        unit_schedule=schedule,
        every_n_unit_epochs=1,
    )
    trainer = type(
        "TrainerStub",
        (),
        {"callbacks": [schedule, checkpoint], "world_size": 2},
    )()

    with pytest.raises(RuntimeError, match="globally synchronized"):
        checkpoint.on_train_start(trainer, _Module())  # type: ignore[arg-type]
