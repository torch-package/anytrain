from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from functools import partial
from numbers import Real
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class TinyRegressionModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 0.01,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(lr, bool) or not isinstance(lr, Real):
            raise TypeError("lr must be a host real number.")
        lr = float(lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive.")
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        if self.optimizer is not None:
            return self.optimizer(self.parameters())
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TinyRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_samples: int = 64,
        input_dim: int = 4,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_samples", num_samples),
            ("input_dim", input_dim),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(self.num_samples, self.input_dim, generator=generator)
        y = x.sum(dim=-1, keepdim=True)
        self.dataset = TensorDataset(x, y)

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("setup() must run before train_dataloader().")
        return DataLoader(self.dataset, batch_size=self.batch_size)


def run_tiny_regression(
    *,
    max_epochs: int = 1,
    num_samples: int = 64,
    input_dim: int = 4,
    batch_size: int = 16,
    default_root_dir: str | Path = "outputs/anytrain/tiny_regression",
    enable_progress_bar: bool = True,
) -> tuple[pl.Trainer, TinyRegressionModule]:
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int):
        raise TypeError("max_epochs must be an integer.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if not isinstance(default_root_dir, (str, Path)):
        raise TypeError("default_root_dir must be a string or Path.")
    if isinstance(default_root_dir, str) and not default_root_dir.strip():
        raise ValueError("default_root_dir must not be empty.")
    if not isinstance(enable_progress_bar, bool):
        raise TypeError("enable_progress_bar must be a bool.")

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(0, workers=True)

    module = TinyRegressionModule(
        model=torch.nn.Linear(input_dim, 1),
        optimizer=partial(torch.optim.Adam, lr=0.01),
    )
    data_module = TinyRegressionDataModule(
        num_samples=num_samples,
        input_dim=input_dim,
        batch_size=batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=enable_progress_bar,
        default_root_dir=default_root_dir,
    )
    trainer.fit(module, datamodule=data_module)
    return trainer, module


def main() -> None:
    run_tiny_regression()


if __name__ == "__main__":
    main()
