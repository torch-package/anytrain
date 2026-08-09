"""Task-agnostic Lightning diagnostics for out-of-memory failures."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from typing import NotRequired, TypedDict, cast

import torch
from lightning import pytorch as pl
from torch import Tensor


class OOMContext(TypedDict):
    phase: str
    inputs: dict[str, object]
    batch_idx: NotRequired[int]
    dataloader_idx: NotRequired[int]


InputReporter = Callable[[object], Mapping[str, object]]
_CONTEXT_KEY = "anytrain_oom_context"


def annotate(
    exception: BaseException,
    *,
    phase: str,
    inputs: Mapping[str, object],
) -> bool:
    """Attach JSON-safe call-site metadata to an OOM without replacing it."""

    if not is_oom(exception):
        return False
    exception.__dict__[_CONTEXT_KEY] = OOMContext(
        phase=phase,
        inputs=dict(inputs),
    )
    return True


def context(exception: BaseException) -> OOMContext | None:
    value = exception.__dict__.get(_CONTEXT_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("OOM exception context must be a mapping.")
    return cast(OOMContext, value)


def is_oom(exception: BaseException) -> bool:
    """Recognize native Torch OOMs and compatible RuntimeError messages."""

    if isinstance(exception, torch.OutOfMemoryError):
        return True
    return isinstance(exception, RuntimeError) and "out of memory" in str(exception).lower()


def tensor_report(value: Tensor | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


class OOMDiagnostics(pl.Callback):
    """Capture Lightning phase context and report an uncaught OOM.

    ``input_reporter`` is deliberately supplied by the downstream project. It
    may summarize a project batch, request, or any other input object, but must
    return JSON-compatible mapping values and must not retain the input object.
    """

    def __init__(self, input_reporter: InputReporter | None = None) -> None:
        super().__init__()
        if input_reporter is not None and not callable(input_reporter):
            raise TypeError("OOM input_reporter must be callable when provided.")
        self.input_reporter = input_reporter
        self._context: dict[str, object] | None = None

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module
        self.capture(
            phase="train_step",
            inputs=self._inputs(batch),
            batch_idx=batch_idx,
        )

    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: Tensor,
    ) -> None:
        del trainer, pl_module, loss
        self._phase("train_backward")

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del trainer, pl_module, optimizer
        self._phase("train_optimizer")

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._phase("train_post_backward")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        self._context = None

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module
        self.capture(
            phase="validation_step",
            inputs=self._inputs(batch),
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        self._context = None

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        if not is_oom(exception):
            return
        write_report(
            trainer,
            pl_module,
            exception,
            context(exception) or self._context,
        )

    def capture(
        self,
        *,
        phase: str,
        inputs: Mapping[str, object],
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> None:
        captured: dict[str, object] = {
            "phase": phase,
            "inputs": dict(inputs),
        }
        if batch_idx is not None:
            captured["batch_idx"] = batch_idx
        if dataloader_idx is not None:
            captured["dataloader_idx"] = dataloader_idx
        self._context = captured

    def _inputs(self, batch: object) -> Mapping[str, object]:
        if self.input_reporter is None:
            return {"type": type(batch).__name__}
        report = self.input_reporter(batch)
        if not isinstance(report, Mapping):
            raise TypeError("OOM input_reporter must return a mapping.")
        return report

    def _phase(self, phase: str) -> None:
        if self._context is not None:
            self._context["phase"] = phase


def report_oom(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    exception: BaseException,
    *,
    phase: str,
    inputs: Mapping[str, object],
) -> bool:
    """Record an OOM at an explicit call site, using a configured callback."""

    if not is_oom(exception):
        return False
    attached = context(exception)
    if attached is not None:
        phase = attached["phase"]
        inputs = attached["inputs"]
    for callback in getattr(trainer, "callbacks", ()):
        if isinstance(callback, OOMDiagnostics):
            callback.capture(phase=phase, inputs=inputs)
            return True
    write_report(
        trainer,
        pl_module,
        exception,
        {"phase": phase, "inputs": dict(inputs)},
    )
    return True


def write_report(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    exception: BaseException,
    context_value: Mapping[str, object] | None,
) -> None:
    payload: dict[str, object] = {
        "event": "out_of_memory",
        "error": {
            "type": type(exception).__name__,
            "message": str(exception),
        },
        "epoch": int(getattr(trainer, "current_epoch", 0)),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "global_rank": int(getattr(trainer, "global_rank", 0)),
        "local_rank": int(getattr(trainer, "local_rank", 0)),
        "context": (
            {"phase": "outside_batch", "inputs": None}
            if context_value is None
            else dict(context_value)
        ),
        "cuda": cuda_report(pl_module),
    }
    print(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def cuda_report(pl_module: pl.LightningModule) -> dict[str, object] | None:
    device = getattr(pl_module, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    try:
        return {
            "device": str(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    except RuntimeError as error:
        return {
            "device": str(device),
            "error": f"{type(error).__name__}: {error}",
        }


__all__ = [
    "InputReporter",
    "OOMContext",
    "OOMDiagnostics",
    "annotate",
    "context",
    "is_oom",
    "report_oom",
    "tensor_report",
]
