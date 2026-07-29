from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from lightning import pytorch as pl

_Kind = Literal["loss", "parameter", "gradient"]
_NamedTensor = tuple[str, _Kind, torch.Tensor]


@dataclass(frozen=True)
class _NonfiniteIssue:
    kind: _Kind
    name: str
    index: tuple[int, ...]
    value: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class DebugCallback(pl.Callback):
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        values: list[_NamedTensor] = [
            (name, "parameter", parameter) for name, parameter in pl_module.named_parameters()
        ]
        issue = _find_first_nonfinite(values)
        if issue is not None:
            _raise(trainer, issue, where="at train start")

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        named_parameters = list(pl_module.named_parameters())
        values: list[_NamedTensor] = [
            (name, "parameter", parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        ]
        for name, parameter in named_parameters:
            if parameter.grad is not None:
                values.append((name, "gradient", parameter.grad))
        issue = _find_first_nonfinite(values)
        if issue is not None:
            _raise(trainer, issue, where="after backward")

    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ) -> None:
        del pl_module
        issue = _find_first_nonfinite([("loss", "loss", loss)])
        if issue is not None:
            _raise(trainer, issue, where="before backward")


def _find_first_nonfinite(
    named_tensors: Iterable[_NamedTensor],
) -> _NonfiniteIssue | None:
    values = list(named_tensors)
    if _all_finite(tensor for _, _, tensor in values):
        return None
    for name, kind, tensor in values:
        issue = _find_nonfinite_tensor(name, kind, tensor)
        if issue is not None:
            return issue
    return None


def _all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = defaultdict(list)
    reductions: dict[torch.device, list[torch.Tensor]] = defaultdict(list)
    for tensor in tensors:
        data = tensor.detach()
        if not (data.is_floating_point() or data.is_complex()):
            continue
        if data.layout is torch.strided:
            groups[(data.device, data.dtype)].append(data)
        else:
            reductions[data.device].append(torch.isfinite(data.to_dense()).all())

    for (device, _), values in groups.items():
        try:
            norms = torch._foreach_norm(values, float("inf"))
        except (RuntimeError, TypeError):
            reductions[device].extend(torch.isfinite(value).all() for value in values)
        else:
            reductions[device].append(torch.stack(norms).isfinite().all())

    return all(bool(torch.stack(values).all()) for values in reductions.values())


def _find_nonfinite_tensor(
    name: str,
    kind: _Kind,
    tensor: torch.Tensor,
) -> _NonfiniteIssue | None:
    data = tensor.detach()
    if data.layout is not torch.strided:
        data = data.to_dense()
    mask = ~torch.isfinite(data)
    if not bool(mask.any().item()):
        return None

    index = tuple(int(item) for item in mask.nonzero(as_tuple=False)[0].tolist())
    value = data[index].item()
    return _NonfiniteIssue(
        kind=kind,
        name=name,
        index=index,
        value=repr(value),
        shape=tuple(data.shape),
        dtype=data.dtype,
        device=data.device,
    )


def _raise(
    trainer: pl.Trainer,
    issue: _NonfiniteIssue,
    *,
    where: str,
) -> None:
    message = _format_issue(trainer, issue, where=where)
    print(message, file=sys.stderr, flush=True)
    raise RuntimeError(message)


def _format_issue(
    trainer: pl.Trainer,
    issue: _NonfiniteIssue,
    *,
    where: str,
) -> str:
    return (
        f"Non-finite {issue.kind} detected {where} "
        f"(epoch={trainer.current_epoch}, global_step={trainer.global_step}, "
        f"rank={trainer.global_rank}). "
        f"name={issue.name!r}, index={issue.index}, value={issue.value}, "
        f"shape={issue.shape}, dtype={issue.dtype}, device={issue.device}."
    )


__all__ = [
    "DebugCallback",
]
