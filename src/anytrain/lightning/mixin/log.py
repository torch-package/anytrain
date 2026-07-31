from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import torch
from lightning import pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from .. import _capability

RankLogMode = Literal["zero", "all"]


def prefixed_log_dict(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    normalized_prefix = prefix.strip("/")
    if not normalized_prefix:
        raise ValueError("prefix must not be empty.")

    prefixed: dict[str, Any] = {}
    for key, value in values.items():
        normalized_key = key.strip("/")
        if not normalized_key:
            raise ValueError("metric keys must not be empty.")
        prefixed[f"{normalized_prefix}/{normalized_key}"] = value
    return prefixed


class LightningLogMixin:
    def log_prefixed_dict(
        self,
        prefix: str,
        values: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        module = _as_lightning_log_module(self)
        module.log_dict(prefixed_log_dict(prefix, values), **kwargs)

    def log_audio(
        self,
        tag: str,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        step: int | None = None,
        rank_mode: RankLogMode = "zero",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        module = _as_lightning_log_module(self)
        trainer = module.trainer
        _require_tag(tag)
        normalized_rank_mode = _validate_rank_mode(rank_mode)
        if normalized_rank_mode == "zero" and not _capability.is_global_zero(trainer):
            return

        _log_audio_to_loggers(
            _capability.loggers(trainer),
            _ranked_tag(tag, trainer=trainer, rank_mode=normalized_rank_mode),
            audio,
            step=module.global_step if step is None else step,
            sample_rate=sample_rate,
        )

    def log_figure(
        self,
        tag: str,
        figure: Any,
        *,
        step: int | None = None,
        rank_mode: RankLogMode = "zero",
    ) -> None:
        module = _as_lightning_log_module(self)
        trainer = module.trainer
        _require_tag(tag)
        normalized_rank_mode = _validate_rank_mode(rank_mode)
        if normalized_rank_mode == "zero" and not _capability.is_global_zero(trainer):
            return

        _log_figure_to_loggers(
            _capability.loggers(trainer),
            _ranked_tag(tag, trainer=trainer, rank_mode=normalized_rank_mode),
            figure,
            step=module.global_step if step is None else step,
        )


def _as_lightning_log_module(module: object) -> pl.LightningModule:
    return cast(pl.LightningModule, module)


def _log_audio_to_loggers(
    loggers: Sequence[Any],
    tag: str,
    audio: torch.Tensor,
    *,
    step: int,
    sample_rate: int,
) -> None:
    _require_tag(tag)
    if not loggers:
        raise RuntimeError("Cannot log audio because Trainer has no configured logger.")

    logged = False
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_audio(tag, audio, global_step=step, sample_rate=sample_rate)
            logged = True

    if not logged:
        _raise_unsupported_logger("audio", loggers)


def _log_figure_to_loggers(
    loggers: Sequence[Any],
    tag: str,
    figure: Any,
    *,
    step: int,
) -> None:
    _require_tag(tag)
    if not loggers:
        raise RuntimeError("Cannot log figure because Trainer has no configured logger.")

    logged = False
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_figure(tag, figure, global_step=step)
            logged = True

    if not logged:
        _raise_unsupported_logger("figure", loggers)


def _validate_rank_mode(rank_mode: str) -> RankLogMode:
    if rank_mode not in ("zero", "all"):
        raise ValueError("rank_mode must be 'zero' or 'all'.")
    return cast(RankLogMode, rank_mode)


def _ranked_tag(tag: str, *, trainer: pl.Trainer, rank_mode: RankLogMode) -> str:
    if rank_mode == "zero" or not _capability.is_distributed(trainer):
        return tag
    return f"rank={_capability.global_rank(trainer)}/{tag}"


def _require_tag(tag: str) -> None:
    if not tag.strip():
        raise ValueError("tag must not be empty.")


def _raise_unsupported_logger(feature: str, loggers: Sequence[Any]) -> None:
    logger_names = ", ".join(type(logger).__name__ for logger in loggers)
    raise RuntimeError(
        f"No configured Lightning logger supports {feature} logging. "
        f"Supported backends: TensorBoardLogger. Configured loggers: {logger_names}."
    )
