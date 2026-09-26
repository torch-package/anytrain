from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from lightning import pytorch as pl
from torch import Tensor, nn

from .. import _capability

if TYPE_CHECKING:
    from collections.abc import Sequence


class ParameterEMA:
    """Float32 exponential moving average of selected module parameters."""

    def __init__(
        self,
        module: nn.Module,
        *,
        decay: float,
        parameter_names: Iterable[str] | None = None,
        tracked_parameter_names: Iterable[str] | None = None,
    ) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise TypeError("decay must be a number.")
        if not 0 <= float(decay) < 1:
            raise ValueError(f"decay must be in [0, 1), got {decay}.")
        self.decay = float(decay)
        names = None if parameter_names is None else tuple(parameter_names)
        if names is not None:
            if not names:
                raise ValueError("parameter_names must not be empty when provided.")
            if len(set(names)) != len(names):
                raise ValueError("parameter_names must be unique.")
        self._names = names
        named = dict(module.named_parameters())
        tracked_names = (
            None
            if tracked_parameter_names is None
            else tuple(tracked_parameter_names)
        )
        if tracked_names is not None:
            if not tracked_names:
                raise ValueError("tracked_parameter_names must not be empty.")
            if len(set(tracked_names)) != len(tracked_names):
                raise ValueError("tracked_parameter_names must be unique.")
        elif names is None:
            tracked_names = tuple(
                name
                for name, parameter in named.items()
                if parameter.requires_grad and torch.is_floating_point(parameter)
            )
        else:
            tracked_names = names
        self._tracked_names = tracked_names
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in self._parameters(module)
        }
        if not self.shadow:
            raise ValueError("ParameterEMA requires at least one floating parameter.")
        self.num_updates = 0
        self._backup: dict[str, Tensor] | None = None

    def update(self, module: nn.Module) -> None:
        one_minus_decay = 1.0 - self.decay
        with torch.no_grad():
            for name, parameter in self._parameters(module):
                shadow = self.shadow[name]
                value = parameter.detach().float()
                if self.num_updates == 0:
                    shadow.copy_(value)
                else:
                    shadow.mul_(self.decay).add_(value, alpha=one_minus_decay)
        self.num_updates += 1

    def copy_to(self, module: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in self._parameters(module):
                parameter.copy_(self.shadow[name].to(dtype=parameter.dtype))

    def store(self, module: nn.Module) -> None:
        if self._backup is not None:
            raise RuntimeError("ParameterEMA.store() called while a backup is active.")
        self._backup = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters(module)
        }

    def restore(self, module: nn.Module) -> None:
        backup = self._backup
        if backup is None:
            raise RuntimeError("ParameterEMA.restore() requires an active backup.")
        with torch.no_grad():
            for name, parameter in self._parameters(module):
                parameter.copy_(backup[name])
        self._backup = None

    @contextmanager
    def average_parameters(self, module: nn.Module) -> Iterator[None]:
        self.store(module)
        try:
            self.copy_to(module)
            yield
        finally:
            self.restore(module)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "names": self._names,
            "tracked_names": self._tracked_names,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        decay = state.get("decay")
        if not isinstance(decay, (int, float)) or isinstance(decay, bool):
            raise TypeError("EMA state decay must be a number.")
        if float(decay) != self.decay:
            raise ValueError(f"EMA decay mismatch: expected {self.decay}, got {decay}.")
        names = state.get("names")
        if names != self._names:
            raise ValueError("EMA parameter name set mismatch.")
        tracked_names = state.get("tracked_names", tuple(state.get("shadow", {})))
        if tuple(tracked_names) != self._tracked_names:
            raise ValueError("EMA tracked parameter name set mismatch.")
        num_updates = state.get("num_updates")
        if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates < 0:
            raise TypeError("EMA num_updates must be a non-negative integer.")
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise TypeError("EMA shadow must be a mapping.")
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA shadow keys must match the tracked parameters.")
        loaded: dict[str, Tensor] = {}
        for name, value in shadow.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"EMA shadow {name!r} must be a Tensor.")
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA shadow {name!r} has incompatible shape.")
            loaded[name] = value
        with torch.no_grad():
            for name, value in loaded.items():
                self.shadow[name].copy_(value.detach())
        self.num_updates = num_updates
        self._backup = None

    def _parameters(self, module: nn.Module) -> list[tuple[str, nn.Parameter]]:
        lookup = dict(module.named_parameters())
        selected = []
        for name in self._tracked_names:
            parameter = lookup.get(name)
            if parameter is None:
                raise ValueError(f"EMA parameter {name!r} is missing from the module.")
            if not torch.is_floating_point(parameter):
                raise TypeError(f"EMA parameter {name!r} must be floating point.")
            selected.append((name, parameter))
        return selected


class EMACallback(pl.Callback):
    """Maintain parameter EMA and swap averaged weights for eval hooks."""

    def __init__(
        self,
        *,
        decay: float = 0.999,
        parameter_names: Sequence[str] | None = None,
        update_after_step: int = 0,
        use_ema_weights: bool = True,
        parameter_filter: Callable[[str, nn.Parameter], bool] | None = None,
    ) -> None:
        if isinstance(update_after_step, bool) or not isinstance(update_after_step, int):
            raise TypeError("update_after_step must be an integer.")
        if update_after_step < 0:
            raise ValueError("update_after_step must be non-negative.")
        if not isinstance(use_ema_weights, bool):
            raise TypeError("use_ema_weights must be a boolean.")
        if parameter_filter is not None and not callable(parameter_filter):
            raise TypeError("parameter_filter must be callable.")
        if parameter_names is not None and parameter_filter is not None:
            raise ValueError("parameter_names and parameter_filter are mutually exclusive.")
        self.decay = float(decay)
        self.parameter_names = None if parameter_names is None else tuple(parameter_names)
        self.update_after_step = update_after_step
        self.use_ema_weights = use_ema_weights
        self.parameter_filter = parameter_filter
        self.ema: ParameterEMA | None = None
        self._last_global_step: int | None = None
        self._active = False
        self._pending_ema: Mapping[str, Any] | None = None
        self._restored_last_global_step = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._ensure_initialized(trainer, pl_module)
        self._active = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        ema = self._require_ema()
        step = _capability.global_step(trainer)
        if self._last_global_step is None:
            raise RuntimeError("EMACallback requires on_fit_start before training updates.")
        if step < self._last_global_step:
            raise RuntimeError("trainer.global_step moved backwards during EMA updates.")
        while self._last_global_step < step:
            next_step = self._last_global_step + 1
            if next_step > self.update_after_step:
                ema.update(pl_module)
            self._last_global_step = next_step

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._ensure_initialized(trainer, pl_module)
        self._enter_ema(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        self._exit_ema(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._ensure_initialized(trainer, pl_module)
        self._enter_ema(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        self._exit_ema(pl_module)

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._ensure_initialized(trainer, pl_module)
        self._enter_ema(pl_module)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        self._exit_ema(pl_module)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, exception
        self._exit_ema(pl_module)

    @contextmanager
    def average_parameters(self, pl_module: pl.LightningModule) -> Iterator[None]:
        ema = self._require_ema()
        if not self.use_ema_weights:
            yield
            return
        with ema.average_parameters(pl_module):
            yield

    def state_dict(self) -> dict[str, Any]:
        ema = self.ema
        if ema is None:
            return {
                "decay": self.decay,
                "parameter_names": self.parameter_names,
                "update_after_step": self.update_after_step,
                "use_ema_weights": self.use_ema_weights,
                "ema": None,
                "last_global_step": self._last_global_step,
            }
        return {
            "decay": self.decay,
            "parameter_names": self.parameter_names,
            "update_after_step": self.update_after_step,
            "use_ema_weights": self.use_ema_weights,
            "ema": ema.state_dict(),
            "last_global_step": self._last_global_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        decay = state_dict.get("decay")
        if not isinstance(decay, (int, float)) or isinstance(decay, bool):
            raise TypeError("EMACallback state decay must be a number.")
        if float(decay) != self.decay:
            raise ValueError(f"EMACallback decay mismatch: expected {self.decay}, got {decay}.")
        names = state_dict.get("parameter_names")
        if names != self.parameter_names:
            raise ValueError("EMACallback parameter_names mismatch.")
        update_after_step = state_dict.get("update_after_step")
        if (
            isinstance(update_after_step, bool)
            or not isinstance(update_after_step, int)
            or update_after_step != self.update_after_step
        ):
            raise ValueError("EMACallback update_after_step mismatch.")
        use_ema_weights = state_dict.get("use_ema_weights")
        if use_ema_weights != self.use_ema_weights:
            raise ValueError("EMACallback use_ema_weights mismatch.")
        last_global_step = state_dict.get("last_global_step")
        if last_global_step is not None and (
            isinstance(last_global_step, bool) or not isinstance(last_global_step, int)
        ):
            raise TypeError("EMACallback last_global_step must be an integer or None.")
        self._last_global_step = last_global_step
        self._restored_last_global_step = True
        payload = state_dict.get("ema")
        if payload is None:
            self._pending_ema = None
            return
        if not isinstance(payload, Mapping):
            raise TypeError("EMACallback ema state must be a mapping.")
        ema = self.ema
        if ema is None:
            # Lightning restores callback state before on_fit_start creates EMA.
            self._pending_ema = payload
            return
        self._pending_ema = None
        ema.load_state_dict(payload)

    def _enter_ema(self, pl_module: pl.LightningModule) -> None:
        if not self.use_ema_weights or self._active:
            return
        ema = self._require_ema()
        ema.store(pl_module)
        self._active = True
        try:
            ema.copy_to(pl_module)
        except BaseException:
            self._exit_ema(pl_module)
            raise

    def _exit_ema(self, pl_module: pl.LightningModule) -> None:
        if not self._active:
            return
        self._require_ema().restore(pl_module)
        self._active = False

    def _require_ema(self) -> ParameterEMA:
        if self.ema is None:
            raise RuntimeError("EMACallback requires on_fit_start before use.")
        return self.ema

    def _ensure_initialized(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.ema is not None:
            return
        names = self.parameter_names
        if names is None and self.parameter_filter is not None:
            names = tuple(
                name
                for name, parameter in pl_module.named_parameters()
                if self.parameter_filter(name, parameter)
            )
        pending = self._pending_ema
        pending_tracked_names = None
        if pending is not None:
            pending_tracked_names = pending.get(
                "tracked_names", pending.get("shadow", {})
            )
            if not isinstance(pending_tracked_names, (tuple, list)):
                raise TypeError("pending EMA tracked parameter names must be a sequence.")
        self.ema = ParameterEMA(
            pl_module,
            decay=self.decay,
            parameter_names=names,
            tracked_parameter_names=pending_tracked_names,
        )
        if pending is not None:
            self.ema.load_state_dict(pending)
            self._pending_ema = None
        if not self._restored_last_global_step:
            self._last_global_step = _capability.global_step(trainer)
        self._restored_last_global_step = False


def find_ema_callback(trainer: pl.Trainer) -> EMACallback | None:
    callbacks = getattr(trainer, "callbacks", None)
    if callbacks is None:
        return None
    for callback in callbacks:
        if isinstance(callback, EMACallback):
            return callback
    return None


__all__ = ["EMACallback", "ParameterEMA", "find_ema_callback"]
