from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .layout import Layout


class Embedding(nn.Module):
    def __init__(
        self,
        layout: Layout,
        *,
        adapters: Mapping[str, nn.Module] | None = None,
        **embeddings: nn.Embedding,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.embeddings = _init_embeddings(layout, embeddings)
        self.adapters = _init_adapters(layout, adapters)
        self._homogeneous_output_dim = _homogeneous_output_dim(self.embeddings, self.adapters)

    @property
    def num_embeddings(self) -> int:
        return self.layout.vocab_size

    @property
    def vocab_size(self) -> int:
        return self.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty.")
        if self._homogeneous_output_dim is not None and _can_use_homogeneous_path(self.embeddings):
            return self._forward_homogeneous(input_ids, output_dim=self._homogeneous_output_dim)
        return self._forward_with_adapters(input_ids)

    def _forward_homogeneous(self, input_ids: torch.Tensor, *, output_dim: int) -> torch.Tensor:
        first = next(iter(self.embeddings.values()))
        output = first.weight.new_empty((*input_ids.shape, output_dim))
        covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for name, embed in self.embeddings.items():
            start, end = self.layout.blocks[name]
            mask = (input_ids >= start) & (input_ids < end)
            output[mask] = embed(input_ids[mask] - start)
            covered |= mask

        _validate_covered(input_ids, covered)
        return output

    def _forward_with_adapters(self, input_ids: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor | None = None
        covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for name, embed in self.embeddings.items():
            start, end = self.layout.blocks[name]
            mask = (input_ids >= start) & (input_ids < end)
            if not bool(mask.any()):
                continue

            local_ids = input_ids[mask] - start
            local_output = embed(local_ids)
            if name in self.adapters:
                local_output = self.adapters[name](local_output)
            if not isinstance(local_output, torch.Tensor):
                raise TypeError(f"embedding block {name!r} must return a tensor.")

            if output is None:
                output = local_output.new_empty((*input_ids.shape, local_output.shape[-1]))
            elif local_output.shape[-1] != output.shape[-1]:
                raise ValueError("all selected embedding blocks must produce the same output dim.")
            output[mask] = local_output
            covered |= mask

        if not bool(covered.all()):
            bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"input_ids contains id outside space: {bad_id}.")
        if output is None:
            raise RuntimeError("embedding routing produced no output.")
        return output


def _init_embeddings(
    layout: Layout,
    embeddings: Mapping[str, nn.Embedding],
) -> nn.ModuleDict:
    expected = set(layout.block_names)
    actual = set(embeddings)
    missing = expected - actual
    if missing:
        raise ValueError(f"missing embeddings for id blocks: {sorted(missing)!r}.")
    unknown = actual - expected
    if unknown:
        raise ValueError(f"unknown embedding blocks: {sorted(unknown)!r}.")

    modules = nn.ModuleDict()
    for name in layout.block_names:
        embed = embeddings[name]
        _validate_weight_size(name, embed, layout.blocks[name])
        modules[name] = embed
    return modules


def _init_adapters(
    layout: Layout,
    adapters: Mapping[str, nn.Module] | None,
) -> nn.ModuleDict:
    if adapters is None:
        return nn.ModuleDict()
    unknown = set(adapters) - set(layout.block_names)
    if unknown:
        raise ValueError(f"unknown adapter blocks: {sorted(unknown)!r}.")

    modules = nn.ModuleDict()
    for name, adapter in adapters.items():
        if not isinstance(adapter, nn.Module):
            raise TypeError(f"adapters[{name!r}] must be an nn.Module.")
        modules[name] = adapter
    return modules


def _validate_weight_size(name: str, embed: nn.Embedding, block: tuple[int, int]) -> None:
    start, end = block
    expected = end - start
    if embed.num_embeddings != expected:
        raise ValueError(
            f"embeddings[{name!r}].num_embeddings must match id block size {expected}."
        )


def _homogeneous_output_dim(
    embeddings: nn.ModuleDict,
    adapters: nn.ModuleDict,
) -> int | None:
    if len(adapters) > 0:
        return None
    output_dims = {embed.embedding_dim for embed in embeddings.values()}
    if len(output_dims) != 1:
        return None
    return output_dims.pop()


def _validate_covered(input_ids: torch.Tensor, covered: torch.Tensor) -> None:
    if not bool(covered.all()):
        bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
        raise ValueError(f"input_ids contains id outside space: {bad_id}.")


def _can_use_homogeneous_path(embeddings: nn.ModuleDict) -> bool:
    weights = [embed.weight for embed in embeddings.values()]
    first = weights[0]
    if any(weight.device != first.device or weight.dtype != first.dtype for weight in weights[1:]):
        return False
    return not torch.is_grad_enabled() or not any(weight.requires_grad for weight in weights)


__all__ = [
    "Embedding",
]
