from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import cast

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
        if not isinstance(layout, Layout):
            raise TypeError("layout must be a Layout.")
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

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        selected_blocks: Collection[str] | None = None,
        validate: bool = True,
    ) -> torch.Tensor:
        _input_ids(input_ids)
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool.")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty.")
        selected = _checked_selected_blocks(selected_blocks, self.layout)
        if not validate and selected is None:
            raise ValueError("selected_blocks is required when validate=False.")
        if (
            selected is None
            and self._homogeneous_output_dim is not None
            and _can_use_homogeneous_path(self.embeddings)
        ):
            return self._forward_homogeneous(input_ids, output_dim=self._homogeneous_output_dim)
        return self._forward_with_adapters(
            input_ids,
            selected_blocks=selected,
            validate=validate,
        )

    def selected_blocks(self, input_ids: torch.Tensor) -> frozenset[str]:
        """Return the exact routed blocks after validating global ids.

        Call this while ``input_ids`` are still on CPU when a trusted caller needs
        to carry routing metadata across an asynchronous device transfer.
        """
        _input_ids(input_ids)
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty.")
        return frozenset(_selected_blocks(input_ids, self.layout))

    def _forward_homogeneous(self, input_ids: torch.Tensor, *, output_dim: int) -> torch.Tensor:
        first = cast(nn.Embedding, next(iter(self.embeddings.values())))
        output = first.weight.new_empty((*input_ids.shape, output_dim))
        covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for name, embed in self.embeddings.items():
            start, end = self.layout.blocks[name]
            mask = (input_ids >= start) & (input_ids < end)
            output[mask] = embed(input_ids[mask] - start)
            covered |= mask

        _validate_covered(input_ids, covered)
        return output

    def _forward_with_adapters(
        self,
        input_ids: torch.Tensor,
        *,
        selected_blocks: frozenset[str] | None,
        validate: bool,
    ) -> torch.Tensor:
        output: torch.Tensor | None = None
        selected = _selected_blocks(input_ids, self.layout) if validate else selected_blocks
        if selected is None:
            raise ValueError("selected_blocks is required when validate=False.")
        assert selected is not None

        for name, embed in self.embeddings.items():
            if name not in selected:
                continue
            start, end = self.layout.blocks[name]
            mask = (input_ids >= start) & (input_ids < end)
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
            assert output is not None
            output[mask] = local_output

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
        if not isinstance(embed, nn.Embedding):
            raise TypeError(f"embeddings[{name!r}] must be an nn.Embedding.")
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
    output_dims = {cast(int, embed.embedding_dim) for embed in embeddings.values()}
    if len(output_dims) != 1:
        return None
    return output_dims.pop()


def _validate_covered(input_ids: torch.Tensor, covered: torch.Tensor) -> None:
    if not bool(covered.all()):
        bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
        raise ValueError(f"input_ids contains id outside space: {bad_id}.")


def _selected_blocks(input_ids: torch.Tensor, layout: Layout) -> set[str]:
    covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    hits: list[torch.Tensor] = []
    for start, end in layout.blocks.values():
        mask = (input_ids >= start) & (input_ids < end)
        hits.append(mask.any())
        covered |= mask

    # Read all routing predicates together so unselected modules stay out of autograd
    # without introducing one device synchronization per block.
    first_uncovered = (~covered).reshape(-1).to(dtype=torch.int64).argmax()
    first_uncovered_id = input_ids.reshape(-1).gather(0, first_uncovered.reshape(1))
    summary = torch.cat(
        (
            torch.stack((*hits, covered.all())).to(dtype=torch.int64),
            first_uncovered_id.to(dtype=torch.int64),
        )
    ).detach().cpu().tolist()
    *selected, all_covered, bad_id = summary
    if not all_covered:
        raise ValueError(f"input_ids contains id outside space: {bad_id}.")
    return {
        name
        for name, hit in zip(layout.block_names, selected, strict=False)
        if hit
    }


def _checked_selected_blocks(
    selected_blocks: Collection[str] | None,
    layout: Layout,
) -> frozenset[str] | None:
    if selected_blocks is None:
        return None
    if isinstance(selected_blocks, (str, bytes)) or not isinstance(
        selected_blocks,
        Collection,
    ):
        raise TypeError("selected_blocks must be a collection of block names.")
    if any(not isinstance(name, str) for name in selected_blocks):
        raise TypeError("selected_blocks must contain only strings.")
    selected = frozenset(selected_blocks)
    unknown = selected - set(layout.block_names)
    if unknown:
        raise ValueError(f"selected_blocks contains unknown blocks: {sorted(unknown)!r}.")
    if not selected:
        raise ValueError("selected_blocks must not be empty.")
    return selected


def _can_use_homogeneous_path(embeddings: nn.ModuleDict) -> bool:
    weights = [embed.weight for embed in embeddings.values()]
    first = weights[0]
    if any(weight.device != first.device or weight.dtype != first.dtype for weight in weights[1:]):
        return False
    return not torch.is_grad_enabled() or not any(weight.requires_grad for weight in weights)


def _input_ids(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor.")


__all__ = [
    "Embedding",
]
