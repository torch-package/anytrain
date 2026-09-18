from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .. import experiment
from ._naming import join_tag
from ._observation_state import _HistogramSnapshot, _ProfileSnapshot


def _merge_metrics(target: dict[str, Tensor], values: Mapping[str, Tensor]) -> None:
    collisions = sorted(set(target).intersection(values))
    if collisions:
        raise RuntimeError(
            "observation metrics produced duplicate logger tags: "
            + ", ".join(repr(tag) for tag in collisions)
        )
    target.update(values)


def _histogram_metrics(
    tag: str,
    scope: str,
    snapshot: _HistogramSnapshot,
) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    tiny = torch.finfo(snapshot.mass.dtype).tiny
    for group, size in enumerate(snapshot.schema.sizes):
        group_tag = join_tag(tag, snapshot.schema.group_names[group], scope)
        observations = snapshot.observations[group]
        total = snapshot.total[group]
        mass = snapshot.mass[group, :size]
        probabilities = torch.where(
            observations > 0,
            mass / observations.clamp_min(tiny),
            torch.zeros_like(mass),
        )
        safe = probabilities.clamp_min(tiny)
        entropy = -torch.where(
            probabilities > 0,
            probabilities * safe.log(),
            torch.zeros_like(probabilities),
        ).sum()
        values[join_tag(group_tag, "observations")] = observations
        values[join_tag(group_tag, "active_ratio")] = torch.where(
            total > 0,
            observations / total.clamp_min(tiny),
            torch.zeros_like(observations),
        )
        values[join_tag(group_tag, "entropy")] = entropy
        values[join_tag(group_tag, "perplexity")] = torch.where(
            observations > 0,
            entropy.exp(),
            torch.zeros_like(entropy),
        )
        if snapshot.schema.kind == "bins":
            denominator = mass.new_tensor(float(size))
            occupied = (mass > 0).sum().to(dtype=mass.dtype)
            dead = denominator - occupied
            values[join_tag(group_tag, "occupied_count")] = occupied
            values[join_tag(group_tag, "occupied_ratio")] = occupied / denominator
            values[join_tag(group_tag, "dead_count")] = dead
            values[join_tag(group_tag, "dead_ratio")] = dead / denominator
    return values


def _profile_metrics(
    tag: str,
    scope: str,
    snapshot: _ProfileSnapshot,
) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    scope_tag = join_tag(tag, scope)
    observations = snapshot.counts.sum()
    tiny = torch.finfo(snapshot.counts.dtype).tiny
    values[join_tag(scope_tag, "observations")] = observations
    values[join_tag(scope_tag, "active_ratio")] = torch.where(
        snapshot.total > 0,
        observations / snapshot.total.clamp_min(tiny),
        torch.zeros_like(observations),
    )
    for index in range(snapshot.schema.bins):
        bin_tag = join_tag(scope_tag, f"bin_{index}")
        count = snapshot.counts[index]
        mean = torch.where(
            count > 0,
            snapshot.sums[index] / count.clamp_min(tiny),
            torch.zeros_like(count),
        )
        variance = torch.where(
            count > 0,
            snapshot.sum_squares[index] / count.clamp_min(tiny) - mean.square(),
            torch.zeros_like(count),
        ).clamp_min(0)
        values[join_tag(bin_tag, "count")] = count
        values[join_tag(bin_tag, "mean")] = mean
        values[join_tag(bin_tag, "std")] = variance.sqrt()
    return values


def _write_heatmap(
    writer: experiment.Image,
    tag: str,
    image: Tensor,
    *,
    step: int,
) -> None:
    writer.add_image(
        tag,
        image,
        step,
        dataformats="HW",
    )


def _write_histogram(
    writer: experiment.HistogramRaw,
    tag: str,
    scope: str,
    snapshot: _HistogramSnapshot,
    *,
    step: int,
) -> None:
    for group, size in enumerate(snapshot.schema.sizes):
        observations = float(snapshot.observations[group].detach().cpu().item())
        if observations <= 0:
            continue
        group_tag = join_tag(tag, snapshot.schema.group_names[group], scope)
        counts = snapshot.mass[group, :size].detach().float().cpu().tolist()
        minimum, maximum = snapshot.schema.ranges[group]
        writer.add_histogram_raw(
            group_tag,
            min=minimum,
            max=maximum,
            num=observations,
            sum=float(snapshot.sums[group].detach().cpu().item()),
            sum_squares=float(snapshot.sum_squares[group].detach().cpu().item()),
            bucket_limits=list(snapshot.schema.bucket_limits[group]),
            bucket_counts=counts,
            global_step=step,
        )
