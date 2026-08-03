from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TypeVar

import torch.nn as nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)
ValueT = TypeVar("ValueT")


def module_monitor_names(
    pl_module: nn.Module,
    module_type: type[ModuleT] | tuple[type[ModuleT], ...],
) -> dict[int, str]:
    """Return compact unique monitor names for matching submodules.

    Names are inferred from pl_module.named_modules() and shortened to the
    shortest unique dot-component suffix among modules of module_type. For
    example, a single encoder.router.gate monitor is named gate; paired
    encoder.router.gate and decoder.router.gate monitors keep their full paths
    because neither gate nor router.gate is unique.

    If the same module object appears more than once, the first path wins.
    """

    return _module_monitor_names(
        pl_module,
        lambda module: isinstance(module, module_type),
    )


def module_monitor_names_by_predicate(
    pl_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
) -> dict[int, str]:
    """Return compact unique monitor names for modules matching predicate."""

    return _module_monitor_names(pl_module, predicate)


def join_tag(*parts: object) -> str:
    """Join logger tag parts with / while preserving dot-style names."""

    cleaned = [str(part).strip("/") for part in parts if str(part).strip("/")]
    return "/".join(cleaned)


def validate_tag_part(value: object, name: str = "tag") -> str:
    """Normalize and validate one logger tag component."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    text = value.strip("/")
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def unique_tag_items(
    items: Iterable[tuple[object, ValueT]],
    *,
    name: str = "monitor name",
) -> list[tuple[str, ValueT]]:
    """Normalize named values and reject collisions introduced by normalization."""

    result: list[tuple[str, ValueT]] = []
    source_by_name: dict[str, object] = {}
    for source, value in items:
        normalized = validate_tag_part(source, name)
        if normalized in source_by_name:
            previous = source_by_name[normalized]
            raise ValueError(
                f"{name} values {previous!r} and {source!r} normalize to "
                f"the same tag {normalized!r}."
            )
        source_by_name[normalized] = source
        result.append((normalized, value))
    return result


def _module_monitor_names(
    pl_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
) -> dict[int, str]:
    paths = _matching_module_paths(pl_module, predicate)
    path_by_id = {module_id: path for module_id, path in paths}
    suffix_by_path = _shortest_unique_suffixes(path_by_id.values())
    return {
        module_id: suffix_by_path[path]
        for module_id, path in path_by_id.items()
    }


def _matching_module_paths(
    pl_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
) -> list[tuple[int, str]]:
    seen: set[int] = set()
    result: list[tuple[int, str]] = []
    for name, module in pl_module.named_modules():
        if not predicate(module):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        result.append((module_id, _module_path(name)))
    return result


def _shortest_unique_suffixes(paths: Iterable[str]) -> Mapping[str, str]:
    path_list = list(paths)
    suffixes: dict[str, str] = {}
    for path in path_list:
        components = _components(path)
        for length in range(1, len(components) + 1):
            suffix = ".".join(components[-length:])
            if _suffix_is_unique(path, suffix, path_list):
                suffixes[path] = suffix
                break
        else:
            suffixes[path] = path
    return suffixes


def _suffix_is_unique(path: str, suffix: str, paths: list[str]) -> bool:
    return sum(
        candidate == suffix or candidate.endswith(f".{suffix}")
        for candidate in paths
    ) == 1


def _components(path: str) -> list[str]:
    return [component for component in path.split(".") if component] or ["module"]


def _module_path(name: str) -> str:
    text = name.strip(".")
    return text or "module"


__all__ = [
    "join_tag",
    "module_monitor_names",
    "module_monitor_names_by_predicate",
    "unique_tag_items",
    "validate_tag_part",
]
