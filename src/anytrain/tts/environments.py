"""Inspectable distribution contracts for optional TTS runtimes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

EnvironmentId: TypeAlias = Literal[
    "qwen_torch",
    "moss_torch",
    "cosyvoice_vllm",
    "vllm_omni",
]


@dataclass(frozen=True)
class Requirement:
    """One installed-distribution requirement."""

    distribution: str
    versions: str

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, str) or not self.distribution:
            raise ValueError("requirement distribution must be a non-empty string.")
        if not isinstance(self.versions, str) or not self.versions:
            raise ValueError("requirement versions must be a non-empty string.")

    def __str__(self) -> str:
        return f"{self.distribution}{self.versions}"


@dataclass(frozen=True)
class Conflict:
    """An installed distribution that makes a runtime contract invalid."""

    requirement: Requirement
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("conflict reason must be a non-empty string.")


@dataclass(frozen=True)
class EnvironmentContract:
    """A reusable, fail-fast contract for one TTS runtime family."""

    id: EnvironmentId
    name: str
    requirements: tuple[Requirement, ...]
    install_hint: str
    conflicts: tuple[Conflict, ...] = ()

    @property
    def incompatible(self) -> tuple[Requirement, ...]:
        """Return conflicting requirements for callers migrating from 0.x."""

        return tuple(conflict.requirement for conflict in self.conflicts)

    @property
    def incompatibility(self) -> str | None:
        """Return a compact conflict explanation for legacy inspection code."""

        reasons = tuple(dict.fromkeys(conflict.reason for conflict in self.conflicts))
        return " ".join(reasons) if reasons else None

    def validate(self, installed: Mapping[str, str] | None = None) -> None:
        """Validate installed distributions or an explicit version inventory."""

        versions = None if installed is None else _inventory(installed)
        for conflict in self.conflicts:
            requirement = conflict.requirement
            found = _version(requirement.distribution, versions)
            if found is not None and _matches(found, requirement.versions):
                raise RuntimeError(
                    f"{self.name} cannot use an environment containing "
                    f"{requirement.distribution} {found}. {conflict.reason} "
                    f"{self.install_hint}"
                )

        for requirement in self.requirements:
            found = _version(requirement.distribution, versions)
            if found is None:
                raise ImportError(
                    f"{self.name} requires {requirement}, but "
                    f"{requirement.distribution} is not installed. {self.install_hint}"
                )
            if not _matches(found, requirement.versions):
                raise RuntimeError(
                    f"{self.name} requires {requirement}, but the current environment "
                    f"has {requirement.distribution} {found}. {self.install_hint}"
                )


_SEPARATE_QWEN_VLLM_ENVIRONMENTS = (
    "Qwen3-TTS pins transformers==4.57.3 while vLLM 0.26 requires "
    "transformers>=5.5.3; use separate environments and process boundaries."
)
_SEPARATE_MOSS_ENVIRONMENT = (
    "MOSS-TTS torch uses Transformers 5.0 and a Torch 2.9/TorchCodec 0.8 runtime; "
    "keep it separate from Qwen3-TTS torch and vLLM-Omni."
)
_SEPARATE_COSYVOICE_ENVIRONMENT = (
    "CosyVoice embedded vLLM uses the validated vLLM 0.11 and Transformers 4.57.1 "
    "runtime; keep it separate from Qwen3-TTS torch and vLLM-Omni."
)

QWEN_TORCH_ENVIRONMENT = EnvironmentContract(
    id="qwen_torch",
    name="Qwen3-TTS torch",
    requirements=(
        Requirement("transformers", "==4.57.3"),
        Requirement("qwen-tts", "==0.1.1"),
    ),
    conflicts=(
        Conflict(Requirement("vllm", ">=0.26,<0.27"), _SEPARATE_QWEN_VLLM_ENVIRONMENTS),
        Conflict(
            Requirement("vllm-omni", ">=0.26,<0.27"),
            _SEPARATE_QWEN_VLLM_ENVIRONMENTS,
        ),
    ),
    install_hint="Install this backend with `python -m pip install 'anytrain[qwen-tts]'`.",
)
MOSS_TORCH_ENVIRONMENT = EnvironmentContract(
    id="moss_torch",
    name="MOSS-TTS v1.5 torch",
    requirements=(
        Requirement("torch", ">=2.9,<2.10"),
        Requirement("torchaudio", ">=2.9,<2.10"),
        Requirement("torchcodec", ">=0.8,<0.9"),
        Requirement("transformers", "==5.0.0"),
    ),
    conflicts=(
        Conflict(Requirement("qwen-tts", "==0.1.1"), _SEPARATE_MOSS_ENVIRONMENT),
        Conflict(Requirement("vllm", ">=0.26,<0.27"), _SEPARATE_MOSS_ENVIRONMENT),
        Conflict(
            Requirement("vllm-omni", ">=0.26,<0.27"),
            _SEPARATE_MOSS_ENVIRONMENT,
        ),
    ),
    install_hint="Install this backend with `python -m pip install 'anytrain[moss-tts]'`.",
)
COSYVOICE_VLLM_ENVIRONMENT = EnvironmentContract(
    id="cosyvoice_vllm",
    name="CosyVoice embedded vLLM",
    requirements=(
        Requirement("torch", "==2.8.0"),
        Requirement("torchaudio", "==2.8.0"),
        Requirement("transformers", "==4.57.1"),
        Requirement("vllm", "==0.11.0"),
        Requirement("onnxruntime-gpu", "==1.20.2"),
    ),
    conflicts=(
        Conflict(Requirement("qwen-tts", "==0.1.1"), _SEPARATE_COSYVOICE_ENVIRONMENT),
        Conflict(
            Requirement("vllm-omni", ">=0.26,<0.27"),
            _SEPARATE_COSYVOICE_ENVIRONMENT,
        ),
    ),
    install_hint=(
        "Install the engine runtime with `python -m pip install "
        "'anytrain[cosyvoice-vllm]'`, then install the pinned CosyVoice source checkout "
        "and its Matcha-TTS submodule."
    ),
)
VLLM_OMNI_ENVIRONMENT = EnvironmentContract(
    id="vllm_omni",
    name="vLLM-Omni TTS",
    requirements=(
        Requirement("transformers", ">=5.5.3"),
        Requirement("vllm", ">=0.26,<0.27"),
        Requirement("vllm-omni", ">=0.26,<0.27"),
    ),
    conflicts=(
        Conflict(Requirement("qwen-tts", "==0.1.1"), _SEPARATE_QWEN_VLLM_ENVIRONMENTS),
    ),
    install_hint=(
        "Install this backend in its own environment with "
        "`python -m pip install 'anytrain[tts-vllm-omni]'`."
    ),
)

ENVIRONMENT_CONTRACTS: Mapping[EnvironmentId, EnvironmentContract] = MappingProxyType(
    {
        contract.id: contract
        for contract in (
            QWEN_TORCH_ENVIRONMENT,
            MOSS_TORCH_ENVIRONMENT,
            COSYVOICE_VLLM_ENVIRONMENT,
            VLLM_OMNI_ENVIRONMENT,
        )
    }
)


def environment_contract(environment: str) -> EnvironmentContract:
    """Return one named TTS environment contract."""

    if not isinstance(environment, str):
        raise TypeError("environment must be a string.")
    if environment not in ENVIRONMENT_CONTRACTS:
        choices = ", ".join(ENVIRONMENT_CONTRACTS)
        raise KeyError(f"unknown TTS environment {environment!r}; expected one of: {choices}.") from None
    return ENVIRONMENT_CONTRACTS[cast(EnvironmentId, environment)]


def _inventory(installed: Mapping[str, str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, found in installed.items():
        if not isinstance(distribution, str) or not distribution:
            raise ValueError("installed distribution names must be non-empty strings.")
        if not isinstance(found, str) or not found:
            raise ValueError("installed distribution versions must be non-empty strings.")
        canonical = _canonical(distribution)
        if canonical in versions:
            raise ValueError(
                f"installed distribution inventory contains duplicate normalized name "
                f"{canonical!r}."
            )
        versions[canonical] = found
    return versions


def _canonical(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _version(distribution: str, installed: Mapping[str, str] | None) -> str | None:
    if installed is not None:
        return installed.get(_canonical(distribution))
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _matches(installed: str, versions: str) -> bool:
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:
        raise ImportError(
            "TTS environment validation requires `packaging`; reinstall the selected "
            "AnyTrain TTS extra."
        ) from exc

    try:
        return Version(installed) in SpecifierSet(versions)
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise RuntimeError(
            f"invalid TTS environment version contract {versions!r} or installed "
            f"version {installed!r}."
        ) from exc


__all__ = [
    "COSYVOICE_VLLM_ENVIRONMENT",
    "ENVIRONMENT_CONTRACTS",
    "MOSS_TORCH_ENVIRONMENT",
    "QWEN_TORCH_ENVIRONMENT",
    "VLLM_OMNI_ENVIRONMENT",
    "Conflict",
    "EnvironmentContract",
    "EnvironmentId",
    "Requirement",
    "environment_contract",
]
