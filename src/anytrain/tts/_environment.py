"""Fail-fast distribution contracts for optional TTS runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class Requirement:
    distribution: str
    versions: str

    def __str__(self) -> str:
        return f"{self.distribution}{self.versions}"


@dataclass(frozen=True)
class EnvironmentContract:
    name: str
    requirements: tuple[Requirement, ...]
    install_hint: str
    incompatible: tuple[Requirement, ...] = ()
    incompatibility: str | None = None

    def validate(self) -> None:
        for requirement in self.incompatible:
            installed = _version(requirement.distribution)
            if installed is not None and _matches(installed, requirement.versions):
                detail = "" if self.incompatibility is None else f" {self.incompatibility}"
                raise RuntimeError(
                    f"{self.name} cannot use an environment containing "
                    f"{requirement.distribution} {installed}.{detail} {self.install_hint}"
                )

        for requirement in self.requirements:
            installed = _version(requirement.distribution)
            if installed is None:
                raise ImportError(
                    f"{self.name} requires {requirement}, but "
                    f"{requirement.distribution} is not installed. {self.install_hint}"
                )
            if not _matches(installed, requirement.versions):
                raise RuntimeError(
                    f"{self.name} requires {requirement}, but the current environment "
                    f"has {requirement.distribution} {installed}. {self.install_hint}"
                )


_SEPARATE_QWEN_VLLM_ENVIRONMENTS = (
    "Qwen3-TTS pins transformers==4.57.3 while vLLM 0.26 requires "
    "transformers>=5.5.3; use separate environments and process boundaries."
)
QWEN_TORCH_ENVIRONMENT = EnvironmentContract(
    name="Qwen3-TTS torch",
    requirements=(
        Requirement("transformers", "==4.57.3"),
        Requirement("qwen-tts", "==0.1.1"),
    ),
    incompatible=(
        Requirement("vllm", ">=0.26,<0.27"),
        Requirement("vllm-omni", ">=0.26,<0.27"),
    ),
    incompatibility=_SEPARATE_QWEN_VLLM_ENVIRONMENTS,
    install_hint="Install this backend with `python -m pip install 'anytrain[qwen-tts]'`.",
)
VLLM_OMNI_ENVIRONMENT = EnvironmentContract(
    name="vLLM-Omni TTS",
    requirements=(
        Requirement("transformers", ">=5.5.3"),
        Requirement("vllm", ">=0.26,<0.27"),
        Requirement("vllm-omni", ">=0.26,<0.27"),
    ),
    incompatible=(Requirement("qwen-tts", "==0.1.1"),),
    incompatibility=_SEPARATE_QWEN_VLLM_ENVIRONMENTS,
    install_hint=(
        "Install this backend in its own environment with "
        "`python -m pip install 'anytrain[tts-vllm-omni]'`."
    ),
)


def _version(distribution: str) -> str | None:
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


__all__ = ["QWEN_TORCH_ENVIRONMENT", "VLLM_OMNI_ENVIRONMENT"]
