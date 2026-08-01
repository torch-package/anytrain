from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal, TypedDict
from urllib.request import urlopen

from .._cache import anytrain_cache_dir

ModelType = Literal["16khz", "24khz", "44khz"]
ModelBitrate = Literal["8kbps", "16kbps"]

DEFAULT_MODEL_TYPE: ModelType = "44khz"
DEFAULT_MODEL_BITRATE: ModelBitrate = "8kbps"

LATEST_TAGS: dict[tuple[ModelType, ModelBitrate], str] = {
    ("16khz", "8kbps"): "0.0.5",
    ("24khz", "8kbps"): "0.0.4",
    ("44khz", "8kbps"): "0.0.1",
    ("44khz", "16kbps"): "1.0.0",
}

MODEL_URLS: dict[tuple[ModelType, ModelBitrate, str], str] = {
    (
        "16khz",
        "8kbps",
        "0.0.5",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "24khz",
        "8kbps",
        "0.0.4",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "44khz",
        "8kbps",
        "0.0.1",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "44khz",
        "16kbps",
        "1.0.0",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}
MODEL_SHA256S: dict[tuple[ModelType, ModelBitrate, str], str] = {}
DEFAULT_DOWNLOAD_TIMEOUT = 30.0


class DACAssets(TypedDict):
    cache_dir: Path
    checkpoint: Path
    model_type: ModelType
    model_bitrate: ModelBitrate
    tag: str


def ensure_dac_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    model_type: ModelType = DEFAULT_MODEL_TYPE,
    model_bitrate: ModelBitrate = DEFAULT_MODEL_BITRATE,
    tag: str = "latest",
    local_files_only: bool = False,
    force_download: bool = False,
    allow_unsafe_download: bool = False,
) -> DACAssets:
    resolved_tag, url = _model(model_type, model_bitrate, tag)
    root = anytrain_cache_dir("dac", cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / f"weights_{model_type}_{model_bitrate}_{resolved_tag}.pth"

    if force_download or not checkpoint.exists():
        if local_files_only:
            raise FileNotFoundError(f"DAC checkpoint is not available locally: {checkpoint}.")
        key = (model_type, model_bitrate, resolved_tag)
        expected_sha256 = MODEL_SHA256S.get(key)
        if expected_sha256 is None and not allow_unsafe_download:
            raise ValueError(
                "DAC remote checkpoint downloads require SHA256 metadata. "
                "Pass allow_unsafe_download=True for an explicitly unsafe download, "
                "or pre-populate the checkpoint and use local_files_only=True."
            )
        _download(url, checkpoint, expected_sha256=expected_sha256)

    return {
        "cache_dir": root,
        "checkpoint": checkpoint,
        "model_type": model_type,
        "model_bitrate": model_bitrate,
        "tag": resolved_tag,
    }


def _model(
    model_type: ModelType,
    model_bitrate: ModelBitrate,
    tag: str,
) -> tuple[str, str]:
    if tag == "latest":
        try:
            tag = LATEST_TAGS[(model_type, model_bitrate)]
        except KeyError as exc:
            raise ValueError(
                f"DAC has no latest checkpoint for {model_type!r} at {model_bitrate!r}."
            ) from exc

    try:
        return tag, MODEL_URLS[(model_type, model_bitrate, tag)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown DAC checkpoint: model_type={model_type!r}, "
            f"model_bitrate={model_bitrate!r}, tag={tag!r}."
        ) from exc


def _download(
    url: str,
    target: Path,
    *,
    expected_sha256: str | None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            with urlopen(url, timeout=timeout) as response:
                shutil.copyfileobj(response, output)
            if expected_sha256 is not None:
                output.flush()
                actual_sha256 = _sha256_file(temporary)
                if actual_sha256.lower() != expected_sha256.lower():
                    raise ValueError(
                        "DAC checkpoint SHA256 mismatch: "
                        f"expected {expected_sha256}, got {actual_sha256}."
                    )
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DACAssets",
    "DEFAULT_MODEL_BITRATE",
    "DEFAULT_MODEL_TYPE",
    "LATEST_TAGS",
    "MODEL_SHA256S",
    "MODEL_URLS",
    "ModelBitrate",
    "ModelType",
    "ensure_dac_assets",
]
