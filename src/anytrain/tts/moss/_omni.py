"""MOSS-TTS v1.5 prompt construction for vLLM-Omni."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import Tensor

from anytrain._compat import strict_zip
from anytrain.tts import AudioReference, TTSOptions

from ._runtime import materialized_audio_references, processor_message_kwargs
from ._types import Processor


def build_prompts(
    processor: Processor,
    texts: Sequence[str],
    options: TTSOptions,
    references: Sequence[AudioReference | None],
) -> list[dict[str, object]]:
    """Use the upstream processor to build text ids and reference codec codes."""

    prompts: list[dict[str, object]] = []
    with materialized_audio_references(references) as paths:
        for text, path in strict_zip(texts, paths):
            message = processor.build_user_message(
                **processor_message_kwargs(text, options, path)
            )
            batch = processor([[message]], mode="generation")
            if not isinstance(batch, Mapping):
                raise TypeError("MOSS-TTS processor output must be a mapping.")
            unified = batch.get("input_ids")
            if not isinstance(unified, Tensor):
                raise KeyError("MOSS-TTS processor output must include tensor `input_ids`.")
            unified = cast(Tensor, unified)
            if unified.ndim != 3 or unified.shape[0] != 1 or unified.shape[-1] < 2:
                raise ValueError(
                    "MOSS-TTS processor input_ids must have shape "
                    "[1, sequence, 1 + codebooks]."
                )
            item = unified[0]
            text_ids = item[:, 0].detach().to(device="cpu", dtype=torch.int64).tolist()
            audio_codes = (
                item[:, 1:].detach().to(device="cpu", dtype=torch.int64).contiguous()
            )
            prompts.append(
                {
                    "prompt_token_ids": text_ids,
                    "additional_information": {"codes": {"ref": audio_codes}},
                }
            )
    return prompts
