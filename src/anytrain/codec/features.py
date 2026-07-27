from __future__ import annotations

import torch
from torch import Tensor

from .protocol import SemanticAcousticCodec


@torch.no_grad()
def masked_acoustic_features(
    codec: SemanticAcousticCodec,
    acoustic_codes: Tensor,
    mask: Tensor,
    *,
    validate: bool = True,
) -> Tensor:
    """Convert masked acoustic code units to features without padding leakage."""
    if validate:
        if acoustic_codes.dim() != 3 or mask.shape != acoustic_codes.shape[:2]:
            raise ValueError(
                "acoustic_codes and mask must have shapes [B, unit, K] and [B, unit]."
            )
        if mask.dtype != torch.bool:
            raise TypeError("acoustic feature mask must be boolean.")
        if mask.device != acoustic_codes.device:
            raise ValueError("acoustic_codes and mask must use the same device.")

    safe_codes = acoustic_codes.masked_fill(~mask[..., None], 0)
    features = codec.acoustic_codes_to_features(safe_codes)
    if validate and (features.dim() != 3 or features.shape[:2] != mask.shape):
        raise ValueError("codec acoustic features must align with the acoustic code units.")
    feature_mask = mask.to(device=features.device)
    return features.masked_fill(~feature_mask[..., None], 0)


__all__ = ["masked_acoustic_features"]
