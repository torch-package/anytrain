"""Pure learning-rate interpolation shared by optimizer and Lightning APIs."""
from math import cos, pi


def cosine_interpolate(progress: float, *, start: float, end: float) -> float:
    resolved = min(max(progress, 0.0), 1.0)
    weight = 0.5 * (1.0 - cos(pi * resolved))
    return start + (end - start) * weight
