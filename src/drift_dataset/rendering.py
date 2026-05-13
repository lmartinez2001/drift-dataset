from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drift_dataset.types import FloatArray


def render_circle_sequence(
    coords: FloatArray,
    width: int,
    height: int,
    radius: float,
    background: float = 0.0,
    foreground: float = 1.0,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = np.empty((width, height, coords.shape[0]), dtype=np.float32)
    masks = np.empty((coords.shape[0], height, width), dtype=bool)

    for idx, (cx, cy) in enumerate(coords):
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        masks[idx] = mask
        frame = np.full((height, width), background, dtype=np.float32)
        frame[mask] = foreground
        frames[:, :, idx] = frame.T

    return frames, masks
