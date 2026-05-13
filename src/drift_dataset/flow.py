from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drift_dataset.types import FloatArray


def compute_optical_flow(coords: FloatArray, masks: NDArray[np.bool_]) -> FloatArray:
    """Compute dense per-frame flow with non-zero vectors on the moving disk support."""
    frame_count, height, width = masks.shape
    flow = np.zeros((width, height, max(frame_count - 1, 0), 2), dtype=np.float32)

    for idx, delta in enumerate(np.diff(coords, axis=0)):
        support = masks[idx] | masks[idx + 1]
        flow[:, :, idx, 0] = np.where(support.T, delta[0], 0.0)
        flow[:, :, idx, 1] = np.where(support.T, delta[1], 0.0)

    return flow
