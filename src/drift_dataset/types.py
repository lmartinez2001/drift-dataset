from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
CurveFn = Callable[[NDArray[np.float32]], FloatArray]


@dataclass(frozen=True)
class PathSpec:
    name: str
    params: dict[str, float]
    curve: CurveFn
    closed: bool = False


@dataclass(frozen=True)
class Sample:
    frames: FloatArray
    optical_flow: FloatArray
    coords: FloatArray
    metadata: dict[str, object]
