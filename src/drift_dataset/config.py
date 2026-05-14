from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FLUX_2_KLEIN_DEFAULT_RESOLUTION = 1024


@dataclass(frozen=True)
class DatasetConfig:
    sample_count: int
    frame_count: int
    width: int = FLUX_2_KLEIN_DEFAULT_RESOLUTION
    height: int = FLUX_2_KLEIN_DEFAULT_RESOLUTION
    radius_min: float = 6.0
    radius_max: float = 16.0
    seed: int | None = None
    output_dir: Path = Path("data")
    show_progress: bool = True

    def validate(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.radius_min <= 0 or self.radius_max <= 0:
            raise ValueError("radius bounds must be positive")
        if self.radius_min > self.radius_max:
            raise ValueError("radius_min must be <= radius_max")

        required_margin = self.radius_max + max(self.width, self.height) * 0.04
        if required_margin >= min(self.width, self.height) / 2:
            raise ValueError(
                "radius_max is too large for the image dimensions after reserving path margin"
            )
