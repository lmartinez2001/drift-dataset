from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from drift_dataset.config import DatasetConfig
from drift_dataset.flow import compute_optical_flow
from drift_dataset.paths import random_path, sample_arc_length
from drift_dataset.rendering import render_circle_sequence
from drift_dataset.types import Sample


def generate_sample(
    rng: np.random.Generator,
    frame_count: int,
    width: int,
    height: int,
    radius_range: tuple[float, float],
) -> Sample:
    radius = float(rng.uniform(*radius_range))
    path = random_path(rng, width, height, radius)
    coords = sample_arc_length(path, frame_count)
    frames, masks = render_circle_sequence(coords, width, height, radius)
    optical_flow = compute_optical_flow(coords, masks)

    metadata = {
        "path_name": path.name,
        "path_params": path.params,
        "radius": radius,
        "width": width,
        "height": height,
        "closed": path.closed,
    }
    return Sample(
        frames=frames,
        optical_flow=optical_flow,
        coords=coords,
        metadata=metadata,
    )


def generate_dataset(config: DatasetConfig) -> list[Path]:
    config.validate()

    rng = np.random.default_rng(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    with tqdm(
        total=config.sample_count,
        desc="Creating samples",
        unit="sample",
        disable=not config.show_progress,
    ) as progress:
        while len(output_paths) < config.sample_count:
            sample = generate_sample(
                rng=rng,
                frame_count=config.frame_count,
                width=config.width,
                height=config.height,
                radius_range=(config.radius_min, config.radius_max),
            )
            sample_hash = compute_sample_hash(sample)
            output_path = config.output_dir / f"{sample_hash}.npz"
            if output_path.exists():
                continue

            write_sample_npz(sample, output_path, sample_hash)
            output_paths.append(output_path)
            progress.update()

    return output_paths


def write_sample_npz(sample: Sample, output_path: Path, sample_hash: str | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hash_value = sample_hash if sample_hash is not None else compute_sample_hash(sample)
    np.savez_compressed(
        output_path,
        frames=sample.frames,
        optical_flow=sample.optical_flow,
        coords=sample.coords,
        path_name=np.array(str(sample.metadata["path_name"])),
        metadata_json=np.array(json.dumps(sample.metadata)),
        sample_hash=np.array(hash_value),
    )


def compute_sample_hash(sample: Sample, digest_size: int = 16) -> str:
    hasher = hashlib.sha256()
    _update_array_hash(hasher, sample.frames)
    _update_array_hash(hasher, sample.optical_flow)
    _update_array_hash(hasher, sample.coords)
    metadata_json = json.dumps(sample.metadata, sort_keys=True, separators=(",", ":"))
    hasher.update(metadata_json.encode("utf-8"))
    return hasher.hexdigest()[:digest_size]


def _update_array_hash(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())
