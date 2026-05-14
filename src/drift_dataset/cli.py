from __future__ import annotations

import argparse
from pathlib import Path

from drift_dataset.config import DatasetConfig, FLUX_2_KLEIN_DEFAULT_RESOLUTION
from drift_dataset.dataset import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate moving-circle image sequences and analytic optical flow."
    )
    parser.add_argument("--frames", type=_positive_int, help="frames per sample")
    parser.add_argument("--samples", type=_positive_int, help="number of samples")
    parser.add_argument("--width", type=_positive_int, default=FLUX_2_KLEIN_DEFAULT_RESOLUTION)
    parser.add_argument("--height", type=_positive_int, default=FLUX_2_KLEIN_DEFAULT_RESOLUTION)
    parser.add_argument("--radius-min", type=float, default=6.0)
    parser.add_argument("--radius-max", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_count = args.frames if args.frames is not None else _prompt_int("Frames per sample")
    sample_count = args.samples if args.samples is not None else _prompt_int("Number of samples")
    config = DatasetConfig(
        sample_count=sample_count,
        frame_count=frame_count,
        width=args.width,
        height=args.height,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    output_paths = generate_dataset(config)

    print(f"Wrote {len(output_paths)} samples to {config.output_dir}")
    print(f"sample frames shape: ({config.width}, {config.height}, {config.frame_count})")
    print(
        "sample optical_flow shape: "
        f"({config.width}, {config.height}, {max(config.frame_count - 1, 0)}, 2)"
    )
    print(f"first sample: {output_paths[0]}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _prompt_int(label: str) -> int:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value > 0:
            return value
        print("Please enter a positive integer.")
