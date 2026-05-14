from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from drift_dataset.visualization import (
    find_valid_sample_files,
    is_valid_sample_file,
    load_sample_optical_flow,
    write_optical_flow_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render generated optical-flow arrays as color videos."
    )
    parser.add_argument("input", type=Path, help="path to a sample .npz file or directory")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/flow.mp4"))
    parser.add_argument("--fps", type=_positive_int, default=12)
    parser.add_argument("--scale", type=_positive_int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.is_dir():
        output_paths = _render_directory(args.input, args.output, args.fps, args.scale)
        print(f"Wrote {len(output_paths)} optical-flow videos to {args.output}")
        return

    if not is_valid_sample_file(args.input):
        raise ValueError(f"{args.input} is not a valid sample .npz file")

    optical_flow = load_sample_optical_flow(args.input, sample_index=args.sample_index)
    write_optical_flow_video(
        optical_flow=optical_flow,
        output_path=args.output,
        fps=args.fps,
        scale=args.scale,
    )
    print(f"Wrote sample {args.sample_index} optical-flow video to {args.output}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _render_directory(input_dir: Path, output_dir: Path, fps: int, scale: int) -> list[Path]:
    sample_paths = find_valid_sample_files(input_dir)
    if not sample_paths:
        raise ValueError(f"{input_dir} does not contain any valid sample .npz files")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for sample_path in tqdm(sample_paths, desc="Converting optical flow", unit="video"):
        output_path = output_dir / f"{sample_path.stem}_flow.mp4"
        optical_flow = load_sample_optical_flow(sample_path)
        write_optical_flow_video(optical_flow, output_path, fps=fps, scale=scale)
        output_paths.append(output_path)

    return output_paths
