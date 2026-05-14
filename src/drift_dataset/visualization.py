from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from drift_dataset.types import FloatArray


REQUIRED_SAMPLE_KEYS = frozenset(
    {"frames", "optical_flow", "coords", "path_name", "metadata_json", "sample_hash"}
)


def find_valid_sample_files(input_dir: Path) -> list[Path]:
    valid_paths: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if is_valid_sample_file(path):
            valid_paths.append(path)
    return valid_paths


def is_valid_sample_file(path: Path) -> bool:
    if path.suffix.lower() != ".npz":
        return False

    try:
        with np.load(path) as data:
            return REQUIRED_SAMPLE_KEYS.issubset(data.files)
    except Exception:
        return False


def load_sample_frames(dataset_path: Path, sample_index: int = 0) -> FloatArray:
    data = np.load(dataset_path)
    if "frames" not in data:
        raise ValueError(f"{dataset_path} does not contain a 'frames' array")

    frames = data["frames"]
    if frames.ndim == 3:
        if sample_index != 0:
            raise IndexError("sample_index must be 0 when loading a single-sample array")
        return frames.astype(np.float32, copy=False)
    if frames.ndim != 4:
        raise ValueError("expected frames with shape (W, H, T) or (N, W, H, T)")
    if sample_index < 0 or sample_index >= frames.shape[0]:
        raise IndexError(f"sample_index must be in [0, {frames.shape[0] - 1}]")

    return frames[sample_index].astype(np.float32, copy=False)


def load_sample_optical_flow(dataset_path: Path, sample_index: int = 0) -> FloatArray:
    data = np.load(dataset_path)
    if "optical_flow" not in data:
        raise ValueError(f"{dataset_path} does not contain an 'optical_flow' array")

    flow = data["optical_flow"]
    if flow.ndim == 4:
        if sample_index != 0:
            raise IndexError("sample_index must be 0 when loading a single-sample array")
        return flow.astype(np.float32, copy=False)
    if flow.ndim != 5:
        raise ValueError("expected optical_flow with shape (W, H, T, 2) or (N, W, H, T, 2)")
    if sample_index < 0 or sample_index >= flow.shape[0]:
        raise IndexError(f"sample_index must be in [0, {flow.shape[0] - 1}]")

    return flow[sample_index].astype(np.float32, copy=False)


def sample_to_video_frames(sample_frames: FloatArray, scale: int = 1) -> np.ndarray:
    if sample_frames.ndim != 3:
        raise ValueError("sample_frames must have shape (W, H, T)")
    if scale <= 0:
        raise ValueError("scale must be positive")

    normalized = _normalize_to_uint8(sample_frames)
    video = np.moveaxis(normalized, 2, 0)
    video = np.transpose(video, (0, 2, 1))
    video = np.repeat(video[:, :, :, None], 3, axis=3)

    if scale > 1:
        video = np.repeat(np.repeat(video, scale, axis=1), scale, axis=2)

    return video


def optical_flow_to_video_frames(optical_flow: FloatArray, scale: int = 1) -> np.ndarray:
    if optical_flow.ndim != 4 or optical_flow.shape[-1] != 2:
        raise ValueError("optical_flow must have shape (W, H, T, 2)")
    if scale <= 0:
        raise ValueError("scale must be positive")

    dx = np.transpose(optical_flow[..., 0], (2, 1, 0))
    dy = np.transpose(optical_flow[..., 1], (2, 1, 0))
    magnitude = np.hypot(dx, dy)
    max_magnitude = float(np.max(magnitude))
    value = np.zeros_like(magnitude, dtype=np.float32)
    if max_magnitude > 0.0:
        value = (magnitude / max_magnitude).astype(np.float32)

    video = np.repeat((value * 255.0).astype(np.uint8)[:, :, :, None], 3, axis=3)

    if scale > 1:
        video = np.repeat(np.repeat(video, scale, axis=1), scale, axis=2)

    return video


def write_sample_video(
    sample_frames: FloatArray,
    output_path: Path,
    fps: int = 12,
    scale: int = 1,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = sample_to_video_frames(sample_frames, scale=scale)
    _write_video(video, output_path, fps)


def write_optical_flow_video(
    optical_flow: FloatArray,
    output_path: Path,
    fps: int = 12,
    scale: int = 1,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = optical_flow_to_video_frames(optical_flow, scale=scale)
    _write_video(video, output_path, fps)


def _write_video(video: np.ndarray, output_path: Path, fps: int) -> None:
    if output_path.suffix.lower() == ".gif":
        iio.imwrite(output_path, video, duration=1000.0 / fps)
        return

    iio.imwrite(output_path, video, fps=fps)


def _normalize_to_uint8(frames: FloatArray) -> np.ndarray:
    min_value = float(np.min(frames))
    max_value = float(np.max(frames))
    if max_value <= min_value:
        return np.zeros(frames.shape, dtype=np.uint8)

    normalized = (frames - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
