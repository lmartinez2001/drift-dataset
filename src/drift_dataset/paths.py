from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drift_dataset.types import FloatArray, PathSpec


def sample_arc_length(
    path: PathSpec,
    frame_count: int,
    resolution: int = 2048,
) -> FloatArray:
    """Sample a parametric path at approximately uniform arc-length intervals."""
    t_dense = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    points = path.curve(t_dense)

    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
    )

    total_length = cumulative[-1]
    if total_length <= 1e-6:
        return np.repeat(points[:1], frame_count, axis=0).astype(np.float32)

    target_lengths = np.linspace(
        0.0,
        total_length,
        frame_count,
        endpoint=not path.closed,
        dtype=np.float32,
    )
    x = np.interp(target_lengths, cumulative, points[:, 0])
    y = np.interp(target_lengths, cumulative, points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def random_path(rng: np.random.Generator, width: int, height: int, radius: float) -> PathSpec:
    margin = radius + max(width, height) * 0.04
    builders = [
        _random_line,
        _random_ellipse,
        _random_lissajous,
        _random_sine_wave,
        _random_bezier,
    ]
    builder = rng.choice(builders)
    return builder(rng, width, height, margin)


def _random_line(rng: np.random.Generator, width: int, height: int, margin: float) -> PathSpec:
    start = np.array(
        [rng.uniform(margin, width - margin), rng.uniform(margin, height - margin)],
        dtype=np.float32,
    )
    end = np.array(
        [rng.uniform(margin, width - margin), rng.uniform(margin, height - margin)],
        dtype=np.float32,
    )

    def curve(t: NDArray[np.float32]) -> FloatArray:
        return (start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]).astype(
            np.float32
        )

    return PathSpec(
        name="line",
        params={
            "x0": float(start[0]),
            "y0": float(start[1]),
            "x1": float(end[0]),
            "y1": float(end[1]),
        },
        curve=curve,
    )


def _random_ellipse(rng: np.random.Generator, width: int, height: int, margin: float) -> PathSpec:
    rx_max = max(1.0, (width - 2.0 * margin) * 0.48)
    ry_max = max(1.0, (height - 2.0 * margin) * 0.48)
    rx = rng.uniform(min(width * 0.08, rx_max), rx_max)
    ry = rng.uniform(min(height * 0.08, ry_max), ry_max)
    cx = rng.uniform(margin + rx, width - margin - rx)
    cy = rng.uniform(margin + ry, height - margin - ry)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    loops = rng.choice([0.5, 0.75, 1.0, 1.25])

    def curve(t: NDArray[np.float32]) -> FloatArray:
        theta = phase + 2.0 * np.pi * loops * t
        x = cx + rx * np.cos(theta)
        y = cy + ry * np.sin(theta)
        return np.stack([x, y], axis=-1).astype(np.float32)

    return PathSpec(
        name="ellipse",
        params={
            "cx": float(cx),
            "cy": float(cy),
            "rx": float(rx),
            "ry": float(ry),
            "loops": float(loops),
        },
        curve=curve,
        closed=bool(loops == 1.0),
    )


def _random_lissajous(rng: np.random.Generator, width: int, height: int, margin: float) -> PathSpec:
    ax_max = max(1.0, (width - 2.0 * margin) * 0.48)
    ay_max = max(1.0, (height - 2.0 * margin) * 0.48)
    ax = rng.uniform(min(width * 0.10, ax_max), ax_max)
    ay = rng.uniform(min(height * 0.10, ay_max), ay_max)
    cx = rng.uniform(margin + ax, width - margin - ax)
    cy = rng.uniform(margin + ay, height - margin - ay)
    freq_x = int(rng.integers(1, 4))
    freq_y = int(rng.integers(2, 5))
    phase_x = rng.uniform(0.0, 2.0 * np.pi)
    phase_y = rng.uniform(0.0, 2.0 * np.pi)

    def curve(t: NDArray[np.float32]) -> FloatArray:
        theta = 2.0 * np.pi * t
        x = cx + ax * np.sin(freq_x * theta + phase_x)
        y = cy + ay * np.sin(freq_y * theta + phase_y)
        return np.stack([x, y], axis=-1).astype(np.float32)

    return PathSpec(
        name="lissajous",
        params={
            "cx": float(cx),
            "cy": float(cy),
            "ax": float(ax),
            "ay": float(ay),
            "freq_x": float(freq_x),
            "freq_y": float(freq_y),
        },
        curve=curve,
        closed=True,
    )


def _random_sine_wave(rng: np.random.Generator, width: int, height: int, margin: float) -> PathSpec:
    left_to_right = bool(rng.integers(0, 2))
    amplitude_max = max(1.0, (height - 2.0 * margin) * 0.48)
    amplitude = rng.uniform(min(height * 0.08, amplitude_max), amplitude_max)
    baseline = rng.uniform(margin + amplitude, height - margin - amplitude)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    cycles = rng.uniform(0.75, 2.5)
    x0, x1 = (margin, width - margin) if left_to_right else (width - margin, margin)

    def curve(t: NDArray[np.float32]) -> FloatArray:
        x = x0 * (1.0 - t) + x1 * t
        y = baseline + amplitude * np.sin(2.0 * np.pi * cycles * t + phase)
        return np.stack([x, y], axis=-1).astype(np.float32)

    return PathSpec(
        name="sine_wave",
        params={
            "baseline": float(baseline),
            "amplitude": float(amplitude),
            "cycles": float(cycles),
            "left_to_right": float(left_to_right),
        },
        curve=curve,
    )


def _random_bezier(rng: np.random.Generator, width: int, height: int, margin: float) -> PathSpec:
    points = rng.uniform(
        [margin, margin],
        [width - margin, height - margin],
        size=(4, 2),
    ).astype(np.float32)

    def curve(t: NDArray[np.float32]) -> FloatArray:
        u = 1.0 - t
        return (
            (u**3)[:, None] * points[0]
            + (3.0 * u**2 * t)[:, None] * points[1]
            + (3.0 * u * t**2)[:, None] * points[2]
            + (t**3)[:, None] * points[3]
        ).astype(np.float32)

    return PathSpec(
        name="cubic_bezier",
        params={
            f"p{i}_{axis}": float(points[i, j])
            for i in range(4)
            for j, axis in enumerate(("x", "y"))
        },
        curve=curve,
    )
