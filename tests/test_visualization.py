from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from drift_dataset.visualization import (
    find_valid_sample_files,
    is_valid_sample_file,
    optical_flow_to_video_frames,
    sample_to_video_frames,
    write_optical_flow_video,
    write_sample_video,
)


class VisualizationTest(unittest.TestCase):
    def test_sample_to_video_frames_converts_layout(self) -> None:
        sample = np.zeros((3, 2, 4), dtype=np.float32)
        sample[1, 0, 2] = 1.0

        video = sample_to_video_frames(sample, scale=2)

        self.assertEqual(video.shape, (4, 4, 6, 3))
        self.assertEqual(video.dtype, np.uint8)
        self.assertEqual(int(video[2, 0, 2, 0]), 255)

    def test_write_sample_video_creates_file(self) -> None:
        sample = np.zeros((8, 8, 3), dtype=np.float32)
        sample[2:5, 2:5, :] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample.gif"
            write_sample_video(sample, output_path, fps=3)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_optical_flow_to_video_frames_encodes_direction_and_magnitude(self) -> None:
        flow = np.zeros((3, 2, 2, 2), dtype=np.float32)
        flow[1, 0, 0] = [1.0, 0.0]
        flow[1, 0, 1] = [0.0, 0.5]

        video = optical_flow_to_video_frames(flow, scale=2)

        self.assertEqual(video.shape, (2, 4, 6, 3))
        self.assertEqual(video.dtype, np.uint8)
        self.assertGreater(int(video[0, 0, 2, 0]), 200)
        self.assertEqual(int(video[0, 0, 0].max()), 0)

    def test_write_optical_flow_video_creates_file(self) -> None:
        flow = np.zeros((8, 8, 3, 2), dtype=np.float32)
        flow[2:5, 2:5, :, 0] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "flow.gif"
            write_optical_flow_video(flow, output_path, fps=3)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_find_valid_sample_files_checks_extension_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid = root / "valid.npz"
            missing_key = root / "missing_key.npz"
            wrong_extension = root / "wrong.txt"

            np.savez(
                valid,
                frames=np.zeros((4, 4, 2), dtype=np.float32),
                optical_flow=np.zeros((4, 4, 1, 2), dtype=np.float32),
                coords=np.zeros((2, 2), dtype=np.float32),
                path_name=np.array("line"),
                metadata_json=np.array("{}"),
                sample_hash=np.array("valid"),
            )
            np.savez(missing_key, frames=np.zeros((4, 4, 2), dtype=np.float32))
            wrong_extension.write_text("not a sample")

            self.assertTrue(is_valid_sample_file(valid))
            self.assertFalse(is_valid_sample_file(missing_key))
            self.assertFalse(is_valid_sample_file(wrong_extension))
            self.assertEqual(find_valid_sample_files(root), [valid])


if __name__ == "__main__":
    unittest.main()
