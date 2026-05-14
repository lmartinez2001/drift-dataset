from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from drift_dataset import DatasetConfig, generate_dataset
from drift_dataset.config import FLUX_2_KLEIN_DEFAULT_RESOLUTION
from drift_dataset.flow import compute_optical_flow


class DatasetGenerationTest(unittest.TestCase):
    def test_default_resolution_matches_flux_2_klein_4b(self) -> None:
        config = DatasetConfig(sample_count=1, frame_count=1, show_progress=False)

        self.assertEqual(config.width, FLUX_2_KLEIN_DEFAULT_RESOLUTION)
        self.assertEqual(config.height, FLUX_2_KLEIN_DEFAULT_RESOLUTION)
        self.assertEqual(FLUX_2_KLEIN_DEFAULT_RESOLUTION, 1024)

    def test_generate_dataset_writes_one_hashed_file_per_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = DatasetConfig(
                sample_count=4,
                frame_count=6,
                width=64,
                height=48,
                seed=1,
                output_dir=output_dir,
                show_progress=False,
            )

            output_paths = generate_dataset(config)

            self.assertEqual(len(output_paths), 4)
            self.assertEqual(len(set(output_paths)), 4)

            for output_path in output_paths:
                self.assertEqual(output_path.parent, output_dir)
                self.assertEqual(output_path.suffix, ".npz")
                self.assertTrue(output_path.exists())

            data = np.load(output_paths[0])
            self.assertEqual(data["frames"].shape, (64, 48, 6))
            self.assertEqual(data["optical_flow"].shape, (64, 48, 5, 2))
            self.assertEqual(data["coords"].shape, (6, 2))
            self.assertEqual(data["frames"].dtype, np.float32)
            self.assertEqual(data["optical_flow"].dtype, np.float32)
            self.assertEqual(float(data["frames"].min()), 0.0)
            self.assertEqual(float(data["frames"].max()), 1.0)
            self.assertEqual(output_paths[0].stem, str(data["sample_hash"]))

            metadata = json.loads(str(data["metadata_json"]))
            self.assertIn("path_name", metadata)
            self.assertIn("path_params", metadata)
            self.assertIn("radius", metadata)

    def test_optical_flow_uses_current_frame_support_only(self) -> None:
        coords = np.array([[1.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        masks = np.zeros((2, 3, 5), dtype=bool)
        masks[0, 1, 1] = True
        masks[1, 1, 3] = True

        flow = compute_optical_flow(coords, masks)
        nonzero_support = np.linalg.norm(flow[:, :, 0], axis=-1) > 0.0

        expected_support = np.zeros((5, 3), dtype=bool)
        expected_support[1, 1] = True
        self.assertTrue(np.array_equal(nonzero_support, expected_support))
        self.assertEqual(float(flow[1, 1, 0, 0]), 2.0)
        self.assertEqual(float(flow[1, 1, 0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
