from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from drift_dataset import DatasetConfig, generate_dataset


class DatasetGenerationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
