"""
Unit tests for Dataset Discovery, Pairing, Validation, and Inspection utilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

from src.dataset.loader import DatasetLoader
from src.dataset.inspector import DatasetInspector

class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        """Creates a temporary mock dataset directory structure for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.test_dir) / "Dataset"

        # Create disease category subdirectories
        self.wheat_dir = self.dataset_path / "wheat_stripe_rust"
        self.soybean_dir = self.dataset_path / "soybean_bacterial_blight"
        
        self.wheat_dir.mkdir(parents=True, exist_ok=True)
        self.soybean_dir.mkdir(parents=True, exist_ok=True)

        # Create mock valid sample: sample01.jpg + sample01_label.png + sample01_black.png
        dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        dummy_label = np.zeros((100, 100), dtype=np.uint8)
        dummy_label[20:60, 20:60] = 100 # Healthy
        dummy_label[40:50, 40:50] = 200 # Diseased

        cv2.imwrite(str(self.wheat_dir / "sample01.jpg"), dummy_img)
        cv2.imwrite(str(self.wheat_dir / "sample01_label.png"), dummy_label)
        cv2.imwrite(str(self.wheat_dir / "sample01_black.png"), dummy_img)

        # Create mock sample with missing label: sample02.jpg (no label file)
        cv2.imwrite(str(self.soybean_dir / "sample02.jpg"), dummy_img)

    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.test_dir)

    def test_discover_samples_and_pairing(self):
        """Test dataset discovery and image-label pair matching."""
        loader = DatasetLoader(str(self.dataset_path))
        samples = loader.discover_samples()

        self.assertEqual(len(samples), 2)
        
        # Check wheat sample (valid pair)
        wheat_sample = next(s for s in samples if s.sample_id == "sample01")
        self.assertTrue(wheat_sample.is_valid)
        self.assertEqual(wheat_sample.disease_category, "wheat_stripe_rust")
        self.assertIsNotNone(wheat_sample.image_path)
        self.assertIsNotNone(wheat_sample.label_path)
        self.assertIsNotNone(wheat_sample.black_bg_path)

        # Check soybean sample (missing label pair)
        soybean_sample = next(s for s in samples if s.sample_id == "sample02")
        self.assertFalse(soybean_sample.is_valid)
        self.assertEqual(soybean_sample.disease_category, "soybean_bacterial_blight")
        self.assertIsNotNone(soybean_sample.image_path)
        self.assertIsNone(soybean_sample.label_path)
        self.assertIn("Missing segmentation label", soybean_sample.validation_error)

    def test_dataset_summary_df(self):
        """Test summary metrics and DataFrame generation."""
        loader = DatasetLoader(str(self.dataset_path))
        summary = loader.get_summary()

        self.assertTrue(summary["directory_exists"])
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["total_valid_pairs"], 1)
        self.assertEqual(summary["total_missing_labels"], 1)
        self.assertIn("summary_df", summary)
        self.assertFalse(summary["summary_df"].empty)

    def test_dynamic_label_inspection(self):
        """Test dynamic label inspection on mock label file without hardcoded assumptions."""
        inspector = DatasetInspector()
        label_file = str(self.wheat_dir / "sample01_label.png")
        
        inspection = inspector.inspect_sample_label(label_file)
        self.assertEqual(inspection["format"], "grayscale")
        self.assertEqual(inspection["num_classes"], 3)
        self.assertEqual(inspection["suggested_mapping"]["background"], 0)
        self.assertEqual(inspection["suggested_mapping"]["healthy_leaf"], 100)
        self.assertEqual(inspection["suggested_mapping"]["disease_spot"], 200)

if __name__ == "__main__":
    unittest.main()
