"""
Unit tests for BatchProcessor module and dataset aggregate statistics.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from src.analysis.batch_processor import BatchProcessor

class TestBatchProcessor(unittest.TestCase):

    def setUp(self):
        """Creates mock temporary dataset and output directory."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.test_dir) / "Dataset"
        self.output_dir = Path(self.test_dir) / "results"

        # Create category subfolder
        self.wheat_dir = self.dataset_dir / "wheat_stripe_rust"
        self.wheat_dir.mkdir(parents=True, exist_ok=True)

        # Create 2 valid image-label pairs
        dummy_img = np.full((100, 100, 3), 120, dtype=np.uint8)

        # Pair 1: Low infection (5% disease)
        lbl1 = np.zeros((100, 100), dtype=np.uint8)
        lbl1[20:80, 20:80] = 127 # 3600 healthy
        lbl1[30:35, 30:35] = 255 # 25 disease

        cv2.imwrite(str(self.wheat_dir / "sample01.jpg"), dummy_img)
        cv2.imwrite(str(self.wheat_dir / "sample01_label.png"), lbl1)

        # Pair 2: Severe infection (60% disease)
        lbl2 = np.zeros((100, 100), dtype=np.uint8)
        lbl2[20:80, 20:80] = 127
        lbl2[20:60, 20:80] = 255 # Large disease area

        cv2.imwrite(str(self.wheat_dir / "sample02.jpg"), dummy_img)
        cv2.imwrite(str(self.wheat_dir / "sample02_label.png"), lbl2)

    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.test_dir)

    def test_summary_stats_computation(self):
        """Test aggregate metrics calculation logic."""
        mock_df = pd.DataFrame([
            {"image_name": "s1", "disease_percentage": 5.0, "severity": "Low"},
            {"image_name": "s2", "disease_percentage": 25.0, "severity": "Moderate"},
            {"image_name": "s3", "disease_percentage": 45.0, "severity": "High"},
            {"image_name": "s4", "disease_percentage": 65.0, "severity": "Severe"}
        ])

        stats = BatchProcessor.compute_summary_stats(mock_df)
        self.assertEqual(stats["total_images_analyzed"], 4)
        self.assertEqual(stats["average_disease_percentage"], 35.0)
        self.assertEqual(stats["minimum_disease_percentage"], 5.0)
        self.assertEqual(stats["maximum_disease_percentage"], 65.0)
        self.assertEqual(stats["num_low_severity"], 1)
        self.assertEqual(stats["num_moderate_severity"], 1)
        self.assertEqual(stats["num_high_severity"], 1)
        self.assertEqual(stats["num_severe_severity"], 1)

    def test_batch_process_category(self):
        """Test processing category folder and writing CSV file."""
        processor = BatchProcessor(dataset_dir=str(self.dataset_dir), output_dir=str(self.output_dir))
        res = processor.process_category("wheat_stripe_rust")

        self.assertEqual(res["total_pairs_discovered"], 2)
        self.assertTrue(Path(res["csv_path"]).exists())

        df = res["results_df"]
        expected_cols = [
            "image_name", "disease_type", "healthy_pixels", "disease_pixels",
            "vegetation_pixels", "healthy_percentage", "disease_percentage", "severity"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

if __name__ == "__main__":
    unittest.main()
