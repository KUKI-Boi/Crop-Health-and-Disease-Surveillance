"""
Unit tests for Label Analysis and Diagnostic Visualization module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

from src.image_processing.label_analysis import LabelAnalyzer, LabelClassConfig

class TestLabelAnalysis(unittest.TestCase):

    def setUp(self):
        """Create temporary directory for diagnostic figure output."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.test_dir)

    def test_grayscale_label_analysis(self):
        """Test profiling 1-channel label with 0 (background), 127 (healthy), 255 (disease)."""
        label_img = np.zeros((100, 100), dtype=np.uint8)
        label_img[20:80, 20:80] = 127 # Healthy
        label_img[40:60, 40:60] = 255 # Disease

        analyzer = LabelAnalyzer()
        res = analyzer.analyze_label(label_img)

        self.assertEqual(res["format"], "grayscale")
        self.assertEqual(res["num_unique_classes"], 3)
        self.assertEqual(res["total_pixels"], 10000)

        # Check non-zero pixel counts in generated binary masks
        self.assertEqual(np.count_nonzero(res["background_mask"]), 6400) # 10000 - 3600
        self.assertEqual(np.count_nonzero(res["healthy_mask"]), 3200)   # 3600 - 400
        self.assertEqual(np.count_nonzero(res["disease_mask"]), 400)     # 20x20

    def test_custom_configurable_mapping(self):
        """Test custom class value assignment (e.g. background=10, healthy=50, disease=90)."""
        label_img = np.full((50, 50), 10, dtype=np.uint8)
        label_img[10:30, 10:30] = 50
        label_img[15:20, 15:20] = 90

        custom_cfg = LabelClassConfig(
            background_val=10,
            healthy_val=50,
            disease_val=90
        )
        analyzer = LabelAnalyzer(config=custom_cfg)
        res = analyzer.analyze_label(label_img)

        self.assertEqual(np.count_nonzero(res["healthy_mask"]), 400 - 25)
        self.assertEqual(np.count_nonzero(res["disease_mask"]), 25)

    def test_diagnostic_visualization_export(self):
        """Test diagnostic 3-panel plot generation and file saving."""
        orig_img = np.full((100, 100, 3), 120, dtype=np.uint8)
        label_img = np.zeros((100, 100), dtype=np.uint8)
        label_img[30:70, 30:70] = 127
        label_img[45:55, 45:55] = 255

        analyzer = LabelAnalyzer()
        res = analyzer.analyze_label(label_img)

        fig, saved_path = analyzer.generate_diagnostic_visualization(
            original_bgr=orig_img,
            label_img=label_img,
            analysis_result=res,
            sample_id="test_sample",
            save_dir=self.test_dir
        )

        self.assertIsNotNone(fig)
        self.assertTrue(Path(saved_path).exists())
        self.assertTrue(saved_path.endswith("diagnostic_test_sample.png"))

if __name__ == "__main__":
    unittest.main()
