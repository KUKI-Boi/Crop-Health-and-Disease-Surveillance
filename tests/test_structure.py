"""
Unit tests for Crop Health project modules.
Verifies label inspection, metrics calculations, severity classification, and configurations.
"""

import unittest
import numpy as np
from src.config import SUPPORTED_CROPS, SEVERITY_THRESHOLDS
from src.image_processing.label_inspector import LabelInspector
from src.analysis.metrics import calculate_health_metrics
from src.analysis.severity import classify_severity

class TestCropHealthPipeline(unittest.TestCase):

    def test_config_loaded(self):
        """Ensure supported crop types and severity thresholds are defined."""
        self.assertIn("wheat_stripe_rust", SUPPORTED_CROPS)
        self.assertIn("soybean_bacterial_blight", SUPPORTED_CROPS)
        self.assertIn("cedar_apple_rust", SUPPORTED_CROPS)
        self.assertIn("Moderate", SEVERITY_THRESHOLDS)

    def test_label_inspector_grayscale(self):
        """Test dynamic label inspection on synthetic grayscale label array."""
        inspector = LabelInspector()
        # Create synthetic label with 0 (background), 128 (healthy), 255 (diseased)
        dummy_label = np.zeros((100, 100), dtype=np.uint8)
        dummy_label[20:60, 20:60] = 128 # Healthy
        dummy_label[30:40, 30:40] = 255 # Diseased

        res = inspector.inspect_label_image(dummy_label)
        self.assertEqual(res["format"], "grayscale")
        self.assertEqual(res["num_classes"], 3)
        self.assertEqual(res["suggested_mapping"]["background"], 0)
        self.assertEqual(res["suggested_mapping"]["healthy_leaf"], 128)
        self.assertEqual(res["suggested_mapping"]["disease_spot"], 255)

    def test_metrics_calculation(self):
        """Test mathematical accuracy of pixel metric calculations."""
        metrics = calculate_health_metrics(
            healthy_pixels=750,
            diseased_pixels=250,
            image_total_pixels=10000
        )
        self.assertEqual(metrics["vegetation_pixels"], 1000)
        self.assertEqual(metrics["healthy_percentage"], 75.0)
        self.assertEqual(metrics["disease_percentage"], 25.0)
        self.assertEqual(metrics["frame_coverage_pct"], 10.0)

    def test_severity_classification(self):
        """Test severity level mapping according to project thresholds."""
        sev_low = classify_severity(8.5)
        self.assertEqual(sev_low["severity_level"], "Low")

        sev_mod = classify_severity(22.0)
        self.assertEqual(sev_mod["severity_level"], "Moderate")

        sev_severe = classify_severity(65.0)
        self.assertEqual(sev_severe["severity_level"], "Severe")

if __name__ == "__main__":
    unittest.main()
