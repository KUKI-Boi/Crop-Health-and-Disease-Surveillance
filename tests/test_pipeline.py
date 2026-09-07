"""
Comprehensive Unit Tests for Full Image-Processing Analysis Pipeline.
Verifies percentage math, zero-vegetation protection, severity thresholds,
label class extraction, classical CV inference, and structured result dict output.
"""

import unittest
import numpy as np
import cv2

from src.analysis.metrics import calculate_health_metrics
from src.analysis.severity import classify_severity
from src.image_processing.label_analysis import LabelAnalyzer
from src.image_processing.segmentation import VegetationSegmenter
from src.analysis.pipeline import CropHealthPipeline

class TestAnalysisPipeline(unittest.TestCase):

    def test_percentage_calculation(self):
        """Verify exact formulas: Disease % = (Disease / Vegetation) * 100."""
        # 1000 healthy, 250 disease -> 1250 total vegetation.
        # Healthy % = 1000/1250 * 100 = 80.0%
        # Disease % = 250/1250 * 100 = 20.0%
        metrics = calculate_health_metrics(
            healthy_pixels=1000,
            diseased_pixels=250,
            image_total_pixels=10000
        )
        self.assertEqual(metrics["vegetation_pixels"], 1250)
        self.assertEqual(metrics["healthy_percentage"], 80.0)
        self.assertEqual(metrics["disease_percentage"], 20.0)

    def test_zero_vegetation_case(self):
        """Verify division-by-zero protection when vegetation pixels equal 0."""
        metrics = calculate_health_metrics(
            healthy_pixels=0,
            diseased_pixels=0,
            image_total_pixels=5000
        )
        self.assertEqual(metrics["vegetation_pixels"], 0)
        self.assertEqual(metrics["healthy_percentage"], 0.0)
        self.assertEqual(metrics["disease_percentage"], 0.0)

    def test_severity_classification_thresholds(self):
        """Verify project thresholds: Low (0-10%), Moderate (10-30%), High (30-50%), Severe (>50%)."""
        # Low (e.g. 5%)
        sev_low = classify_severity(5.0)
        self.assertEqual(sev_low["severity_level"], "Low")

        # Low boundary (10%)
        sev_low_bound = classify_severity(10.0)
        self.assertEqual(sev_low_bound["severity_level"], "Low")

        # Moderate (e.g. 25%)
        sev_mod = classify_severity(25.0)
        self.assertEqual(sev_mod["severity_level"], "Moderate")

        # High (e.g. 45%)
        sev_high = classify_severity(45.0)
        self.assertEqual(sev_high["severity_level"], "High")

        # Severe (e.g. 65%)
        sev_severe = classify_severity(65.0)
        self.assertEqual(sev_severe["severity_level"], "Severe")

    def test_label_class_extraction(self):
        """Verify mask extraction for background (0), healthy (127), and disease (255)."""
        label_img = np.zeros((100, 100), dtype=np.uint8)
        label_img[10:50, 10:50] = 127 # 1600 pixels healthy
        label_img[20:30, 20:30] = 255 # 100 pixels disease

        analyzer = LabelAnalyzer()
        res = analyzer.analyze_label(label_img)

        self.assertEqual(np.count_nonzero(res["healthy_mask"]), 1500)
        self.assertEqual(np.count_nonzero(res["disease_mask"]), 100)

    def test_full_pipeline_structured_result(self):
        """Verify full pipeline performs inference on original image and returns structured result dictionary."""
        # Synthetic leaf image (green leaf with yellow rust spot)
        bgr_img = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_img[20:80, 20:80] = (34, 197, 94) # Green leaf (3600 pixels)
        bgr_img[40:50, 40:50] = (0, 215, 255) # Yellow disease spot (100 pixels)

        pipeline = CropHealthPipeline()
        result = pipeline.analyze_sample(
            image_input=bgr_img,
            crop_category="wheat_stripe_rust",
            image_name="test_wheat_sample"
        )

        required_keys = [
            "crop_category", "image_name", "total_image_pixels", "vegetation_pixels",
            "healthy_pixels", "disease_pixels", "healthy_percentage", "disease_percentage",
            "severity_level", "healthy_mask", "disease_mask", "colorized_map"
        ]

        for k in required_keys:
            self.assertIn(k, result)

        self.assertEqual(result["crop_category"], "wheat_stripe_rust")
        self.assertEqual(result["image_name"], "test_wheat_sample")
        self.assertEqual(result["total_image_pixels"], 10000)
        self.assertAlmostEqual(result["vegetation_pixels"], 3600, delta=20)
        self.assertGreater(result["healthy_pixels"], 0)
        self.assertGreaterEqual(result["disease_pixels"], 0)

if __name__ == "__main__":
    unittest.main()
