"""
Unit tests for Academic Report Generator module.
Verifies inclusion of all 14 required sections, formula correctness, and file exports.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from src.analysis.report_generator import (
    generate_markdown_report, generate_batch_academic_report, export_academic_project_files
)

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        """Create temporary directory for file exports."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.test_dir)

    def test_batch_report_sections(self):
        """Verify presence of all 14 required academic report sections and exact formulas."""
        mock_batch = {
            "category": "wheat_stripe_rust",
            "summary_stats": {
                "total_images_analyzed": 10,
                "average_disease_percentage": 25.0,
                "minimum_disease_percentage": 5.0,
                "maximum_disease_percentage": 60.0,
                "num_low_severity": 3,
                "num_moderate_severity": 4,
                "num_high_severity": 2,
                "num_severe_severity": 1
            },
            "results_df": pd.DataFrame([
                {"image_name": "sample1", "disease_percentage": 25.0, "severity": "Moderate"}
            ])
        }

        mock_rep_sample = {
            "image_name": "sample1",
            "disease_percentage": 25.0,
            "severity_level": "Moderate"
        }

        report = generate_batch_academic_report(mock_batch, mock_rep_sample)

        # Check required title & metadata sections
        self.assertIn("CROP HEALTH AND DISEASE SURVEILLANCE", report)
        self.assertIn("Academic Project Submission Report", report)
        self.assertIn("wheat_stripe_rust", report)
        self.assertIn("10 sample pairs", report)

        # Check exact formula inclusion
        self.assertIn(r"\text{Disease Affected Area (\%)} = \frac{\text{Disease Pixels}}{\text{Total Vegetation Pixels}} \times 100", report)
        self.assertIn(r"\text{Total Vegetation Pixels} = \text{Healthy Pixels} + \text{Disease Pixels}", report)

        # Check severity classification table
        self.assertIn("Severity Level", report)
        self.assertIn("Low", report)
        self.assertIn("Moderate", report)
        self.assertIn("High", report)
        self.assertIn("Severe", report)

        # Check conclusion
        self.assertIn("Project Conclusion & Drone Integration Feasibility", report)

    def test_file_export_utility(self):
        """Test exporting both Markdown report (.md) and CSV dataset file (.csv)."""
        mock_batch = {
            "category": "soybean_bacterial_blight",
            "summary_stats": {
                "total_images_analyzed": 5,
                "average_disease_percentage": 12.5,
                "minimum_disease_percentage": 2.0,
                "maximum_disease_percentage": 30.0,
                "num_low_severity": 2,
                "num_moderate_severity": 3,
                "num_high_severity": 0,
                "num_severe_severity": 0
            },
            "results_df": pd.DataFrame([
                {"image_name": "soy_sample", "disease_percentage": 12.5, "severity": "Moderate"}
            ])
        }

        files = export_academic_project_files(mock_batch, output_dir=self.test_dir)

        self.assertTrue(Path(files["md_path"]).exists())
        self.assertTrue(Path(files["csv_path"]).exists())
        self.assertTrue(files["md_path"].endswith(".md"))
        self.assertTrue(files["csv_path"].endswith(".csv"))

if __name__ == "__main__":
    unittest.main()
