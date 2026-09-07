"""
Batch Processor Module.
Executes batch image-processing analysis across entire disease folders or full datasets.
Generates structured CSV files and aggregate summary statistics.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.dataset.loader import DatasetLoader, SamplePair
from src.analysis.pipeline import CropHealthPipeline
from src.image_processing.label_analysis import LabelClassConfig

class BatchProcessor:
    """
    Automates batch analysis over dataset directories, executing the CropHealthPipeline
    on all valid image-label pairs, generating summary statistics, and saving CSV reports.
    """

    def __init__(self, dataset_dir: str = "Dataset", output_dir: str = "results"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DatasetLoader(str(self.dataset_dir))
        self.pipeline = CropHealthPipeline()

    def process_category(
        self, 
        category_name: str, 
        custom_config: Optional[LabelClassConfig] = None
    ) -> Dict[str, Any]:
        """
        Processes all valid image-label pairs within a specific disease category folder.

        Args:
            category_name (str): Subdirectory name (e.g. wheat_stripe_rust).
            custom_config (LabelClassConfig, optional): Class mapping overrides.

        Returns:
            Dict[str, Any]: Contains sample results list, summary stats dict, and DataFrame.
        """
        if custom_config:
            self.pipeline = CropHealthPipeline(label_config=custom_config)

        all_samples = self.loader.discover_samples()
        valid_samples = [
            s for s in all_samples 
            if s.disease_category == category_name and s.is_valid and s.label_path and s.image_path
        ]

        results_rows = []
        for sample in valid_samples:
            try:
                res = self.pipeline.analyze_sample(
                    image_input=sample.image_path,
                    label_input=sample.label_path,
                    crop_category=sample.disease_category,
                    image_name=sample.sample_id
                )

                row = {
                    "image_name": sample.sample_id,
                    "disease_type": sample.disease_category,
                    "healthy_pixels": res["healthy_pixels"],
                    "disease_pixels": res["disease_pixels"],
                    "vegetation_pixels": res["vegetation_pixels"],
                    "healthy_percentage": res["healthy_percentage"],
                    "disease_percentage": res["disease_percentage"],
                    "severity": res["severity_level"]
                }
                results_rows.append(row)

            except Exception as err:
                print(f"Error processing sample '{sample.sample_id}': {err}")

        df_results = pd.DataFrame(results_rows)
        summary_stats = self.compute_summary_stats(df_results)

        # Save CSV file to output directory
        csv_filename = self.output_dir / f"batch_{category_name}.csv"
        df_results.to_csv(csv_filename, index=False)

        return {
            "category": category_name,
            "total_pairs_discovered": len(valid_samples),
            "results_df": df_results,
            "summary_stats": summary_stats,
            "csv_path": str(csv_filename)
        }

    def process_all_categories(
        self, 
        custom_config: Optional[LabelClassConfig] = None
    ) -> Dict[str, Any]:
        """
        Processes all disease categories present in the dataset directory.

        Returns:
            Dict[str, Any]: Aggregated summary table, combined DataFrame, and CSV path.
        """
        if custom_config:
            self.pipeline = CropHealthPipeline(label_config=custom_config)

        all_samples = self.loader.discover_samples()
        valid_samples = [s for s in all_samples if s.is_valid and s.label_path and s.image_path]

        results_rows = []
        for sample in valid_samples:
            try:
                res = self.pipeline.analyze_sample(
                    image_input=sample.image_path,
                    label_input=sample.label_path,
                    crop_category=sample.disease_category,
                    image_name=sample.sample_id
                )

                row = {
                    "image_name": sample.sample_id,
                    "disease_type": sample.disease_category,
                    "healthy_pixels": res["healthy_pixels"],
                    "disease_pixels": res["disease_pixels"],
                    "vegetation_pixels": res["vegetation_pixels"],
                    "healthy_percentage": res["healthy_percentage"],
                    "disease_percentage": res["disease_percentage"],
                    "severity": res["severity_level"]
                }
                results_rows.append(row)

            except Exception as err:
                print(f"Error processing sample '{sample.sample_id}': {err}")

        df_results = pd.DataFrame(results_rows)
        summary_stats = self.compute_summary_stats(df_results)

        # Save master CSV report
        csv_filename = self.output_dir / "batch_analysis_all.csv"
        df_results.to_csv(csv_filename, index=False)

        return {
            "category": "All Disease Categories",
            "total_pairs_discovered": len(valid_samples),
            "results_df": df_results,
            "summary_stats": summary_stats,
            "csv_path": str(csv_filename)
        }

    @staticmethod
    def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes aggregate metrics from a batch results DataFrame.

        Required Statistics:
        - total images analyzed
        - average disease percentage
        - minimum disease percentage
        - maximum disease percentage
        - number of low severity images
        - number of moderate severity images
        - number of high severity images
        - number of severe images
        """
        if df.empty:
            return {
                "total_images_analyzed": 0,
                "average_disease_percentage": 0.0,
                "minimum_disease_percentage": 0.0,
                "maximum_disease_percentage": 0.0,
                "num_low_severity": 0,
                "num_moderate_severity": 0,
                "num_high_severity": 0,
                "num_severe_severity": 0
            }

        total_count = len(df)
        disease_pcts = df["disease_percentage"].astype(float)
        severities = df["severity"].astype(str)

        avg_pct = round(float(disease_pcts.mean()), 2)
        min_pct = round(float(disease_pcts.min()), 2)
        max_pct = round(float(disease_pcts.max()), 2)

        num_low = int((severities == "Low").sum())
        num_moderate = int((severities == "Moderate").sum())
        num_high = int((severities == "High").sum())
        num_severe = int((severities == "Severe").sum())

        return {
            "total_images_analyzed": total_count,
            "average_disease_percentage": avg_pct,
            "minimum_disease_percentage": min_pct,
            "maximum_disease_percentage": max_pct,
            "num_low_severity": num_low,
            "num_moderate_severity": num_moderate,
            "num_high_severity": num_high,
            "num_severe_severity": num_severe
        }
