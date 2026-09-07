"""
Dataset Inspector Utility.
Performs dynamic inspection of ground-truth segmentation label files (*_label.png)
to discover exact pixel intensity distributions and unique color channels without assuming hardcoded values.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from src.dataset.loader import SamplePair
from src.image_processing.label_inspector import LabelInspector

class DatasetInspector:
    """
    Utility class for deep dataset inspection and label value profiling.
    """

    def __init__(self):
        self.label_inspector = LabelInspector()

    def inspect_sample_label(self, label_path: str) -> Dict[str, Any]:
        """
        Loads and inspects a single label image file.

        Args:
            label_path (str): Path to the `*_label.png` file.

        Returns:
            Dict[str, Any]: Comprehensive inspection findings.
        """
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_img is None:
            raise ValueError(f"Unable to read label image at path: {label_path}")

        inspection = self.label_inspector.inspect_label_image(label_img)
        inspection["file_path"] = label_path
        inspection["shape"] = label_img.shape
        return inspection

    def batch_inspect_category(
        self, 
        samples: List[SamplePair], 
        max_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Batch inspects multiple label files within a disease category to find global unique values.

        Args:
            samples (List[SamplePair]): List of sample pairs.
            max_samples (int): Maximum number of sample files to inspect.

        Returns:
            Dict[str, Any]: Aggregated unique values and inspection stats across samples.
        """
        valid_samples = [s for s in samples if s.is_valid and s.label_path]
        sampled_entries = valid_samples[:max_samples]

        all_unique_values = set()
        sample_results = []
        is_grayscale = True

        for sample in sampled_entries:
            try:
                res = self.inspect_sample_label(sample.label_path)
                sample_results.append({
                    "sample_id": sample.sample_id,
                    "format": res["format"],
                    "num_classes": res["num_classes"],
                    "suggested_mapping": res["suggested_mapping"],
                    "unique_values": res["unique_values"]
                })

                if res["format"] == "grayscale":
                    for item in res["unique_values"]:
                        all_unique_values.add(item["pixel_value"])
                else:
                    is_grayscale = False
                    for item in res["unique_values"]:
                        all_unique_values.add(tuple(item["color_bgr"]))

            except Exception as err:
                sample_results.append({
                    "sample_id": sample.sample_id,
                    "error": str(err)
                })

        return {
            "num_samples_inspected": len(sampled_entries),
            "format": "grayscale" if is_grayscale else "color",
            "global_unique_pixel_values": sorted(list(all_unique_values)) if is_grayscale else [list(x) for x in all_unique_values],
            "sample_inspections": sample_results
        }
