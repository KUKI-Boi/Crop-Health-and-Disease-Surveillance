"""
Full Image-Processing Analysis Pipeline Module.
Orchestrates image loading, label mask processing, vegetation extraction,
area percentage calculation, infection severity classification, and result reporting.
"""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2

from src.image_processing.label_analysis import LabelAnalyzer, LabelClassConfig
from src.analysis.metrics import calculate_health_metrics
from src.analysis.severity import classify_severity

class CropHealthPipeline:
    """
    Complete non-machine-learning image processing analysis pipeline.
    Segments healthy vs diseased vegetation from ground-truth labels, computes area percentages,
    and classifies infection severity level.
    """

    def __init__(self, label_config: Optional[LabelClassConfig] = None):
        self.label_analyzer = LabelAnalyzer(config=label_config)

    def analyze_sample(
        self,
        image_input: Union[str, Path, np.ndarray],
        label_input: Union[str, Path, np.ndarray],
        crop_category: str = "wheat_stripe_rust",
        image_name: str = "sample"
    ) -> Dict[str, Any]:
        """
        Executes full analysis pipeline on input image and ground-truth label.

        Args:
            image_input: File path or BGR image array.
            label_input: File path or uint8 label image array.
            crop_category (str): Category identifier (e.g. wheat_stripe_rust).
            image_name (str): Sample image name identifier.

        Returns:
            Dict[str, Any]: Complete structured analysis result.
        """
        # 1. Load & Validate Image
        if isinstance(image_input, (str, Path)):
            bgr_img = cv2.imread(str(image_input))
        else:
            bgr_img = image_input

        if isinstance(label_input, (str, Path)):
            lbl_img = cv2.imread(str(label_input), cv2.IMREAD_UNCHANGED)
        else:
            lbl_img = label_input

        if lbl_img is None or lbl_img.size == 0:
            raise ValueError(f"Invalid or unreadable label input for sample: {image_name}")

        h, w = lbl_img.shape[:2]
        total_img_pixels = h * w

        # 2. Label Processing & Background Removal
        analysis_res = self.label_analyzer.analyze_label(lbl_img)
        healthy_mask = analysis_res["healthy_mask"]
        disease_mask = analysis_res["disease_mask"]

        healthy_cnt = int(np.count_nonzero(healthy_mask))
        disease_cnt = int(np.count_nonzero(disease_mask))

        # 3. Pixel-Area Calculation & Zero-Division Protection
        metrics = calculate_health_metrics(
            healthy_pixels=healthy_cnt,
            diseased_pixels=disease_cnt,
            image_total_pixels=total_img_pixels
        )

        # 4. Infection Severity Classification
        severity = classify_severity(metrics["disease_percentage"])

        # 5. Build Structured Result Dictionary
        result = {
            "crop_category": crop_category,
            "image_name": image_name,
            "total_image_pixels": metrics["total_image_pixels"],
            "vegetation_pixels": metrics["vegetation_pixels"],
            "healthy_pixels": metrics["healthy_pixels"],
            "disease_pixels": metrics["disease_pixels"],
            "healthy_percentage": metrics["healthy_percentage"],
            "disease_percentage": metrics["disease_percentage"],
            "severity_level": severity["severity_level"],
            "severity_details": severity,
            "label_analysis_details": analysis_res,
            "healthy_mask": healthy_mask,
            "disease_mask": disease_mask,
            "colorized_map": analysis_res["colorized_map"],
            "bgr_image": bgr_img
        }

        return result
