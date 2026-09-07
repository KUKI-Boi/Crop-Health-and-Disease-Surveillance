"""
Full Image-Processing Analysis Pipeline Module.
Orchestrates image loading, classical computer vision vegetation segmentation,
disease region detection, area percentage calculation, infection severity classification,
and optional ground-truth evaluation metrics (IoU, Dice).
"""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2

from src.image_processing.segmentation import VegetationSegmenter
from src.image_processing.label_analysis import LabelAnalyzer, LabelClassConfig
from src.image_processing.preprocessing import load_and_preprocess_image
from src.visualization.display import create_colorized_health_map
from src.analysis.metrics import calculate_health_metrics
from src.analysis.severity import classify_severity

class CropHealthPipeline:
    """
    Complete non-machine-learning image processing analysis pipeline.
    Performs genuine classical CV inference on original crop images to detect diseased spots,
    compute quantitative vegetation area percentages, and classify infection severity level.
    """

    def __init__(self, label_config: Optional[LabelClassConfig] = None):
        self.segmenter = VegetationSegmenter()
        self.label_analyzer = LabelAnalyzer(config=label_config)

    def evaluate_prediction_vs_ground_truth(
        self, 
        pred_disease_mask: np.ndarray, 
        gt_disease_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates IoU, Dice Similarity Coefficient, Precision, and Recall
        comparing predicted disease mask against ground-truth label mask.

        Args:
            pred_disease_mask (np.ndarray): Predicted binary disease mask.
            gt_disease_mask (np.ndarray): Ground-truth binary disease mask.

        Returns:
            Dict[str, float]: Evaluation metrics (iou, dice, precision, recall).
        """
        pred_bin = (pred_disease_mask > 0).astype(np.uint8)
        gt_bin = (gt_disease_mask > 0).astype(np.uint8)

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        pred_sum = pred_bin.sum()
        gt_sum = gt_bin.sum()

        iou = float(intersection / union) if union > 0 else 1.0
        dice = float(2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 1.0
        precision = float(intersection / pred_sum) if pred_sum > 0 else 1.0
        recall = float(intersection / gt_sum) if gt_sum > 0 else 1.0

        return {
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }

    def analyze_sample(
        self,
        image_input: Union[str, Path, np.ndarray],
        label_input: Optional[Union[str, Path, np.ndarray]] = None,
        crop_category: str = "wheat_stripe_rust",
        image_name: str = "sample"
    ) -> Dict[str, Any]:
        """
        Executes full analysis pipeline on input image.

        Args:
            image_input: File path or BGR image array.
            label_input (Optional): Ground-truth label mask array or path for evaluation if available.
            crop_category (str): Category identifier (e.g. wheat_stripe_rust).
            image_name (str): Sample image name identifier.

        Returns:
            Dict[str, Any]: Complete structured analysis result.
        """
        # 1. Load Original Image
        if isinstance(image_input, (str, Path)):
            bgr_img = cv2.imread(str(image_input))
        else:
            bgr_img = image_input

        if bgr_img is None or bgr_img.size == 0:
            raise ValueError(f"Invalid or unreadable image input for sample: {image_name}")

        h, w = bgr_img.shape[:2]
        total_img_pixels = h * w

        # 2. Genuine Classical Image Processing Inference on Original Image
        seg_result = self.segmenter.segment_healthy_vs_diseased(bgr_img)
        healthy_mask = seg_result["healthy_mask"]
        disease_mask = seg_result["diseased_mask"]

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

        # Colorized Health Map (Healthy Green / Disease Red / Background Dark)
        colorized_map = create_colorized_health_map(healthy_mask, disease_mask)

        # 5. Optional Ground-Truth Evaluation / Validation
        gt_analysis_res = None
        evaluation_metrics = None
        
        if label_input is not None:
            if isinstance(label_input, (str, Path)):
                lbl_img = cv2.imread(str(label_input), cv2.IMREAD_UNCHANGED)
            else:
                lbl_img = label_input

            if lbl_img is not None and lbl_img.size > 0:
                gt_analysis_res = self.label_analyzer.analyze_label(lbl_img)
                gt_disease_mask = gt_analysis_res["disease_mask"]
                evaluation_metrics = self.evaluate_prediction_vs_ground_truth(
                    pred_disease_mask=disease_mask,
                    gt_disease_mask=gt_disease_mask
                )

        # 6. Build Structured Result Dictionary
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
            "healthy_mask": healthy_mask,
            "disease_mask": disease_mask,
            "colorized_map": colorized_map,
            "bgr_image": bgr_img,
            "label_analysis_details": gt_analysis_res if gt_analysis_res else {"colorized_map": colorized_map},
            "evaluation_metrics": evaluation_metrics
        }

        return result
