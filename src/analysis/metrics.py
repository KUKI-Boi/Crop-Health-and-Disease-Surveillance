"""
Metrics Calculation Module.
Computes quantitative pixel area metrics, healthy percentage, and disease-affected area percentage.
Ensures background pixels are excluded from the vegetation area denominator.
Provides division-by-zero protection.
"""

from typing import Dict, Any
import numpy as np

def calculate_health_metrics(
    healthy_pixels: int, 
    diseased_pixels: int,
    image_total_pixels: int
) -> Dict[str, Any]:
    """
    Computes vegetation area and relative healthy / disease percentages.

    Formulas:
        Vegetation Area = Healthy Pixels + Disease Pixels
        Healthy Area (%) = (Healthy Pixels / Vegetation Area) * 100
        Disease Affected Area (%) = (Disease Pixels / Vegetation Area) * 100

    Args:
        healthy_pixels (int): Count of healthy vegetation pixels.
        diseased_pixels (int): Count of diseased vegetation pixels.
        image_total_pixels (int): Total pixel count of the frame.

    Returns:
        Dict[str, Any]: Calculated numerical area metrics.
    """
    healthy_cnt = max(0, int(healthy_pixels))
    diseased_cnt = max(0, int(diseased_pixels))
    vegetation_area = healthy_cnt + diseased_cnt
    total_img_pixels = max(1, int(image_total_pixels))

    # Frame coverage ratio (vegetation / total frame)
    frame_coverage_pct = round((vegetation_area / float(total_img_pixels)) * 100.0, 2)

    # Protection against division by zero if vegetation_area == 0
    if vegetation_area == 0:
        healthy_pct = 0.0
        diseased_pct = 0.0
    else:
        healthy_pct = round((healthy_cnt / float(vegetation_area)) * 100.0, 2)
        diseased_pct = round((diseased_cnt / float(vegetation_area)) * 100.0, 2)

    return {
        "total_image_pixels": total_img_pixels,
        "vegetation_pixels": vegetation_area,
        "healthy_pixels": healthy_cnt,
        "disease_pixels": diseased_cnt,
        "frame_coverage_pct": frame_coverage_pct,
        "healthy_percentage": healthy_pct,
        "disease_percentage": diseased_pct
    }

def calculate_metrics_from_masks(
    healthy_mask: np.ndarray, 
    diseased_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Utility wrapper to count non-zero pixels directly from binary uint8 masks.
    """
    h_count = int(np.count_nonzero(healthy_mask))
    d_count = int(np.count_nonzero(diseased_mask))
    total_img = healthy_mask.shape[0] * healthy_mask.shape[1]

    return calculate_health_metrics(
        healthy_pixels=h_count,
        diseased_pixels=d_count,
        image_total_pixels=total_img
    )
