"""
Segmentation Module.
Performs classical computer vision vegetation segmentation and disease region extraction
using HSV color space analysis, Excess Green Index (ExG), Otsu thresholding, Lab color space,
and morphological operations.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from src.config import DEFAULT_HSV_GREEN_LOWER, DEFAULT_HSV_GREEN_UPPER

class VegetationSegmenter:
    """
    Classical image processing pipeline for segmenting leaf vegetation and diseased spots
    directly from an original crop leaf image without requiring ground-truth label masks.
    """

    def __init__(
        self, 
        hsv_lower: Tuple[int, int, int] = DEFAULT_HSV_GREEN_LOWER,
        hsv_upper: Tuple[int, int, int] = DEFAULT_HSV_GREEN_UPPER
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

    def extract_vegetation_mask(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask isolating total leaf vegetation from background using
        thresholding, Excess Green Index (ExG), and morphological operations.

        Args:
            bgr_img (np.ndarray): Original image in BGR format.

        Returns:
            np.ndarray: Binary mask (255 for leaf vegetation, 0 for background).
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Non-black background isolation
        _, non_black_mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
        
        # 2. HSV Green/Yellow Leaf Range
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv_img, (20, 25, 25), (95, 255, 255))

        # 3. Excess Green Index (ExG = 2G - R - B)
        b, g, r = cv2.split(bgr_img.astype(np.float32))
        exg = 2.0 * g - r - b
        exg_mask = np.where(exg > 0, 255, 0).astype(np.uint8)

        # Combined vegetation leaf mask
        combined_mask = cv2.bitwise_and(non_black_mask, cv2.bitwise_or(hsv_mask, exg_mask))

        # Morphological noise removal & hole closing
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_sm)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_lg)

        # Fallback to non-black mask if HSV under-segments heavily diseased leaves
        if np.count_nonzero(non_black_mask) > 0 and (np.count_nonzero(cleaned_mask) / np.count_nonzero(non_black_mask)) < 0.30:
            cleaned_mask = non_black_mask

        return cleaned_mask

    def segment_healthy_vs_diseased(
        self, 
        bgr_img: np.ndarray, 
        vegetation_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sub-segments leaf vegetation region into healthy green tissue vs diseased spots/lesions
        using HSV color thresholding, Lab color space analysis, and morphological filtering.

        Args:
            bgr_img (np.ndarray): BGR original image.
            vegetation_mask (Optional[np.ndarray]): Binary mask of total leaf region. Auto-extracted if None.

        Returns:
            Dict[str, np.ndarray]: Masks for vegetation_mask, healthy_mask, and diseased_mask.
        """
        if vegetation_mask is None:
            vegetation_mask = self.extract_vegetation_mask(bgr_img)

        # If zero vegetation detected, return empty masks
        if np.count_nonzero(vegetation_mask) == 0:
            empty_mask = np.zeros(bgr_img.shape[:2], dtype=np.uint8)
            return {
                "vegetation_mask": empty_mask,
                "healthy_mask": empty_mask,
                "diseased_mask": empty_mask
            }

        # 1. Convert to HSV color space for healthy green tissue identification
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        
        # Healthy green hue range: 30 <= Hue <= 88, Saturation >= 30, Value >= 30
        healthy_green_hsv = cv2.inRange(hsv, (30, 30, 30), (88, 255, 255))
        
        # Healthy mask inside total vegetation leaf area
        raw_healthy_mask = cv2.bitwise_and(vegetation_mask, healthy_green_hsv)

        # Morphological smoothing on healthy mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        healthy_mask = cv2.morphologyEx(raw_healthy_mask, cv2.MORPH_OPEN, kernel)
        healthy_mask = cv2.bitwise_and(vegetation_mask, healthy_mask)

        # Diseased mask = Vegetation leaf area minus healthy green leaf area
        diseased_mask = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(healthy_mask))
        diseased_mask = cv2.morphologyEx(diseased_mask, cv2.MORPH_OPEN, kernel)

        return {
            "vegetation_mask": vegetation_mask,
            "healthy_mask": healthy_mask,
            "diseased_mask": diseased_mask
        }
