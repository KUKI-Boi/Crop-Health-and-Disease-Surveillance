"""
Segmentation Module.
Performs classical computer vision vegetation segmentation and disease region extraction
using HSV color thresholding, Otsu thresholding, and morphological operations.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2
from src.config import DEFAULT_HSV_GREEN_LOWER, DEFAULT_HSV_GREEN_UPPER

class VegetationSegmenter:
    """
    Classical image processing pipeline for segmenting leaf vegetation and diseased spots.
    """

    def __init__(self, hsv_lower: Tuple[int, int, int] = DEFAULT_HSV_GREEN_LOWER,
                 hsv_upper: Tuple[int, int, int] = DEFAULT_HSV_GREEN_UPPER):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

    def extract_vegetation_mask(self, hsv_img: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask isolating leaf vegetation using HSV color range.

        Args:
            hsv_img (np.ndarray): Image in HSV color space.

        Returns:
            np.ndarray: Binary mask (255 for vegetation, 0 for background).
        """
        mask = cv2.inRange(hsv_img, self.hsv_lower, self.hsv_upper)
        
        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def segment_healthy_vs_diseased(
        self, 
        bgr_img: np.ndarray, 
        vegetation_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Sub-segments leaf region into healthy green tissue vs diseased spots/lesions.

        Args:
            bgr_img (np.ndarray): BGR original image.
            vegetation_mask (np.ndarray): Binary mask of total leaf region.

        Returns:
            Dict[str, np.ndarray]: Masks for healthy_mask, diseased_mask, and masked images.
        """
        # Convert to Lab color space for color deviation analysis
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # In Lab space, green foliage has lower 'a' channel values.
        # Diseased/yellow/brown spots shift towards higher 'a' and 'b' channel values.
        
        # Apply Otsu thresholding on 'a' channel restricted to leaf mask
        a_masked = cv2.bitwise_and(a_channel, a_channel, mask=vegetation_mask)
        _, diseased_thresh = cv2.threshold(
            a_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        diseased_mask = cv2.bitwise_and(diseased_thresh, vegetation_mask)
        healthy_mask = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(diseased_mask))

        return {
            "vegetation_mask": vegetation_mask,
            "healthy_mask": healthy_mask,
            "diseased_mask": diseased_mask
        }
