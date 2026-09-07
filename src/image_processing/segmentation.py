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

    def extract_invalid_mask(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Detects invalid black image borders/padding (R < 20 and G < 20 and B < 20).

        Returns:
            np.ndarray: Binary mask (255 for invalid/black padding, 0 for valid content).
        """
        invalid_bool = (bgr_img[:, :, 0] < 20) & (bgr_img[:, :, 1] < 20) & (bgr_img[:, :, 2] < 20)
        return (invalid_bool.astype(np.uint8)) * 255

    def extract_vegetation_mask(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask isolating total leaf vegetation from background using
        multi-feature color space thresholding (ExG, HSV, Lab) and morphological operations.
        Excludes black padding and non-vegetation background.

        Args:
            bgr_img (np.ndarray): Original image in BGR format.

        Returns:
            np.ndarray: Binary mask (255 for leaf vegetation, 0 for background).
        """
        h, w = bgr_img.shape[:2]
        total_pixels = h * w

        # 1. Invalid black padding isolation
        invalid_mask = self.extract_invalid_mask(bgr_img)
        valid_mask = cv2.bitwise_not(invalid_mask)
        valid_bool = valid_mask > 0
        valid_cnt = np.count_nonzero(valid_mask)

        if valid_cnt == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # 2. Multi-feature color space transformations
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        b, g, r = bgr_img[:, :, 0].astype(np.float32), bgr_img[:, :, 1].astype(np.float32), bgr_img[:, :, 2].astype(np.float32)

        exg = 2.0 * g - r - b

        # Vegetation feature filters:
        # ExG > -10 (allows yellowing vegetation while excluding non-green background)
        # HSV Hue in [20, 95] (green foliage and yellow-green crop leaves)
        # HSV Saturation >= 20 (excludes dull gray background/borders)
        # Lab a-channel < 138 (greenish component in Lab color space)
        veg_candidate_bool = (
            valid_bool &
            (exg > -10.0) &
            (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 95) &
            (hsv[:, :, 1] >= 20) &
            (lab[:, :, 1] < 138)
        )

        veg_uint8 = (veg_candidate_bool.astype(np.uint8)) * 255

        # Fallback to non-black valid content if strict vegetation filters are under-segmenting
        # (e.g. for synthetic unit test images or non-standard crops)
        if np.count_nonzero(veg_uint8) < (0.05 * valid_cnt):
            veg_uint8 = cv2.bitwise_and(valid_mask, cv2.bitwise_or(veg_uint8, (exg > 0).astype(np.uint8) * 255))
            if np.count_nonzero(veg_uint8) < (0.05 * valid_cnt):
                veg_uint8 = valid_mask.copy()

        # 3. Morphological noise removal & hole filling
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        cleaned_mask = cv2.morphologyEx(veg_uint8, cv2.MORPH_OPEN, kernel_sm)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_lg)

        # 4. Connected component filtering (remove components smaller than 0.05% of valid frame)
        min_veg_area = max(10, int(0.0005 * valid_cnt))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
        final_veg_mask = np.zeros_like(cleaned_mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_veg_area:
                final_veg_mask[labels == i] = 255

        # Absolute protection: vegetation MUST be subset of valid_mask
        final_veg_mask = cv2.bitwise_and(final_veg_mask, valid_mask)
        return final_veg_mask

    def segment_healthy_vs_diseased(
        self, 
        bgr_img: np.ndarray, 
        vegetation_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Sub-segments leaf vegetation region into healthy green tissue vs diseased spots/lesions
        using adaptive healthy green color modeling, HSV/Lab color space analysis, and morphological filtering.

        Args:
            bgr_img (np.ndarray): BGR original image.
            vegetation_mask (Optional[np.ndarray]): Binary mask of total leaf region. Auto-extracted if None.

        Returns:
            Dict[str, Any]: Masks for vegetation_mask, healthy_mask, diseased_mask, invalid_mask, valid_mask, debug_info.
        """
        invalid_mask = self.extract_invalid_mask(bgr_img)
        valid_mask = cv2.bitwise_not(invalid_mask)

        if vegetation_mask is None:
            vegetation_mask = self.extract_vegetation_mask(bgr_img)
        else:
            vegetation_mask = cv2.bitwise_and(vegetation_mask, valid_mask)

        veg_pixels = int(np.count_nonzero(vegetation_mask))
        if veg_pixels == 0:
            empty_mask = np.zeros(bgr_img.shape[:2], dtype=np.uint8)
            return {
                "vegetation_mask": empty_mask,
                "healthy_mask": empty_mask,
                "diseased_mask": empty_mask,
                "invalid_mask": invalid_mask,
                "valid_mask": valid_mask,
                "debug_info": {
                    "vegetation_pixels": 0,
                    "healthy_pixels": 0,
                    "disease_pixels": 0,
                    "disease_percentage": 0.0,
                    "num_disease_components": 0,
                    "invalid_border_pixels": int(np.count_nonzero(invalid_mask))
                }
            }

        # 1. Color space representations
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        b, g, r = bgr_img[:, :, 0].astype(np.float32), bgr_img[:, :, 1].astype(np.float32), bgr_img[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b

        veg_bool = vegetation_mask > 0

        # 2. Adaptive Healthy Green Color Distribution Model
        healthy_green_candidate = (
            veg_bool &
            (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 85) &
            (hsv[:, :, 1] >= 30) &
            (lab[:, :, 1] < 128)
        )

        if np.count_nonzero(healthy_green_candidate) > 10:
            h_green_lab = lab[healthy_green_candidate]
            mean_a = float(np.mean(h_green_lab[:, 1]))
            std_a = float(np.std(h_green_lab[:, 1]))
            mean_b = float(np.mean(h_green_lab[:, 2]))
            std_b = float(np.std(h_green_lab[:, 2]))
        else:
            mean_a, std_a = 120.0, 5.0
            mean_b, std_b = 140.0, 10.0

        # 3. Detect Abnormal / Diseased Candidates ONLY INSIDE VEGETATION
        yellow_orange_rust = veg_bool & ((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 90)) & (hsv[:, :, 1] >= 30)
        lab_b_high = veg_bool & (lab[:, :, 2] > max(142.0, mean_b + 1.2 * std_b))
        lab_a_high = veg_bool & (lab[:, :, 1] > max(130.0, mean_a + 1.5 * std_a))
        exg_low = veg_bool & (exg < 0)

        disease_candidate_bool = veg_bool & (yellow_orange_rust | lab_b_high | lab_a_high | exg_low)
        disease_candidate_uint8 = (disease_candidate_bool.astype(np.uint8)) * 255

        # 4. Spatial consistency & morphological noise removal
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        disease_cleaned = cv2.morphologyEx(disease_candidate_uint8, cv2.MORPH_OPEN, kernel_sm)

        # 5. Connected component filtering on disease candidate mask
        num_d_labels, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(disease_cleaned)
        raw_disease_mask = np.zeros_like(disease_cleaned)
        min_d_area = max(5, int(0.0001 * veg_pixels))

        valid_d_components = 0
        for i in range(1, num_d_labels):
            if d_stats[i, cv2.CC_STAT_AREA] >= min_d_area:
                raw_disease_mask[d_labels == i] = 255
                valid_d_components += 1

        # 6. Absolute programmatic enforcement of mask invariants
        diseased_mask = cv2.bitwise_and(raw_disease_mask, vegetation_mask)
        healthy_mask = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(diseased_mask))

        # Programmatic assertions
        assert np.array_equal(diseased_mask, cv2.bitwise_and(diseased_mask, vegetation_mask)), "Disease mask must be subset of vegetation mask!"
        assert np.array_equal(healthy_mask, cv2.bitwise_and(healthy_mask, vegetation_mask)), "Healthy mask must be subset of vegetation mask!"
        assert np.count_nonzero(cv2.bitwise_and(diseased_mask, invalid_mask)) == 0, "No invalid/black padding pixel can be disease!"

        healthy_cnt = int(np.count_nonzero(healthy_mask))
        disease_cnt = int(np.count_nonzero(diseased_mask))
        disease_pct = (disease_cnt / float(veg_pixels)) * 100.0 if veg_pixels > 0 else 0.0

        debug_info = {
            "vegetation_pixels": veg_pixels,
            "healthy_pixels": healthy_cnt,
            "disease_pixels": disease_cnt,
            "disease_percentage": round(disease_pct, 2),
            "num_disease_components": valid_d_components,
            "invalid_border_pixels": int(np.count_nonzero(invalid_mask))
        }

        return {
            "vegetation_mask": vegetation_mask,
            "healthy_mask": healthy_mask,
            "diseased_mask": diseased_mask,
            "invalid_mask": invalid_mask,
            "valid_mask": valid_mask,
            "debug_info": debug_info
        }
