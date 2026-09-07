"""
Label Inspector Module.
Provides safe dynamic analysis of ground-truth label images without hardcoding pixel assumptions.
Determines background, healthy leaf, and disease spot class values by inspecting actual image arrays.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

class LabelInspector:
    """
    Safely inspects label images (*_label.png) from the dataset to discover
    unique pixel intensities or color values and map them dynamically.
    """

    def __init__(self):
        pass

    def inspect_label_image(self, label_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes a ground-truth label image to identify unique pixel values/colors.

        Args:
            label_img (np.ndarray): Loaded label image array (grayscale or BGR).

        Returns:
            Dict[str, Any]: Dictionary containing unique values, counts, and suggested class mapping.
        """
        if label_img is None or label_img.size == 0:
            raise ValueError("Invalid or empty label image provided.")

        is_grayscale = len(label_img.shape) == 2 or (len(label_img.shape) == 3 and label_img.shape[2] == 1)

        if is_grayscale:
            gray = label_img if len(label_img.shape) == 2 else label_img[:, :, 0]
            unique_vals, counts = np.unique(gray, return_counts=True)
            sorted_indices = np.argsort(-counts) # Most frequent first (usually background)
            
            value_info = []
            for idx in sorted_indices:
                val = int(unique_vals[idx])
                cnt = int(counts[idx])
                pct = (cnt / label_img.size) * 100.0
                value_info.append({"pixel_value": val, "pixel_count": cnt, "percentage": round(pct, 2)})

            mapping = self._infer_grayscale_mapping(value_info)

            return {
                "format": "grayscale",
                "num_classes": len(unique_vals),
                "unique_values": value_info,
                "suggested_mapping": mapping
            }
        else:
            # Color label image
            reshaped = label_img.reshape(-1, label_img.shape[2])
            unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
            sorted_indices = np.argsort(-counts)

            color_info = []
            for idx in sorted_indices:
                color = unique_colors[idx].tolist()
                cnt = int(counts[idx])
                pct = (cnt / (label_img.shape[0] * label_img.shape[1])) * 100.0
                color_info.append({"color_bgr": color, "pixel_count": cnt, "percentage": round(pct, 2)})

            mapping = self._infer_color_mapping(color_info)

            return {
                "format": "bgr",
                "num_classes": len(unique_colors),
                "unique_values": color_info,
                "suggested_mapping": mapping
            }

    def _infer_grayscale_mapping(self, value_info: list) -> Dict[str, Optional[int]]:
        """
        Heuristic mapping for 1-channel label images:
        - Most frequent pixel value is typically Background (e.g. 0 or 255)
        - Mid-range / secondary pixel value is Healthy Leaf
        - Minority pixel value is Disease Spot
        """
        mapping: Dict[str, Optional[int]] = {
            "background": None,
            "healthy_leaf": None,
            "disease_spot": None
        }

        if not value_info:
            return mapping

        # Sort by pixel frequency descending
        by_freq = sorted(value_info, key=lambda x: x["pixel_count"], reverse=True)
        
        # Most frequent is background
        mapping["background"] = by_freq[0]["pixel_value"]

        remaining = by_freq[1:]
        if len(remaining) == 1:
            # Only 2 classes present in label
            mapping["healthy_leaf"] = remaining[0]["pixel_value"]
        elif len(remaining) >= 2:
            # Sort remaining by pixel value intensity
            by_val = sorted(remaining, key=lambda x: x["pixel_value"])
            mapping["healthy_leaf"] = by_val[0]["pixel_value"]
            mapping["disease_spot"] = by_val[-1]["pixel_value"]

        return mapping

    def _infer_color_mapping(self, color_info: list) -> Dict[str, Optional[list]]:
        """
        Heuristic mapping for multi-channel RGB/BGR label images.
        """
        mapping: Dict[str, Optional[list]] = {
            "background": None,
            "healthy_leaf": None,
            "disease_spot": None
        }

        if not color_info:
            return mapping

        by_freq = sorted(color_info, key=lambda x: x["pixel_count"], reverse=True)
        mapping["background"] = by_freq[0]["color_bgr"]

        remaining = by_freq[1:]
        if len(remaining) == 1:
            mapping["healthy_leaf"] = remaining[0]["color_bgr"]
        elif len(remaining) >= 2:
            mapping["healthy_leaf"] = remaining[0]["color_bgr"]
            mapping["disease_spot"] = remaining[1]["color_bgr"]

        return mapping
