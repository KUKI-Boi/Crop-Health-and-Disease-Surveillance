"""
Label Analysis Module.
Performs ground-truth segmentation label profiling, configurable class mapping,
colorized segmentation map generation, and 3-panel diagnostic visualization export.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

@dataclass
class LabelClassConfig:
    """
    Configurable class mapping rules for ground-truth label images.
    Supports both 1-channel grayscale values and 3-channel RGB/BGR color tuples.
    """
    background_val: Union[int, Tuple[int, int, int]] = 0
    healthy_val: Union[int, Tuple[int, int, int]] = 127
    disease_val: Union[int, Tuple[int, int, int]] = 255
    
    # Visualization Color Palette (RGB format)
    background_color: Tuple[int, int, int] = (30, 30, 30)      # Dark Charcoal
    healthy_color: Tuple[int, int, int] = (34, 197, 94)        # Vibrant Green
    disease_color: Tuple[int, int, int] = (239, 68, 68)        # Bright Red
    unknown_color: Tuple[int, int, int] = (148, 163, 184)      # Slate Gray

class LabelAnalyzer:
    """
    Analyzes ground-truth segmentation label files (*_label.png), extracts unique
    pixel/color distributions, maps classes safely, and generates colorized diagnostic maps.
    """

    def __init__(self, config: Optional[LabelClassConfig] = None):
        self.config = config if config is not None else LabelClassConfig()

    def analyze_label(
        self, 
        label_img: np.ndarray, 
        custom_config: Optional[LabelClassConfig] = None
    ) -> Dict[str, Any]:
        """
        Inspects label image array, calculates unique value counts, applies configurable mapping,
        and constructs binary class masks and a colorized segmentation visualization map.

        Args:
            label_img (np.ndarray): Label image array (grayscale uint8 or multi-channel BGR).
            custom_config (LabelClassConfig, optional): Override config.

        Returns:
            Dict[str, Any]: Comprehensive analysis results including masks and colorized map.
        """
        cfg = custom_config if custom_config is not None else self.config

        if label_img is None or label_img.size == 0:
            raise ValueError("Invalid or empty label image provided.")

        is_grayscale = len(label_img.shape) == 2 or (len(label_img.shape) == 3 and label_img.shape[2] == 1)
        h, w = label_img.shape[:2]
        total_pixels = h * w

        unique_value_info = []
        background_mask = np.zeros((h, w), dtype=np.uint8)
        healthy_mask = np.zeros((h, w), dtype=np.uint8)
        disease_mask = np.zeros((h, w), dtype=np.uint8)
        colorized_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Default background color initialization
        colorized_map[:, :] = cfg.background_color

        if is_grayscale:
            gray = label_img if len(label_img.shape) == 2 else label_img[:, :, 0]
            unique_vals, counts = np.unique(gray, return_counts=True)
            sorted_idx = np.argsort(-counts)

            for idx in sorted_idx:
                val = int(unique_vals[idx])
                cnt = int(counts[idx])
                pct = round((cnt / total_pixels) * 100.0, 2)
                
                # Class assignment based on configurable rules
                assigned_class = "unknown"
                if val == cfg.background_val:
                    assigned_class = "background"
                    background_mask[gray == val] = 255
                    colorized_map[gray == val] = cfg.background_color
                elif val == cfg.healthy_val:
                    assigned_class = "healthy"
                    healthy_mask[gray == val] = 255
                    colorized_map[gray == val] = cfg.healthy_color
                elif val == cfg.disease_val:
                    assigned_class = "disease"
                    disease_mask[gray == val] = 255
                    colorized_map[gray == val] = cfg.disease_color
                else:
                    colorized_map[gray == val] = cfg.unknown_color

                unique_value_info.append({
                    "value": val,
                    "count": cnt,
                    "percentage": pct,
                    "assigned_class": assigned_class
                })

        else:
            # Multi-channel RGB/BGR Label
            reshaped = label_img.reshape(-1, label_img.shape[2])
            unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
            sorted_idx = np.argsort(-counts)

            for idx in sorted_idx:
                color = tuple(unique_colors[idx].tolist())
                cnt = int(counts[idx])
                pct = round((cnt / total_pixels) * 100.0, 2)

                # Match against BGR / RGB config rules
                match_mask = cv2.inRange(label_img, np.array(color), np.array(color))
                assigned_class = "unknown"

                if color == cfg.background_val:
                    assigned_class = "background"
                    background_mask[match_mask > 0] = 255
                    colorized_map[match_mask > 0] = cfg.background_color
                elif color == cfg.healthy_val:
                    assigned_class = "healthy"
                    healthy_mask[match_mask > 0] = 255
                    colorized_map[match_mask > 0] = cfg.healthy_color
                elif color == cfg.disease_val:
                    assigned_class = "disease"
                    disease_mask[match_mask > 0] = 255
                    colorized_map[match_mask > 0] = cfg.disease_color
                else:
                    colorized_map[match_mask > 0] = cfg.unknown_color

                unique_value_info.append({
                    "value": color,
                    "count": cnt,
                    "percentage": pct,
                    "assigned_class": assigned_class
                })

        return {
            "format": "grayscale" if is_grayscale else "bgr",
            "resolution": (w, h),
            "total_pixels": total_pixels,
            "num_unique_classes": len(unique_value_info),
            "unique_values": unique_value_info,
            "background_mask": background_mask,
            "healthy_mask": healthy_mask,
            "disease_mask": disease_mask,
            "colorized_map": colorized_map,
            "config_used": {
                "background_val": cfg.background_val,
                "healthy_val": cfg.healthy_val,
                "disease_val": cfg.disease_val
            }
        }

    def generate_diagnostic_visualization(
        self,
        original_bgr: Optional[np.ndarray],
        label_img: np.ndarray,
        analysis_result: Dict[str, Any],
        sample_id: str = "sample",
        save_dir: Optional[str] = "results"
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Creates a 3-panel diagnostic visualization:
        1. Original RGB Image
        2. Original Ground-Truth Label
        3. Colorized Segmentation Map with Legend

        Saves figure output to `save_dir/diagnostic_<sample_id>.png`.

        Returns:
            Tuple[plt.Figure, Optional[str]]: Matplotlib Figure object and output path.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Diagnostic Label Analysis: {sample_id}", fontsize=14, fontweight="bold")

        # 1. Original Image Panel
        if original_bgr is not None:
            rgb_orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            axes[0].imshow(rgb_orig)
            axes[0].set_title("Original RGB Image", fontsize=11, fontweight="bold")
        else:
            axes[0].text(0.5, 0.5, "Original Image\nNot Provided", ha="center", va="center")
            axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Original Label Panel
        if len(label_img.shape) == 2 or (len(label_img.shape) == 3 and label_img.shape[2] == 1):
            lbl_gray = label_img if len(label_img.shape) == 2 else label_img[:, :, 0]
            axes[1].imshow(lbl_gray, cmap="gray")
            axes[1].set_title("Raw Ground-Truth Label", fontsize=11, fontweight="bold")
        else:
            lbl_rgb = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
            axes[1].imshow(lbl_rgb)
            axes[1].set_title("Raw Ground-Truth Label (RGB)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 3. Colorized Segmentation Panel
        colorized = analysis_result["colorized_map"]
        axes[2].imshow(colorized)
        axes[2].set_title("Colorized Segmentation Map", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Build Legend Handles
        legend_patches = [
            mpatches.Patch(color=np.array(self.config.background_color)/255.0, label="Background (0)"),
            mpatches.Patch(color=np.array(self.config.healthy_color)/255.0, label="Healthy Leaf (127)"),
            mpatches.Patch(color=np.array(self.config.disease_color)/255.0, label="Disease Spots (255)")
        ]
        axes[2].legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=9)

        plt.tight_layout()

        # Save result image if save_dir provided
        saved_path = None
        if save_dir:
            out_folder = Path(save_dir)
            out_folder.mkdir(parents=True, exist_ok=True)
            saved_path = str(out_folder / f"diagnostic_{sample_id}.png")
            fig.savefig(saved_path, dpi=150, bbox_inches="tight")

        return fig, saved_path
