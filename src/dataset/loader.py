"""
Dataset Loader Module.
Provides automatic discovery, image-label pair matching, validation, and summary reporting
for the plant disease dataset across all disease categories.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import cv2

@dataclass
class SamplePair:
    """Dataclass representing a discovered dataset sample pair."""
    sample_id: str
    disease_category: str
    image_path: Optional[str]
    label_path: Optional[str]
    black_bg_path: Optional[str]
    is_valid: bool
    validation_error: Optional[str] = None

class DatasetLoader:
    """
    Scans dataset directories, matches original images with corresponding labels,
    validates file readability, and compiles detailed dataset summary metrics.
    """

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, dataset_dir: str = "Dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.samples: List[SamplePair] = []

    def discover_samples(self) -> List[SamplePair]:
        """
        Discovers all image-label sample pairs in the dataset directory.

        Returns:
            List[SamplePair]: List of discovered and validated sample pairs.
        """
        self.samples = []

        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            return self.samples

        # Discover disease category subdirectories or scan root if files exist
        category_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        
        # If no subdirectories found, treat dataset_dir as a single category directory
        if not category_dirs:
            category_dirs = [self.dataset_dir]

        for cat_dir in category_dirs:
            category_name = cat_dir.name
            self._scan_category_directory(cat_dir, category_name)

        return self.samples

    def _scan_category_directory(self, cat_dir: Path, category_name: str):
        """Scans a single category directory for image-label file pairs."""
        all_files = [f for f in cat_dir.iterdir() if f.is_file()]
        
        # Index files by stem / suffix patterns
        # Standard sample naming: <name>.jpg, <name>_label.png, <name>_black.png
        image_candidates: Dict[str, Path] = {}
        label_candidates: Dict[str, Path] = {}
        black_candidates: Dict[str, Path] = {}

        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_IMAGE_EXTS:
                continue

            filename = file_path.name
            stem = file_path.stem

            if stem.endswith("_label"):
                base_id = stem[:-6]
                label_candidates[base_id] = file_path
            elif stem.endswith("_black"):
                base_id = stem[:-6]
                black_candidates[base_id] = file_path
            else:
                image_candidates[stem] = file_path

        # Gather all unique base sample IDs discovered
        all_ids = set(image_candidates.keys()).union(label_candidates.keys()).union(black_candidates.keys())

        for sample_id in sorted(all_ids):
            img_p = image_candidates.get(sample_id)
            lbl_p = label_candidates.get(sample_id)
            blk_p = black_candidates.get(sample_id)

            is_valid = True
            error_msg = None

            if img_p is None:
                is_valid = False
                error_msg = "Missing original image file (.jpg/.png)"
            elif lbl_p is None:
                is_valid = False
                error_msg = "Missing segmentation label file (*_label.png)"
            else:
                # Validate image and label readability
                is_valid, error_msg = self._validate_files(img_p, lbl_p)

            pair = SamplePair(
                sample_id=sample_id,
                disease_category=category_name,
                image_path=str(img_p) if img_p else None,
                label_path=str(lbl_p) if lbl_p else None,
                black_bg_path=str(blk_p) if blk_p else None,
                is_valid=is_valid,
                validation_error=error_msg
            )
            self.samples.append(pair)

    def _validate_files(self, img_path: Path, label_path: Path) -> tuple[bool, Optional[str]]:
        """Validates that both image and label files exist and can be parsed."""
        if not img_path.exists():
            return False, f"Original image file not found: {img_path.name}"
        if not label_path.exists():
            return False, f"Label file not found: {label_path.name}"

        # Test reading file headers / dimensions using cv2 or PIL
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False, f"Original image corrupt or unreadable: {img_path.name}"
            
            lbl = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if lbl is None:
                return False, f"Label image corrupt or unreadable: {label_path.name}"

        except Exception as err:
            return False, f"Error opening files: {str(err)}"

        return True, None

    def get_summary(self) -> Dict[str, Any]:
        """
        Generates a summary dictionary and DataFrame grouped by disease category.

        Returns:
            Dict[str, Any]: Contains summary table DataFrame, category details, and counts.
        """
        if not self.samples:
            self.discover_samples()

        categories: Dict[str, Dict[str, int]] = {}

        for sample in self.samples:
            cat = sample.disease_category
            if cat not in categories:
                categories[cat] = {
                    "total_original_images": 0,
                    "valid_image_label_pairs": 0,
                    "missing_labels": 0,
                    "missing_original_images": 0,
                    "invalid_or_corrupt": 0
                }

            if sample.image_path is not None:
                categories[cat]["total_original_images"] += 1
            else:
                categories[cat]["missing_original_images"] += 1

            if sample.is_valid:
                categories[cat]["valid_image_label_pairs"] += 1
            else:
                categories[cat]["invalid_or_corrupt"] += 1
                if sample.image_path is not None and sample.label_path is None:
                    categories[cat]["missing_labels"] += 1

        summary_rows = []
        for cat, stats in categories.items():
            summary_rows.append({
                "Disease Category": cat,
                "Original Images": stats["total_original_images"],
                "Valid Pairs": stats["valid_image_label_pairs"],
                "Missing Labels": stats["missing_labels"],
                "Corrupt/Invalid Pairs": stats["invalid_or_corrupt"]
            })

        df_summary = pd.DataFrame(summary_rows)

        total_valid = sum(c["valid_image_label_pairs"] for c in categories.values())
        total_orig = sum(c["total_original_images"] for c in categories.values())
        total_missing_labels = sum(c["missing_labels"] for c in categories.values())

        return {
            "dataset_dir": str(self.dataset_dir),
            "directory_exists": self.dataset_dir.exists(),
            "total_samples": len(self.samples),
            "total_original_images": total_orig,
            "total_valid_pairs": total_valid,
            "total_missing_labels": total_missing_labels,
            "category_stats": categories,
            "summary_df": df_summary
        }

    def print_summary(self) -> str:
        """Prints formatted summary report to console and returns string."""
        summary = self.get_summary()

        out_lines = []
        out_lines.append("==========================================================")
        out_lines.append("           PLANT DISEASE DATASET DISCOVERY REPORT         ")
        out_lines.append("==========================================================")
        out_lines.append(f"Target Directory : {summary['dataset_dir']}")
        out_lines.append(f"Directory Exists : {'Yes' if summary['directory_exists'] else 'No (Path Not Found)'}")
        out_lines.append(f"Total Discovered : {summary['total_samples']} sample entries")
        out_lines.append(f"Valid Pairs      : {summary['total_valid_pairs']}")
        out_lines.append(f"Missing Labels   : {summary['total_missing_labels']}")
        out_lines.append("----------------------------------------------------------")

        if not summary['summary_df'].empty:
            out_lines.append(summary['summary_df'].to_string(index=False))
        else:
            out_lines.append("No dataset samples found in target directory.")

        out_lines.append("==========================================================")
        
        report_str = "\n".join(out_lines)
        print(report_str)
        return report_str
