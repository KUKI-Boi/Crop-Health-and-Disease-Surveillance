"""
CLI Script: Analyze Ground-Truth Segmentation Label & Generate Academic Visualizations.
Run:
    python analyze_label.py --sample-id Wheat_C230514_0004
    python analyze_label.py --label Dataset/wheat_stripe_rust/Wheat_C230514_0004_label.png
"""

import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.image_processing.label_analysis import LabelClassConfig
from src.analysis.pipeline import CropHealthPipeline
from src.visualization.display import generate_full_presentation_figure
from src.dataset.loader import DatasetLoader

def main():
    parser = argparse.ArgumentParser(
        description="Inspect ground-truth label images and generate publication-ready academic figures."
    )
    parser.add_argument(
        "--data-dir", type=str, default="Dataset", help="Path to dataset directory (default: 'Dataset')"
    )
    parser.add_argument(
        "--sample-id", type=str, default=None, help="Sample ID stem (e.g. Wheat_C230514_0004)"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Explicit path to original image (.jpg)"
    )
    parser.add_argument(
        "--label", type=str, default=None, help="Explicit path to label image (*_label.png)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="results", help="Directory to save figures (default: 'results')"
    )
    parser.add_argument(
        "--background-val", type=int, default=0, help="Pixel value mapped to background (default: 0)"
    )
    parser.add_argument(
        "--healthy-val", type=int, default=127, help="Pixel value mapped to healthy leaf (default: 127)"
    )
    parser.add_argument(
        "--disease-val", type=int, default=255, help="Pixel value mapped to disease spots (default: 255)"
    )

    args = parser.parse_args()

    img_path = None
    lbl_path = None
    sample_id = "sample"
    category_name = "wheat_stripe_rust"

    if args.label:
        lbl_path = Path(args.label)
        sample_id = lbl_path.stem.replace("_label", "")
        if args.image:
            img_path = Path(args.image)
        else:
            potential_jpg = lbl_path.parent / f"{sample_id}.jpg"
            if potential_jpg.exists():
                img_path = potential_jpg
        category_name = lbl_path.parent.name
    elif args.sample_id:
        loader = DatasetLoader(args.data_dir)
        samples = loader.discover_samples()
        matching = [s for s in samples if s.sample_id == args.sample_id]
        if matching:
            pair = matching[0]
            img_path = Path(pair.image_path) if pair.image_path else None
            lbl_path = Path(pair.label_path) if pair.label_path else None
            sample_id = pair.sample_id
            category_name = pair.disease_category
        else:
            print(f"Error: Sample ID '{args.sample_id}' not found in dataset '{args.data_dir}'.")
            return
    else:
        loader = DatasetLoader(args.data_dir)
        samples = loader.discover_samples()
        valid = [s for s in samples if s.is_valid and s.label_path]
        if valid:
            pair = valid[0]
            img_path = Path(pair.image_path) if pair.image_path else None
            lbl_path = Path(pair.label_path) if pair.label_path else None
            sample_id = pair.sample_id
            category_name = pair.disease_category
            print(f"No specific sample requested. Selected sample: {sample_id} ({category_name})")
        else:
            print("Error: No valid image-label pairs found in dataset.")
            return

    if not lbl_path or not lbl_path.exists():
        print(f"Error: Label image file not found: {lbl_path}")
        return

    print(f"\n==========================================================")
    print(f"       CROP HEALTH SURVEILLANCE PIPELINE: {sample_id}      ")
    print(f"==========================================================")
    print(f"Category   : {category_name}")
    print(f"Label Path : {lbl_path}")
    print(f"Image Path : {img_path if img_path else 'None'}")

    label_config = LabelClassConfig(
        background_val=args.background_val,
        healthy_val=args.healthy_val,
        disease_val=args.disease_val
    )
    pipeline = CropHealthPipeline(label_config=label_config)

    # Execute full analysis pipeline
    analysis_res = pipeline.analyze_sample(
        image_input=str(img_path) if img_path else None,
        label_input=str(lbl_path),
        crop_category=category_name,
        image_name=sample_id
    )

    print("\nUNIQUE PIXEL DISTRIBUTION & CLASS MAPPING")
    print("----------------------------------------------------------")
    df_vals = pd.DataFrame(analysis_res["label_analysis_details"]["unique_values"])
    print(df_vals.to_string(index=False))
    print("----------------------------------------------------------")

    print("\nQUANTITATIVE METRICS & SEVERITY CLASSIFICATION")
    print(f"  • Total Frame Pixels : {analysis_res['total_image_pixels']:,}")
    print(f"  • Vegetation Area    : {analysis_res['vegetation_pixels']:,} pixels")
    print(f"  • Healthy Leaf Area  : {analysis_res['healthy_percentage']:.2f}% ({analysis_res['healthy_pixels']:,} pixels)")
    print(f"  • Disease Area (%)   : {analysis_res['disease_percentage']:.2f}% ({analysis_res['disease_pixels']:,} pixels)")
    print(f"  • Infection Severity : {analysis_res['severity_level']}")
    print("----------------------------------------------------------")

    # Generate professional 6-panel presentation grid figure
    fig, saved_path = generate_full_presentation_figure(
        analysis_result=analysis_res,
        save_dir=args.save_dir
    )

    print(f"\n✅ Professional Academic Presentation Figure Generated!")
    print(f"   Output file saved to: {saved_path}")
    print(f"==========================================================\n")

if __name__ == "__main__":
    main()
