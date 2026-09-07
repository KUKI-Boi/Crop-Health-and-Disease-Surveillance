"""
CLI Script: Generate Academic Project Report & CSV Result Files.
Run:
    python generate_report.py --category wheat_stripe_rust
    python generate_report.py --category all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Force UTF-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.analysis.batch_processor import BatchProcessor
from src.analysis.report_generator import export_academic_project_files
from src.analysis.pipeline import CropHealthPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Generate formal 14-section academic submission report and CSV summary."
    )
    parser.add_argument(
        "--data-dir", type=str, default="Dataset", help="Path to dataset directory (default: 'Dataset')"
    )
    parser.add_argument(
        "--category", type=str, default="all", help="Category name (e.g. 'wheat_stripe_rust' or 'all')"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save report files (default: 'results')"
    )

    args = parser.parse_args()

    processor = BatchProcessor(dataset_dir=args.data_dir, output_dir=args.output_dir)

    print(f"\n==========================================================")
    print(f"       ACADEMIC PROJECT REPORT GENERATOR RUNNER           ")
    print(f"==========================================================")

    if args.category.lower() == "all":
        batch_res = processor.process_all_categories()
    else:
        batch_res = processor.process_category(args.category)

    df_res = batch_res["results_df"]

    # Select representative sample (median sample or first sample)
    rep_sample = None
    if not df_res.empty:
        median_idx = len(df_res) // 2
        rep_row = df_res.iloc[median_idx]
        
        # Load sample details
        pipeline = CropHealthPipeline()
        cat_folder = rep_row["disease_type"]
        sample_id = rep_row["image_name"]
        
        img_p = Path(args.data_dir) / cat_folder / f"{sample_id}.jpg"
        lbl_p = Path(args.data_dir) / cat_folder / f"{sample_id}_label.png"

        if lbl_p.exists():
            rep_sample = pipeline.analyze_sample(
                image_input=str(img_p) if img_p.exists() else None,
                label_input=str(lbl_p),
                crop_category=cat_folder,
                image_name=sample_id
            )

    # Export report and CSV
    files = export_academic_project_files(
        batch_summary=batch_res,
        output_dir=args.output_dir,
        representative_sample=rep_sample
    )

    stats = batch_res["summary_stats"]

    print("\nREPORT GENERATION SUMMARY METRICS")
    print("----------------------------------------------------------")
    print(f"  • Category               : {batch_res['category']}")
    print(f"  • Total Images Analyzed  : {stats['total_images_analyzed']}")
    print(f"  • Average Healthy Area   : {100.0 - stats['average_disease_percentage']:.2f}%")
    print(f"  • Average Disease Area   : {stats['average_disease_percentage']}%")
    print("----------------------------------------------------------")
    print(f"  • Markdown Report File   : {files['md_path']}")
    print(f"  • CSV Summary File       : {files['csv_path']}")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
