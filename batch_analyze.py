"""
CLI Script: Batch Folder Processing & Dataset Analytics.
Run:
    python batch_analyze.py --category wheat_stripe_rust
    python batch_analyze.py --category all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.analysis.batch_processor import BatchProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Execute batch analysis over disease folders and generate dataset CSV reports."
    )
    parser.add_argument(
        "--data-dir", type=str, default="Dataset", help="Path to dataset directory (default: 'Dataset')"
    )
    parser.add_argument(
        "--category", type=str, default="all", help="Category name (e.g. 'wheat_stripe_rust' or 'all')"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save CSV report (default: 'results')"
    )

    args = parser.parse_args()

    processor = BatchProcessor(dataset_dir=args.data_dir, output_dir=args.output_dir)

    print(f"\n==========================================================")
    print(f"             BATCH DATASET ANALYSIS RUNNER                ")
    print(f"==========================================================")
    print(f"Target Dataset : {args.data_dir}")
    print(f"Category Selection : {args.category}")

    if args.category.lower() == "all":
        res = processor.process_all_categories()
    else:
        res = processor.process_category(args.category)

    stats = res["summary_stats"]

    print("\nBATCH SUMMARY METRICS STATISTICS")
    print("----------------------------------------------------------")
    print(f"  • Total Images Analyzed     : {stats['total_images_analyzed']}")
    print(f"  • Average Disease Area (%)  : {stats['average_disease_percentage']}%")
    print(f"  • Minimum Disease Area (%)  : {stats['minimum_disease_percentage']}%")
    print(f"  • Maximum Disease Area (%)  : {stats['maximum_disease_percentage']}%")
    print("----------------------------------------------------------")
    print(f"  • Low Severity Count        : {stats['num_low_severity']}")
    print(f"  • Moderate Severity Count   : {stats['num_moderate_severity']}")
    print(f"  • High Severity Count       : {stats['num_high_severity']}")
    print(f"  • Severe Severity Count     : {stats['num_severe_severity']}")
    print("----------------------------------------------------------")

    if not res["results_df"].empty:
        print("\nFIRST 10 SAMPLE RESULTS:")
        print(res["results_df"].head(10).to_string(index=False))

    print(f"\n✅ Batch Analysis Complete! CSV saved to: {res['csv_path']}")
    print(f"==========================================================\n")

if __name__ == "__main__":
    main()
