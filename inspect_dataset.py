"""
CLI Script: Inspect Dataset Summary & Label Pixel Values.
Run: python inspect_dataset.py [--data-dir Dataset] [--max-samples 5]
"""

import sys
import argparse
from pathlib import Path
from src.dataset.loader import DatasetLoader
from src.dataset.inspector import DatasetInspector

# Force UTF-8 encoding for stdout on Windows console if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(
        description="Inspect Plant Disease Dataset & Label Pixel Distributions."
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="Dataset", 
        help="Path to dataset directory (default: 'Dataset')"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=5, 
        help="Maximum label images to inspect per category (default: 5)"
    )

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    print("\nScanning Dataset at:", data_path.resolve())
    loader = DatasetLoader(str(data_path))
    samples = loader.discover_samples()

    # 1. Print summary table
    loader.print_summary()

    # 2. If samples exist, run label inspection
    valid_samples = [s for s in samples if s.is_valid and s.label_path]

    if not valid_samples:
        if not data_path.exists():
            print("\nNOTICE: Target dataset directory was not found.")
            print("   To populate the dataset:")
            print("   1. Create folder: 'Dataset/'")
            print("   2. Add category folders: 'wheat_stripe_rust', 'soybean_bacterial_blight', 'cedar_apple_rust'")
            print("   3. Place original .jpg images and corresponding *_label.png files inside each folder.")
        else:
            print("\nWARNING: No valid image-label pairs found in the dataset directory.")
        return

    print("\nRUNNING DYNAMIC LABEL INSPECTION (NO HARDCODED PIXEL ASSUMPTIONS)")
    print("----------------------------------------------------------")
    inspector = DatasetInspector()

    # Group by category
    by_category = {}
    for s in valid_samples:
        by_category.setdefault(s.disease_category, []).append(s)

    for cat_name, cat_samples in by_category.items():
        print(f"\nCategory: {cat_name} ({len(cat_samples)} valid pairs)")
        inspection = inspector.batch_inspect_category(cat_samples, max_samples=args.max_samples)
        
        print(f"   - Label Format   : {inspection['format'].upper()}")
        print(f"   - Inspected      : {inspection['num_samples_inspected']} label files")
        print(f"   - Unique Pixels  : {inspection['global_unique_pixel_values']}")
        
        # Sample detail
        if inspection['sample_inspections']:
            first_sample = inspection['sample_inspections'][0]
            print(f"   - Sample Mapping : {first_sample['suggested_mapping']}")

    print("\nDataset Inspection Complete.\n")

if __name__ == "__main__":
    main()
