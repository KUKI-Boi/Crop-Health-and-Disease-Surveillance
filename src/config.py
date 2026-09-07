"""
Configuration parameters for the Crop Health and Disease Surveillance Project.
Contains severity thresholds, crop metadata, color space defaults, and report settings.
"""

from typing import Dict, Any, Tuple

# Supported dataset categories and names
SUPPORTED_CROPS: Dict[str, Dict[str, str]] = {
    "wheat_stripe_rust": {
        "name": "Wheat - Stripe Rust (Puccinia striiformis)",
        "crop": "Wheat",
        "disease": "Stripe Rust",
        "description": "Fungal infection causing yellow-orange pustules aligned in stripes along leaf veins."
    },
    "soybean_bacterial_blight": {
        "name": "Soybean - Bacterial Blight (Pseudomonas syringae)",
        "crop": "Soybean",
        "disease": "Bacterial Blight",
        "description": "Bacterial infection causing angular brown lesions surrounded by yellow halos."
    },
    "cedar_apple_rust": {
        "name": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "Fungal disease creating bright yellow-orange spots on leaves."
    }
}

# Infection severity levels based on percentage of disease-affected area relative to vegetation area.
# NOTE: These are project-defined thresholds for prototype classification, not universal agricultural standards.
# Ranges:
# 0%  - 10% = Low
# 10% - 30% = Moderate
# 30% - 50% = High
# > 50%     = Severe
SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Low": (0.0, 10.0),
    "Moderate": (10.0, 30.0),
    "High": (30.0, 50.0),
    "Severe": (50.0, 100.0)
}

# Default Color Space Parameters for Classical Image Processing
DEFAULT_HSV_GREEN_LOWER = (25, 40, 40)
DEFAULT_HSV_GREEN_UPPER = (85, 255, 255)

# Report Generation Metadata Defaults
REPORT_CONFIG: Dict[str, Any] = {
    "institution_name": "Academic Research & Engineering Project",
    "project_title": "Crop Health and Disease Surveillance",
    "pipeline_type": "Classical Image Processing & Ground-Truth Segmentation Analysis",
    "deployment_note": "Prototype pipeline designed for integration with drone-acquired aerial imagery.",
    "severity_disclaimer": "Severity levels are project-defined thresholds for prototype classification, not universal agricultural standards."
}
