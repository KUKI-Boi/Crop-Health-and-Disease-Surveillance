"""
Severity Classification Module.
Classifies crop infection severity based on calculated disease-affected area percentage
and configurable project severity thresholds.

NOTE:
These severity levels are project-defined thresholds for prototype classification,
not universal agricultural standards.
"""

from typing import Dict, Any, Tuple
from src.config import SEVERITY_THRESHOLDS

def classify_severity(
    diseased_area_pct: float, 
    custom_thresholds: Dict[str, Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Determines infection severity level based on project-defined thresholds:
    - 0% to 10%  : Low
    - 10% to 30% : Moderate
    - 30% to 50% : High
    - > 50%      : Severe

    Args:
        diseased_area_pct (float): Disease-affected area percentage (0.0 to 100.0).
        custom_thresholds (Dict, optional): Custom threshold dict. Defaults to SEVERITY_THRESHOLDS.

    Returns:
        Dict[str, Any]: Classification details including level, description, and color code.
    """
    pct = max(0.0, min(100.0, float(diseased_area_pct)))

    severity_level = "Unknown"
    color_code = "#808080" # Gray default
    recommendation = "Inspect image inputs."

    if pct <= 10.0:
        severity_level = "Low"
        color_code = "#2E7D32" # Green / Low
        recommendation = "Low infection level detected. Routine monitoring recommended."
    elif pct <= 30.0:
        severity_level = "Moderate"
        color_code = "#FBC02D" # Amber / Moderate
        recommendation = "Moderate infection level detected. Localized treatment recommended."
    elif pct <= 50.0:
        severity_level = "High"
        color_code = "#F57C00" # Orange / High
        recommendation = "High infection level detected. Targeted intervention required."
    else:
        severity_level = "Severe"
        color_code = "#D32F2F" # Red / Severe
        recommendation = "Severe infection level detected. Immediate agricultural intervention required."

    return {
        "disease_percentage": round(pct, 2),
        "severity_level": severity_level,
        "color_code": color_code,
        "recommendation": recommendation,
        "disclaimer": "Project-defined threshold for prototype classification, not a universal agricultural standard."
    }
