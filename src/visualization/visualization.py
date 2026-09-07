"""
Professional Academic Visualization Module.
Generates publication-ready figures, multi-panel comparison grids, colorized health maps,
and quantitative area charts suitable for academic reports, presentations (PPT), and viva voce.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# AgriVision Palette Color Conventions (RGB format)
HEALTHY_GREEN = (46, 139, 87)       # Healthy Green #2E8B57
DISEASE_RED = (217, 75, 69)        # Rust Red #D94B45
BACKGROUND_DARK = (23, 35, 29)     # Dark Charcoal #17231D

# Severity Badges & Colors (AgriVision Palette)
SEVERITY_COLORS = {
    "LOW": "#2E8B57",        # Healthy Green
    "MODERATE": "#D99A32",   # Warning Amber
    "HIGH": "#D96B32",       # Orange-Rust
    "SEVERE": "#D94B45"      # Rust Red
}

def create_colorized_health_map(
    healthy_mask: np.ndarray, 
    disease_mask: np.ndarray
) -> np.ndarray:
    """
    Constructs a colorized health map using standard color conventions:
    - Healthy vegetation = Green
    - Disease/stressed region = Red
    - Background = Black/Dark Charcoal

    Args:
        healthy_mask (np.ndarray): Binary uint8 mask for healthy foliage.
        disease_mask (np.ndarray): Binary uint8 mask for diseased spots.

    Returns:
        np.ndarray: RGB image array of the colorized health map.
    """
    h, w = healthy_mask.shape[:2]
    health_map = np.zeros((h, w, 3), dtype=np.uint8)
    health_map[:, :] = BACKGROUND_DARK

    health_map[healthy_mask > 0] = HEALTHY_GREEN
    health_map[disease_mask > 0] = DISEASE_RED

    return health_map

def create_color_coded_overlay(
    bgr_img: np.ndarray, 
    healthy_mask: np.ndarray, 
    diseased_mask: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlays colored highlights on original image:
    - Green (0, 255, 0) for healthy leaf regions.
    - Red (0, 0, 255) for diseased spots.

    Args:
        bgr_img (np.ndarray): Original BGR image.
        healthy_mask (np.ndarray): Binary mask for healthy area.
        diseased_mask (np.ndarray): Binary mask for diseased area.
        alpha (float): Transparency blending factor.

    Returns:
        np.ndarray: Blended RGB image with color overlays.
    """
    overlay = bgr_img.copy()
    overlay[healthy_mask > 0] = [0, 255, 0] # BGR
    overlay[diseased_mask > 0] = [0, 0, 255] # BGR

    blended = cv2.addWeighted(overlay, alpha, bgr_img, 1 - alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

def plot_segmentation_pipeline(
    rgb_img: np.ndarray,
    veg_mask: np.ndarray,
    healthy_mask: np.ndarray,
    diseased_mask: np.ndarray,
    overlay_img: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Creates a clean 4-panel Matplotlib figure showing processing steps.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(veg_mask, cmap='gray')
    axes[1].set_title("Vegetation Mask", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(healthy_mask, cmap='Greens')
    axes[2].set_title("Healthy Region", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(diseased_mask, cmap='Reds')
    axes[3].set_title("Diseased Spots", fontsize=11, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    return fig

def create_summary_chart(
    healthy_pct: float, 
    disease_pct: float,
    figsize: Tuple[int, int] = (6, 4.5)
) -> plt.Figure:
    """
    Generates a clean, highly readable Donut chart summarizing Healthy Area % vs Disease Area %.
    Handles small slice percentages gracefully to prevent text overlap.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")

    labels = [f"Healthy ({healthy_pct:.1f}%)", f"Diseased ({disease_pct:.1f}%)"]
    sizes = [max(0.01, healthy_pct), max(0.01, disease_pct)]
    colors = ["#2E8B57", "#D94B45"]

    # Draw Donut Chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 6.0 else "",
        pctdistance=0.75,
        labeldistance=1.15,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.38, edgecolor="#FFFFFF", linewidth=2.5),
        textprops=dict(fontsize=10, fontweight="bold", color="#17231D")
    )

    # Style inside percentage texts
    for autotext in autotexts:
        autotext.set_color("#FFFFFF")
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    # Center Text inside Donut Hole
    ax.text(
        0, 0, 
        f"{disease_pct:.1f}%\nDISEASED", 
        ha="center", va="center", 
        fontsize=13, fontweight="bold", 
        color="#D94B45" if disease_pct > 10 else "#17231D"
    )

    ax.set_title("Vegetation Health Distribution", fontsize=12, fontweight="bold", color="#17231D", pad=12)
    plt.tight_layout()
    return fig

def generate_full_presentation_figure(
    analysis_result: Dict[str, Any],
    save_dir: str = "results"
) -> Tuple[plt.Figure, str]:
    """
    Generates a comprehensive 6-panel academic presentation grid suitable for reports,
    PPT slides, and viva voce defense.

    Grid Structure:
    1. Original RGB Image
    2. Segmentation Label
    3. Healthy Vegetation Mask
    4. Disease Mask
    5. Colorized Health Map (Healthy Green, Disease Red, Background Black)
    6. Summary Donut Chart (Healthy Area % vs Disease Area %)

    Includes prominent Disease Percentage & Severity Badge headers.

    Args:
        analysis_result (Dict[str, Any]): Result dict from CropHealthPipeline.
        save_dir (str): Destination folder to save figure.

    Returns:
        Tuple[plt.Figure, str]: Generated Figure and saved output path.
    """
    sample_id = analysis_result.get("image_name", "sample")
    bgr_img = analysis_result.get("bgr_image")
    lbl_details = analysis_result.get("label_analysis_details", {})
    
    # Masks
    healthy_mask = analysis_result.get("healthy_mask")
    disease_mask = analysis_result.get("disease_mask")
    
    healthy_pct = analysis_result.get("healthy_percentage", 0.0)
    disease_pct = analysis_result.get("disease_percentage", 0.0)
    severity_level = str(analysis_result.get("severity_level", "LOW")).upper()

    # Colorized Health Map
    health_map = create_colorized_health_map(healthy_mask, disease_mask)

    fig = plt.figure(figsize=(18, 10), facecolor="#0F172A")
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25, top=0.88, bottom=0.08)

    # Main Title Banner
    sev_color = SEVERITY_COLORS.get(severity_level, "#22C55E")
    banner_text = (
        f"CROP HEALTH SURVEILLANCE ANALYSIS  |  SAMPLE: {sample_id}\n"
        f"DISEASE AFFECTED AREA: {disease_pct:.2f}%    |    SEVERITY: {severity_level}"
    )
    fig.suptitle(banner_text, fontsize=15, fontweight="bold", color="white", y=0.96)

    # Panel 1: Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    if bgr_img is not None:
        rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        ax1.imshow(rgb_orig)
    else:
        ax1.text(0.5, 0.5, "Original Image", color="white", ha="center", va="center")
    ax1.set_title("1. Original Image", color="white", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Ground Truth Label
    ax2 = fig.add_subplot(gs[0, 1])
    raw_lbl = analysis_result.get("label_analysis_details", {}).get("colorized_map")
    if raw_lbl is not None:
        ax2.imshow(raw_lbl)
    else:
        ax2.imshow(healthy_mask + disease_mask, cmap="gray")
    ax2.set_title("2. Segmentation Label", color="white", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Healthy Vegetation Mask
    ax3 = fig.add_subplot(gs[0, 2])
    green_vis = np.zeros((*healthy_mask.shape, 3), dtype=np.uint8)
    green_vis[healthy_mask > 0] = HEALTHY_GREEN
    ax3.imshow(green_vis)
    ax3.set_title("3. Healthy Vegetation Mask", color="white", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Panel 4: Disease Mask
    ax4 = fig.add_subplot(gs[1, 0])
    red_vis = np.zeros((*disease_mask.shape, 3), dtype=np.uint8)
    red_vis[disease_mask > 0] = DISEASE_RED
    ax4.imshow(red_vis)
    ax4.set_title("4. Disease Region Mask", color="white", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # Panel 5: Colorized Health Map with Legend
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(health_map)
    ax5.set_title("5. Colorized Health Map", color="white", fontsize=11, fontweight="bold")
    ax5.axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(BACKGROUND_DARK)/255.0, label="Background"),
        mpatches.Patch(color=np.array(HEALTHY_GREEN)/255.0, label=f"Healthy ({healthy_pct:.1f}%)"),
        mpatches.Patch(color=np.array(DISEASE_RED)/255.0, label=f"Diseased ({disease_pct:.1f}%)")
    ]
    ax5.legend(handles=legend_patches, loc="lower right", facecolor="#1E293B", labelcolor="white", fontsize=8)

    # Panel 6: Summary Area Donut Chart
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#0F172A")
    labels = [f"Healthy\n{healthy_pct:.1f}%", f"Disease\n{disease_pct:.1f}%"]
    sizes = [max(0.01, healthy_pct), max(0.01, disease_pct)]
    colors = ["#22C55E", "#EF4444"]

    wedges, texts, autotexts = ax6.pie(
        sizes, 
        labels=labels, 
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 6.0 else "",
        pctdistance=0.75,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="#0F172A", linewidth=2),
        textprops=dict(fontsize=9, fontweight="bold", color="white")
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
        
    ax6.set_title("6. Vegetation Health Distribution", color="white", fontsize=11, fontweight="bold")

    # Save to unique filename in results/
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_path = str(out_dir / f"visualization_{sample_id}.png")
    fig.savefig(saved_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig, saved_path
