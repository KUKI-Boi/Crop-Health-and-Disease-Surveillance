# 🌿 Academic Documentation: Crop Health and Disease Surveillance

## Project Overview
- **Title**: Crop Health and Disease Surveillance
- **Domain**: Classical Computer Vision & Digital Image Processing (DIP)
- **Target Context**: Early disease detection and vegetation stress estimation for drone-assisted precision agriculture.

---

## Problem Statement
Farmers often discover crop diseases only after significant yield loss has occurred. Early disease detection across expansive agricultural fields is essential for timely intervention. Aerial monitoring via drones equipped with multi-spectral or high-resolution RGB sensors provides a continuous surveillance capability. This project presents a classical image processing pipeline designed to isolate vegetation, segment diseased regions, quantify infection severity, and compile academic submission reports.

---

## Technical Architecture & Methodology

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Input Crop Image│ ──> │ Color Space Transformation │ ──> │ Vegetation Isolation   │
│ (JPG / PNG)     │     │ (HSV, Lab, ExG)      │     │ (Green Thresholding)   │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Academic Report │ <── │ Infection Severity   │ <── │ Sub-Segmentation       │
│ & Viva Output   │     │ & Area Metrics (%)   │     │ (Healthy vs Diseased)  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

### 1. Preprocessing & Noise Filtering
- Gaussian Smoothing ($5 \times 5$ kernel) to eliminate high-frequency camera noise and sensor artifact.
- Normalization and color space mapping (BGR to RGB, HSV, Lab, and Excess Green Index $ExG = 2G - R - B$).

### 2. Vegetation Region Isolation
- Leaf vegetation is extracted using dynamic HSV upper/lower bounds.
- Morphological Opening (Erosion followed by Dilation) cleans isolated pixel noise.
- Morphological Closing fills internal voids within healthy leaf structures.

### 3. Sub-Segmentation (Healthy vs. Diseased)
- Analysis within Lab color space: Leaf health is characterized by the `a*` channel (green-red spectrum).
- Adaptive Otsu's thresholding automatically calculates an optimal threshold separating intact green foliage from chlorotic/necrotic lesions.

### 4. Label Inspector Module (`LabelInspector`)
To guarantee dataset compatibility across different labeling schemes:
- **No hardcoded assumptions**: The program dynamically analyzes ground-truth label files (`*_label.png`).
- Evaluates histogram frequency distribution to determine background, healthy tissue, and disease spot classes safely.

---

## Dataset Description
**Source**: Kaggle Field-acquired Plant Disease Dataset ([Alex Chen](https://www.kaggle.com/datasets/alexzcheny/testdataset))

**Classes**:
1. `wheat_stripe_rust` (*Puccinia striiformis*)
2. `soybean_bacterial_blight` (*Pseudomonas syringae*)
3. `cedar_apple_rust` (*Gymnosporangium juniperi-virginianae*)

*Note on Drone Integration*: Dataset images are field-acquired high-resolution close-ups/canopy shots. The developed image processing algorithms form the algorithmic core of a prototype system scalable to drone payload telemetry.

---

## Viva-Voce Key Questions & Answers

**Q1: Why use Classical Image Processing instead of Deep Learning (CNN / YOLO)?**  
*Answer*: Classical image processing algorithms are deterministic, light-weight, require zero training epoch overhead, and run efficiently on edge-computing devices mounted directly on agricultural drones without needing powerful GPU hardware. Furthermore, color-space metrics (HSV, ExG) provide direct physical interpretability.

**Q2: How does Excess Green Index (ExG) work?**  
*Answer*: ExG is calculated as $ExG = 2G - R - B$. It accentuates green vegetation signals while suppressing soil, rock, and shadow background, making it an ideal index for crop canopy extraction.

**Q3: How are pixel label assumptions avoided in `*_label.png`?**  
*Answer*: The `LabelInspector` module loads label arrays, extracts unique pixel intensities or BGR color values, and sorts them by frequency. The dominant frequency corresponds to background, while remaining clusters correspond to healthy leaf tissue and diseased spot regions.
