# 📊 POWERPOINT PRESENTATION SLIDES
# CROP HEALTH AND DISEASE SURVEILLANCE
*Image Processing Based Crop Disease Segmentation and Severity Estimation*

---

## SLIDE 1: Title & Cover
- **Project Title**: Crop Health and Disease Surveillance
- **Subtitle**: Image Processing Based Crop Disease Segmentation and Severity Estimation
- **Domain**: Digital Image Processing & Precision Agriculture
- **Context**: Algorithmic prototype for aerial drone payload integration
- **Key Highlight**: Non-Machine Learning, Deterministic Computer Vision

---

## SLIDE 2: Problem Statement & Motivation
- **Agricultural Loss**: Farmers discover crop disease too late, causing major yield loss.
- **Manual Inspection Deficit**: Field scouting is slow, subjective, and labor-intensive.
- **Drone Surveillance Need**: Drones enable wide-area monitoring but require lightweight, reproducible image analysis.
- **Goal**: Build an image processing pipeline to isolate foliage, compute disease area %, and classify severity.

---

## SLIDE 3: Project Objectives & Solution Overview
- **Objective 1**: Segment healthy foliage and diseased lesion regions.
- **Objective 2**: Compute exact disease area percentage without background noise distortion.
- **Objective 3**: Classify infection severity into project-defined tiers (*Low*, *Moderate*, *High*, *Severe*).
- **Objective 4**: Provide interactive Streamlit UI, batch CSV export, and academic reports.

---

## SLIDE 4: Dataset & Drone Integration Context
- **Dataset**: Kaggle Field-Acquired Plant Disease Dataset ([Alex Chen](https://www.kaggle.com/datasets/alexzcheny/testdataset)).
- **Categories**: `wheat_stripe_rust`, `soybean_bacterial_blight`, `cedar_apple_rust`.
- **Sample Files**: `.jpg` original, `*_black.png` background removed, `*_label.png` segmentation mask.
- **Drone Context**: Dataset images are field canopy shots. The pipeline forms an algorithmic core scalable to high-resolution UAV aerial payloads.

---

## SLIDE 5: System Architecture & Workflow
```text
Input Image & Label ──> Preprocessing & Color Conversion ──> Label Inspector 
                       ──> Background Removal ──> Area Math (%) 
                       ──> Severity Classifier ──> Dashboard & CSV Report
```
- **No GPU Required**: Runs on low-power onboard drone microcontrollers (Raspberry Pi / Jetson Nano).
- **100% Reproducible**: Deterministic matrix math without neural network black-box uncertainty.

---

## SLIDE 6: Ground-Truth Label Inspection (`LabelInspector`)
- **No Hardcoded Assumptions**: `LabelInspector` profiles label file pixel distributions dynamically.
- **Class Values Extracted**:
  - `0`: **Background** (Non-foliage area)
  - `127`: **Healthy Leaf Tissue** (Intact foliage)
  - `255`: **Diseased Spot / Lesion** (Fungal/bacterial pustule)

---

## SLIDE 7: Area Calculation Formula
- **Total Vegetation Area**:
  $$\text{Vegetation Area} = \text{Healthy Pixels} + \text{Disease Pixels}$$
- **Disease Affected Area (%)**:
  $$\text{Disease Area (\%)} = \frac{\text{Disease Pixels}}{\text{Healthy Pixels} + \text{Disease Pixels}} \times 100$$
- **Denominator Isolation**: Background pixels (`0`) are excluded from the vegetation area denominator.
- **Zero-Division Protection**: Returns `0.0%` safely if vegetation pixels equal `0`.

---

## SLIDE 8: Infection Severity Tiers
- **Low (0.0% – 10.0%)**: Routine surveillance; minimal disease.
- **Moderate (10.0% – 30.0%)**: Targeted organic/chemical spot treatment.
- **High (30.0% – 50.0%)**: Therapeutic fungicide application required.
- **Severe (> 50.0%)**: Critical yield risk; immediate isolation required.
- *Disclaimer*: Project-defined prototype thresholds for evaluation, not universal agricultural standards.

---

## SLIDE 9: Experimental Batch Results (`wheat_stripe_rust`)
- **Total Samples Analyzed**: **80 valid image-label pairs**
- **Average Healthy Area**: **77.15%**
- **Average Disease Area**: **22.85%**
- **Severity Breakdown**:
  - `Low`: 23 samples (28.7%)
  - `Moderate`: 33 samples (41.2%)
  - `High`: 19 samples (23.8%)
  - `Severe`: 5 samples (6.2%)

---

## SLIDE 10: Interactive Dashboard & Visualizations
- **Section 1**: Uploaded Crop Image & Metadata Preview.
- **Section 2**: Side-by-side Image Processing Results (Original | Label | Disease Map).
- **Section 3**: Metric Cards & Prominent Severity Display (`LOW`, `MODERATE`, `HIGH`, `SEVERE`).
- **Section 4**: Disease Area Donut Chart & Formula Card.
- **Section 5**: Summary Table & Action Advisory.
- **Section 6**: Download Buttons (Disease Map PNG, Markdown Report, CSV Summary).

---

## SLIDE 11: Limitations & Future Scope
- **Limitations**:
  - Dependent on ground-truth label calibration (`*_label.png`).
  - Specular sunlight highlights on wet leaves.
- **Future Scope**:
  - Onboard UAV flight deployment (Raspberry Pi 4 / Jetson Nano).
  - GIS geospatial field stress heatmaps.
  - Multi-spectral NIR / NDVI band integration.

---

## SLIDE 12: Conclusion & Viva Voce Q&A Highlights
- **Summary**: Delivered a complete, non-machine-learning computer vision system for crop health surveillance.
- **Key Viva Highlights**:
  - *Why Non-ML?* Deterministic, zero GPU requirement, instant execution on drone microcontrollers.
  - *How is background distortion avoided?* Background pixels are excluded from the vegetation area denominator.
  - *How are label values parsed?* Dynamic histogram inspection via `LabelInspector`.
