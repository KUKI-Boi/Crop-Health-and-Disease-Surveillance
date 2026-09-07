# 🌾 Crop Health and Disease Surveillance

An academic Image Processing & Computer Vision project designed to detect plant leaf diseases, segment healthy vs. infected vegetation regions, quantify infection severity percentages, and auto-generate project submission reports.

Designed as an algorithmic prototype for integration with **drone-based aerial crop monitoring**.

---

## 📌 Problem Statement
Farmers often discover crop diseases only after significant yield loss has occurred. Early disease detection across large agricultural fields is vital for preventing crop loss. Drone-based monitoring provides an effective method for identifying abnormal color patterns and vegetation stress indicators across large farms. 

This project implements a classical, non-deep-learning computer vision pipeline using OpenCV and NumPy to segment diseased leaf regions, calculate infection area percentages, and classify disease severity.

---

## 🛠️ Technology Stack
- **Language**: Python 3
- **Image Processing**: OpenCV, NumPy
- **Data & Metrics**: Pandas
- **Visualization**: Matplotlib, Pillow
- **Dashboard Interface**: Streamlit
- *Note*: Pure image processing project — no deep learning (YOLO/CNN/TensorFlow/PyTorch) models are used.

---

## 📁 Project Architecture

```
Crop_Health_Project/
├── app.py                     # Streamlit Academic Dashboard UI
├── requirements.txt           # Python dependency specifications
├── README.md                  # Main academic project documentation
├── config.py                  # Project parameters & severity thresholds
├── src/                       # Core python package
│   ├── __init__.py
│   ├── config.py              # System parameters & threshold settings
│   ├── image_processing/      # Image transformation & segmentation modules
│   │   ├── __init__.py
│   │   ├── label_inspector.py # Dynamic label inspection (*_label.png)
│   │   ├── preprocessing.py   # Color conversions (HSV, Lab, ExG) & smoothing
│   │   └── segmentation.py    # HSV & Otsu thresholding segmentation algorithms
│   ├── analysis/              # Metrics & severity classification modules
│   │   ├── __init__.py
│   │   ├── metrics.py         # Dynamic pixel area metric calculations
│   │   ├── severity.py        # Infection severity level classifier
│   │   └── report_generator.py# Academic markdown report compiler
│   └── visualization/         # Plotting & overlay tools
│       ├── __init__.py
│       └── display.py         # Multi-panel pipeline plots & color overlays
├── tests/                     # Unit tests & verification routines
│   ├── __init__.py
│   └── test_structure.py      # Module sanity & calculation tests
├── results/                   # Directory for generated reports & exported masks
│   └── README.md
└── documentation/             # Detailed academic project write-up & viva guide
    └── project_overview.md
```

---

## 📊 Dataset Overview
The project is designed for compatibility with the **Kaggle Field-acquired Plant Disease Dataset**:
- **Dataset Link**: [Field-acquired Plant Disease Dataset on Kaggle](https://www.kaggle.com/datasets/alexzcheny/testdataset)

### Dataset Categories:
1. `wheat_stripe_rust` (*Puccinia striiformis*)
2. `soybean_bacterial_blight` (*Pseudomonas syringae*)
3. `cedar_apple_rust` (*Gymnosporangium juniperi-virginianae*)

### File Naming Convention per Sample:
- `*.jpg`: Original RGB leaf sample.
- `*_black.png`: Background-removed image.
- `*_label.png`: Segmentation label mask containing background, healthy leaf, and disease spots.

> **Dynamic Label Inspection**: The project's `LabelInspector` module evaluates the actual pixel intensity and color distributions of `*_label.png` files dynamically without assuming hardcoded pixel values.

---

## 🚀 Installation & Running

### 1. Install Dependencies
Ensure Python 3.8+ is installed. Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
To verify project structure and calculation logic:
```bash
python -m unittest discover tests
```

### 3. Launch the Streamlit Dashboard
Launch the interactive academic dashboard:
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 🎯 Key Application Features
1. **Interactive Image Upload**: Support for uploading RGB leaf images and ground-truth segmentation masks.
2. **Multi-Stage Pipeline View**: Visualizes preprocessing, Excess Green Index (ExG), vegetation isolation, healthy region, and disease spot segmentation.
3. **Dynamic Area Calculations**: Calculates healthy leaf percentage and disease-affected area percentage accurately without hardcoded values.
4. **Severity Classification**: Maps disease percentage against academic thresholds (Healthy, Low, Moderate, High, Severe).
5. **Academic Report Generator**: Compiles a formal report formatted in Markdown suitable for project submission.
6. **Viva-Voce Defense Guide**: Includes key computer vision concepts, formulas, and Q&A notes.
