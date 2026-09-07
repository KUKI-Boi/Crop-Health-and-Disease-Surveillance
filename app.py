"""
Crop Health and Disease Surveillance - Academic Dashboard
Image Processing Based Crop Disease Segmentation and Severity Estimation
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd

# Import core modules from src package
from src.config import SUPPORTED_CROPS, SEVERITY_THRESHOLDS, REPORT_CONFIG
from src.dataset.loader import DatasetLoader
from src.image_processing.preprocessing import load_and_preprocess_image
from src.image_processing.label_analysis import LabelAnalyzer, LabelClassConfig
from src.analysis.pipeline import CropHealthPipeline
from src.analysis.batch_processor import BatchProcessor
from src.analysis.report_generator import generate_markdown_report
from src.visualization import (
    create_colorized_health_map, create_summary_chart, generate_full_presentation_figure
)

# Page configuration
st.set_page_config(
    page_title="Crop Health and Disease Surveillance",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI Styling - AgriVision Professional Palette
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">

    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #F5F7F2 !important;
        font-family: 'Inter', sans-serif;
        color: #17231D !important;
    }
    
    /* Ensure main area text readability */
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #17231D !important;
    }

    /* Top Navbar Header */
    header[data-testid="stHeader"] {
        background-color: #F5F7F2 !important;
        border-bottom: 1.5px solid #D9E2DC !important;
    }
    header[data-testid="stHeader"] * {
        color: #174A3A !important;
        fill: #174A3A !important;
    }
    div[data-testid="stDecoration"] {
        background-image: none !important;
        background-color: #174A3A !important;
        height: 3px !important;
    }

    /* Main Header / Hero Banner (Deep Forest #174A3A) */
    .hero-container {
        background: #174A3A;
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(23, 35, 29, 0.08);
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #B8CEC0 !important;
        margin-top: 8px;
        margin-bottom: 14px;
    }
    .hero-badge {
        display: inline-block;
        background-color: #82B892;
        color: #10352A !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 14px;
        border-radius: 999px;
    }
    
    /* Info Panels (Subtle Sage Background) */
    .info-panel {
        background-color: #EAF1EC;
        border: 1.5px solid #C8D8CE;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        color: #46574E !important;
    }
    .info-panel-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #174A3A !important;
        margin-bottom: 6px;
    }
    .info-panel-body {
        color: #46574E !important;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #174A3A !important;
        border-bottom: 2px solid #D9E2DC;
        padding-bottom: 8px;
        margin-top: 28px;
        margin-bottom: 18px;
    }
    
    /* Cards (AgriVision Modern Soft Shadow Cards) */
    .metric-card {
        background-color: #FFFFFF;
        border: 1.5px solid #D9E2DC;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(23, 35, 29, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 35, 29, 0.1);
    }
    .metric-val {
        font-family: 'Manrope', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: #17231D !important;
        line-height: 1.2;
        word-break: break-word;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #65736B !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 6px;
    }
    .card-healthy { border-top: 4px solid #2E8B57; }
    .card-disease { border-top: 4px solid #D94B45; }
    .card-veg { border-top: 4px solid #5F8D72; }
    
    /* Sidebar Styling: Background #10352A, Text #FFFFFF, Secondary Text #B8CEC0 */
    section[data-testid="stSidebar"] {
        background-color: #10352A !important;
        border-right: 1.5px solid #174A3A !important;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #B8CEC0 !important;
    }
    
    /* Sidebar Inputs & Selectbox Styling */
    section[data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #17231D !important;
        border: 1.5px solid #D9E2DC !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stSelectbox"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #17231D !important;
        border: 1.5px solid #D9E2DC !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(23, 35, 29, 0.04) !important;
    }

    div[data-testid="stSelectbox"] *,
    div[data-baseweb="select"] * {
        color: #17231D !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="icon"],
    div[data-baseweb="select"] [data-baseweb="icon"],
    div[data-baseweb="select"] > div > div:last-child {
        background-color: #EAF1EC !important;
        border-left: 1.5px solid #D9E2DC !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
        min-width: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stSelectbox"] svg,
    div[data-baseweb="select"] svg {
        fill: #174A3A !important;
        color: #174A3A !important;
        stroke: #174A3A !important;
        width: 1.2rem !important;
        height: 1.2rem !important;
    }

    div[data-baseweb="popover"] ul[data-baseweb="menu"],
    ul[role="listbox"] {
        border: 1.5px solid #D9E2DC !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(23, 35, 29, 0.1) !important;
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] li,
    ul[role="listbox"] li {
        color: #17231D !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="popover"] li:hover,
    ul[role="listbox"] li:hover {
        background-color: #EAF1EC !important;
        color: #174A3A !important;
    }

    /* Streamlit Native Alert Boxes Override (Subtle Sage) */
    div[data-testid="stAlert"] {
        background-color: #EAF1EC !important;
        border: 1.5px solid #C8D8CE !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(23, 35, 29, 0.04) !important;
    }
    div[data-testid="stAlert"] * {
        color: #174A3A !important;
        font-weight: 600 !important;
    }
    
    /* Primary Button (#174A3A) */
    .stButton > button {
        background-color: #174A3A !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        border: 1.5px solid #10352A !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(23, 74, 58, 0.2) !important;
        transition: all 0.15s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #10352A !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(23, 74, 58, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary / Download Buttons (#FFFFFF background, #174A3A text & border) */
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #174A3A !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        border: 1.5px solid #174A3A !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(23, 74, 58, 0.08) !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: #EAF1EC !important;
        color: #10352A !important;
        transform: translateY(-1px) !important;
    }

    /* Native Streamlit Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D9E2DC !important;
        border-radius: 14px !important;
        padding: 14px 16px !important;
        box-shadow: 0 4px 16px rgba(23, 35, 29, 0.06) !important;
    }
    div[data-testid="stMetric"] * {
        color: #17231D !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        color: #65736B !important;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 1.7rem !important;
        color: #17231D !important;
    }

    /* Expander Box Styling */
    div[data-testid="stExpander"] {
        border: 1.5px solid #D9E2DC !important;
        border-radius: 14px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(23, 35, 29, 0.06) !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: #174A3A !important;
        background-color: #FFFFFF !important;
    }

    /* Upload Zone (#F0F5F1 background, 1.5px dashed #82B892) */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #F0F5F1 !important;
        border: 1.5px dashed #82B892 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #17231D !important;
    }
    div[data-testid="stFileUploaderFileData"],
    section[data-testid="stFileUploaderFileData"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D9E2DC !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(23, 35, 29, 0.05) !important;
        padding: 8px 12px !important;
    }

    /* Table & Dataframe styling */
    .stDataFrame, div[data-testid="stTable"] {
        border: 1.5px solid #D9E2DC !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(23, 35, 29, 0.06) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header (AgriVision Theme Banner)
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🌿 Crop Health & Disease Surveillance</h1>
    <div class="hero-subtitle">Image Processing Based Crop Disease Segmentation & Severity Estimation</div>
    <div class="hero-badge">● Analysis Engine Ready</div>
</div>
<div class="info-panel">
    <div class="info-panel-title">🎓 Academic Prototype Note</div>
    <div class="info-panel-body">This application demonstrates a deterministic, classical image processing pipeline (ground-truth label class extraction and exact pixel area ratio calculations). Designed for integration with agricultural drone surveillance imagery.</div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR CONTROLS
st.sidebar.header("⚙️ Configuration & Inputs")

mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Select Dataset Sample", "Upload Custom Image & Label", "Batch Analysis"]
)

if mode == "Batch Analysis":
    st.sidebar.subheader("📁 Batch Processing Options")
    batch_dataset_path = st.sidebar.text_input("Dataset Directory Path", value="Dataset")
    
    category_options = ["All Disease Categories"] + list(SUPPORTED_CROPS.keys())
    selected_batch_cat = st.sidebar.selectbox("Select Target Disease Folder", options=category_options)
    
    run_batch = st.sidebar.button("🚀 Run Batch Analysis")

    st.markdown('<div class="section-header">📁 Batch Folder Processing & Summary Metrics</div>', unsafe_allow_html=True)

    processor = BatchProcessor(dataset_dir=batch_dataset_path)

    if run_batch or "batch_results" in st.session_state:
        if run_batch:
            with st.spinner("Processing batch image-label pairs..."):
                if selected_batch_cat == "All Disease Categories":
                    batch_res = processor.process_all_categories()
                else:
                    batch_res = processor.process_category(selected_batch_cat)
                st.session_state["batch_results"] = batch_res
        else:
            batch_res = st.session_state["batch_results"]

        df_res = batch_res["results_df"]
        stats = batch_res["summary_stats"]

        if df_res.empty:
            st.warning(f"No valid image-label pairs found in dataset path: `{batch_dataset_path}`.")
        else:
            st.success(f"Batch Processing Complete! Analyzed {stats['total_images_analyzed']} valid samples for `{batch_res['category']}`.")

            # Summary Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{stats["total_images_analyzed"]}</div><div class="metric-label">Total Images Analyzed</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{stats["average_disease_percentage"]:.2f}%</div><div class="metric-label">Average Disease Area %</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{stats["minimum_disease_percentage"]:.1f}% / {stats["maximum_disease_percentage"]:.1f}%</div><div class="metric-label">Min / Max Disease %</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{stats["num_high_severity"] + stats["num_severe_severity"]}</div><div class="metric-label">High / Severe Cases</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Severity Distribution Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Low (0-10%)", stats["num_low_severity"])
            s2.metric("Moderate (10-30%)", stats["num_moderate_severity"])
            s3.metric("High (30-50%)", stats["num_high_severity"])
            s4.metric("Severe (>50%)", stats["num_severe_severity"])

            # Data Table
            st.markdown("### Batch Results Data Table")
            st.dataframe(df_res, use_container_width=True)

            # Charts
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Disease Area Percentage per Sample")
                st.bar_chart(df_res.set_index("image_name")["disease_percentage"])

            with col_chart2:
                st.markdown("#### Infection Severity Distribution")
                sev_counts = df_res["severity"].value_counts()
                st.bar_chart(sev_counts)

            # Report & CSV Downloads
            st.markdown("---")
            st.markdown("### 📄 Formal Academic Report Generation")
            
            from src.analysis.report_generator import generate_batch_academic_report
            
            # Select representative sample (median or first sample)
            rep_sample_info = None
            if not df_res.empty:
                med_row = df_res.iloc[len(df_res) // 2]
                rep_sample_info = {
                    "image_name": med_row["image_name"],
                    "disease_percentage": float(med_row["disease_percentage"]),
                    "severity_level": str(med_row["severity"])
                }

            batch_md_report = generate_batch_academic_report(batch_res, rep_sample_info)

            with st.expander("📄 Preview Full 14-Section Academic Submission Report"):
                st.markdown(batch_md_report)

            col_down1, col_down2 = st.columns(2)
            with col_down1:
                st.download_button(
                    label="📄 Download Academic Project Report (.md)",
                    data=batch_md_report,
                    file_name=f"crop_health_academic_report_{selected_batch_cat}.md",
                    mime="text/markdown"
                )
            with col_down2:
                csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Batch Summary CSV (.csv)",
                    data=csv_bytes,
                    file_name=f"crop_health_summary_{selected_batch_cat}.csv",
                    mime="text/csv"
                )

    else:
        st.info("Click **'🚀 Run Batch Analysis'** in the sidebar to execute batch processing over the selected disease folder.")

else: # Single Sample or Custom Upload Mode
    # Crop Category Selection
    crop_key = st.sidebar.selectbox(
        "Select Crop / Disease Category",
        options=list(SUPPORTED_CROPS.keys()),
        format_func=lambda x: SUPPORTED_CROPS[x]["name"]
    )
    crop_info = SUPPORTED_CROPS[crop_key]

    st.sidebar.info(f"**Target**: {crop_info['crop']}\n\n**Disease**: {crop_info['disease']}\n\n{crop_info['description']}")

    bgr_image = None
    label_image = None
    sample_name = "Sample"

    if mode == "Select Dataset Sample":
        dataset_path = st.sidebar.text_input("Dataset Folder Path", value="Dataset")
        loader = DatasetLoader(dataset_path)
        discovered_samples = loader.discover_samples()
        valid_samples = [s for s in discovered_samples if s.is_valid and s.label_path]

        sample_options = {f"{s.disease_category} / {s.sample_id}": s for s in valid_samples}

        if sample_options:
            selected_key = st.sidebar.selectbox("Choose Dataset Sample Pair", options=list(sample_options.keys()))
            selected_pair = sample_options[selected_key]
            
            sample_name = selected_pair.sample_id
            if selected_pair.image_path:
                bgr_image = cv2.imread(selected_pair.image_path)
            if selected_pair.label_path:
                label_image = cv2.imread(selected_pair.label_path, cv2.IMREAD_UNCHANGED)
        else:
            st.sidebar.warning("No valid sample pairs found in Dataset folder.")

    else: # Upload Custom Mode
        uploaded_img_file = st.sidebar.file_uploader(
            "Upload Crop Leaf Image (.jpg, .png)",
            type=["jpg", "jpeg", "png"]
        )
        uploaded_lbl_file = st.sidebar.file_uploader(
            "Upload Matching Label Mask (*_label.png)",
            type=["png", "jpg"]
        )

        if uploaded_img_file is not None:
            sample_name = uploaded_img_file.name.split('.')[0]
            img_bytes = uploaded_img_file.read()
            bgr_image = load_and_preprocess_image(img_bytes)

        if uploaded_lbl_file is not None:
            lbl_bytes = uploaded_lbl_file.read()
            nparr_lbl = np.frombuffer(lbl_bytes, np.uint8)
            label_image = cv2.imdecode(nparr_lbl, cv2.IMREAD_UNCHANGED)

    # Analyze Button
    analyze_clicked = st.sidebar.button("🔬 Analyze Image")

    # MAIN PAGE BODY

    # SECTION 1: UPLOADED CROP IMAGE
    st.markdown('<div class="section-header">Section 1: Uploaded Crop Image</div>', unsafe_allow_html=True)

    if bgr_image is not None:
        rgb_preview = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, c = bgr_image.shape
        
        col_img, col_meta = st.columns([1.2, 1])
        with col_img:
            st.image(rgb_preview, caption=f"Selected Sample: {sample_name}", use_container_width=True)
        with col_meta:
            st.markdown("### Image Attributes & Metadata")
            st.write(f"**Sample ID**: `{sample_name}`")
            st.write(f"**Crop Name**: {crop_info['crop']}")
            st.write(f"**Suspected Disease**: {crop_info['disease']}")
            st.write(f"**Resolution**: {w} × {h} pixels")
            st.write(f"**Total Frame Pixels**: {w * h:,} pixels")

    else:
        st.info("👈 Select a sample from the Dataset or upload an image in the sidebar to begin.")

    # Check label availability
    if bgr_image is not None and label_image is None:
        st.warning("""
        ⚠️ **Missing Ground-Truth Label Mask (`*_label.png`)**
        
        This prototype system requires a matching ground-truth segmentation label file (`*_label.png`) to compute exact quantitative vegetation area percentages without fabrication.
        
        *Please upload the corresponding `*_label.png` file in the sidebar or select a complete sample pair in Dataset mode.*
        """)

    elif bgr_image is not None and label_image is not None:
        pipeline = CropHealthPipeline()
        analysis = pipeline.analyze_sample(
            image_input=bgr_image,
            label_input=label_image,
            crop_category=crop_key,
            image_name=sample_name
        )

        # SECTION 2: IMAGE PROCESSING RESULTS
        st.markdown('<div class="section-header">Section 2: Image Processing Results</div>', unsafe_allow_html=True)
        
        col_orig, col_seg, col_map = st.columns(3)
        
        with col_orig:
            st.markdown("**Original Image**")
            st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col_seg:
            st.markdown("**Ground-Truth Label**")
            if "colorized_map" in analysis["label_analysis_details"]:
                st.image(analysis["label_analysis_details"]["colorized_map"], use_container_width=True)
            else:
                st.image(label_image, use_container_width=True)
                
        with col_map:
            st.markdown("**Disease Map (Healthy Green / Disease Red)**")
            st.image(analysis["colorized_map"], use_container_width=True)

        # SECTION 3: CROP HEALTH METRICS
        st.markdown('<div class="section-header">Section 3: Crop Health Metrics</div>', unsafe_allow_html=True)
        
        healthy_pct = analysis["healthy_percentage"]
        disease_pct = analysis["disease_percentage"]
        veg_pixels = analysis["vegetation_pixels"]
        sev_level = analysis["severity_level"]
        sev_color = analysis["severity_details"]["color_code"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card card-healthy"><div class="metric-val">{healthy_pct:.2f}%</div><div class="metric-label">Healthy Area %</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card card-disease"><div class="metric-val">{disease_pct:.2f}%</div><div class="metric-label">Disease Area %</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card card-veg"><div class="metric-val">{veg_pixels:,}</div><div class="metric-label">Vegetation Area (Pixels)</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card" style="border-top: 4px solid {sev_color};"><div class="metric-val" style="color: {sev_color};">{sev_level}</div><div class="metric-label">Severity Level</div></div>', unsafe_allow_html=True)

        # SECTION 4: DISEASE AREA CHART
        st.markdown('<div class="section-header">Section 4: Disease Area Chart</div>', unsafe_allow_html=True)
        
        chart_col, note_col = st.columns([1.3, 1])
        with chart_col:
            fig_chart = create_summary_chart(healthy_pct, disease_pct)
            st.pyplot(fig_chart, use_container_width=True)
        with note_col:
            st.markdown("### Area Calculation Formula")
            st.latex(r"\text{Vegetation Area} = \text{Healthy Pixels} + \text{Disease Pixels}")
            st.latex(r"\text{Disease Area (\%)} = \frac{\text{Disease Pixels}}{\text{Vegetation Area}} \times 100")
            st.caption("Note: Background pixels are excluded from the vegetation area denominator.")

        # SECTION 5: ANALYSIS SUMMARY
        st.markdown('<div class="section-header">Section 5: Analysis Summary</div>', unsafe_allow_html=True)
        
        summary_df = pd.DataFrame([{
            "Crop Category": crop_info["crop"],
            "Pathogen / Disease": crop_info["disease"],
            "Disease Affected Area (%)": f"{disease_pct:.2f}%",
            "Severity Classification": sev_level,
            "Action Advisory": analysis["severity_details"]["recommendation"]
        }])
        
        st.table(summary_df)

        st.info(f"**Interpretation Note**: Severity classification (`{sev_level}`) is based on project-defined thresholds: "
                f"Low (0-10%), Moderate (10-30%), High (30-50%), Severe (>50%). These are prototype parameters for project evaluation.")

        # SECTION 6: DOWNLOAD RESULTS
        st.markdown('<div class="section-header">Section 6: Download Results</div>', unsafe_allow_html=True)
        
        d1, d2, d3 = st.columns(3)
        
        with d1:
            map_rgb = cv2.cvtColor(analysis["colorized_map"], cv2.COLOR_RGB2BGR)
            _, map_buf = cv2.imencode(".png", map_rgb)
            st.download_button(
                label="🖼️ Download Disease Map (.png)",
                data=map_buf.tobytes(),
                file_name=f"disease_map_{sample_name}.png",
                mime="image/png"
            )

        with d2:
            report_md = generate_markdown_report(analysis)
            st.download_button(
                label="📄 Download Analysis Report (.md)",
                data=report_md,
                file_name=f"report_{sample_name}.md",
                mime="text/markdown"
            )

        with d3:
            csv_data = pd.DataFrame([{
                "sample_id": sample_name,
                "crop_category": crop_key,
                "total_image_pixels": analysis["total_image_pixels"],
                "vegetation_pixels": analysis["vegetation_pixels"],
                "healthy_pixels": analysis["healthy_pixels"],
                "disease_pixels": analysis["disease_pixels"],
                "healthy_percentage": healthy_pct,
                "disease_percentage": disease_pct,
                "severity_level": sev_level
            }]).to_csv(index=False)

            st.download_button(
                label="📊 Download CSV Summary (.csv)",
                data=csv_data,
                file_name=f"metrics_{sample_name}.csv",
                mime="text/csv"
            )
