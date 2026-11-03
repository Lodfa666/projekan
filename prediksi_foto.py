
import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datetime import datetime

# =====================================================
# KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Pixel Lab | Image Quality Analyzer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_project")

MODEL_PATH = os.path.join(DATASET_DIR, "random_forest_model.pkl")
FEATURE_PATH = os.path.join(DATASET_DIR, "daftar_fitur.pkl")

# =====================================================
# DESAIN / CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F5F5F0;
    color: #252B27;
}

[data-testid="stSidebar"] {
    background: #202C27;
}

[data-testid="stSidebar"] * {
    color: #F3F4EF;
}

[data-testid="stSidebar"] hr {
    border-color: #46534B;
}

.main-title {
    font-family: 'Manrope', sans-serif;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #25342B;
    margin-bottom: 4px;
}

.sub-title {
    color: #788179;
    font-size: 14px;
    margin-bottom: 28px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 700;
    color: #29372E;
    margin-bottom: 12px;
}

.panel {
    background: #FFFFFF;
    border: 1px solid #E7E9E2;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}

.metric-label {
    color: #818A81;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    color: #26372D;
    font-size: 27px;
    font-weight: 800;
    margin-top: 5px;
}

.result-label {
    color: #758078;
    font-size: 12px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.result-value {
    font-family: 'Manrope', sans-serif;
    color: #26372D;
    font-size: 34px;
    font-weight: 800;
    margin-top: 6px;
}

.small-note {
    color: #879087;
    font-size: 12px;
}

div.stButton > button {
    background: #344A3B;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #49634F;
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #B8C4B8;
    border-radius: 12px;
    padding: 12px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MEMUAT MODEL
# =====================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(
            f"Daftar fitur tidak ditemukan: {FEATURE_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return model, feature_columns


# =====================================================
# EKSTRAKSI FITUR
# =====================================================

def extract_features(image):
    image = cv2.resize(image, (224, 224))

    features = {}

    # Rata-rata RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features["red_mean"] = np.mean(rgb[:, :, 0])
    features["green_mean"] = np.mean(rgb[:, :, 1])
    features["blue_mean"] = np.mean(rgb[:, :, 2])

    # Histogram RGB
    for channel, name in enumerate(["red", "green", "blue"]):
        hist = cv2.calcHist(
            [rgb],
            [channel],
            None,
            [16],
            [0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()

        for i, value in enumerate(hist):
            features[f"{name}_hist_{i}"] = float(value)

    # Ketajaman gambar
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features["sharpness"] = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return features


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 15px 0 25px 0;">
        <div style="font-size: 25px; font-weight: 800;
                    font-family: Manrope; letter-spacing: -1px;">
            ◈ PIXEL LAB
        </div>
        <div style="font-size: 10px; color: #AAB8AC;
                    letter-spacing: 2px; margin-top: 5px;">
            IMAGE QUALITY ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Analyze Image",
            "Image Metrics",
            "Prediction History",
            "About Model"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 11px; color: #AAB8AC;">
        MODEL INFORMATION
    </div>
    <div style="font-size: 14px; margin-top: 8px;">
        Random Forest Classifier
    </div>
    <div style="font-size: 11px; color: #AAB8AC; margin-top: 4px;">
        RGB · Histogram · Sharpness
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Digital Image Processing Project")
    st.caption("© 2026 Pixel Lab")


# =====================================================
# STATE APLIKASI
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Image Quality Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">A simple workspace for analyzing digital image quality using machine learning.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <div class="metric-label">WELCOME TO PIXEL LAB</div>
        <div style="font-family:Manrope; font-size:25px;
                    font-weight:800; color:#29372E; margin-top:10px;">
            Understand your image quality.
        </div>
        <div style="color:#7B857C; font-size:14px; margin-top:8px;">
            Upload an image, analyze its visual characteristics,
            and discover its quality classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Workspace overview</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="panel">
            <div class="metric-label">CLASSIFICATION</div>
            <div class="metric-value">04 Classes</div>
            <div class="small-note">Normal, Dark, Overexposed, Blurry</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <div class="metric-label">FEATURES</div>
            <div class="metric-value">52 Features</div>
            <div class="small-note">RGB, histogram and sharpness</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="panel">
            <div class="metric-label">IMAGES ANALYZED</div>
            <div class="metric-value">{len(st.session_state.history)}</div>
            <div class="small-note">Analysis in this session</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Available classifications</div>',
                unsafe_allow_html=True)

    class_cols = st.columns(4)

    class_info = [
        ("Normal", "Image with balanced brightness and visible details."),
        ("Dark", "Image with relatively low brightness."),
        ("Overexposed", "Image with relatively high brightness."),
        ("Blurry", "Image with reduced sharpness.")
    ]

    for col, (name, desc) in zip(class_cols, class_info):
        with col:
            st.markdown(f"""
            <div class="panel" style="min-height:145px;">
                <div style="font-family:Manrope; font-weight:700;
                            font-size:16px; color:#344A3B;">
                    {name}
                </div>
                <div style="font-size:12px; color:#818A81;
                            margin-top:10px;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.info("Open Analyze Image from the sidebar to start an analysis.")


# =====================================================
# ANALYZE IMAGE
# =====================================================

elif page == "Analyze Image":

    st.markdown(
        '<div class="main-title">Analyze Image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Upload an image to identify its visual quality.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "jfif"],
        help="Supported formats: JPG, JPEG, PNG and JFIF."
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        image_array = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image_array is None:
            st.error("The uploaded file could not be read as an image.")

        else:
            col1, col2 = st.columns([1.1, 0.9], gap="large")

            with col1:
                st.markdown(
                    '<div class="section-title">Image preview</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    Image.open(BytesIO(image_bytes)),
                    use_container_width=True
                )

                st.caption(
                    f"File: {uploaded_file.name} · "
                    f"Size: {len(image_bytes) / 1024:.1f} KB"
                )

            with col2:
                st.markdown(
                    '<div class="section-title">Analysis</div>',
                    unsafe_allow_html=True
                )

                if st.button("Analyze image", use_container_width=True):

                    try:
                        model, feature_columns = load_model()

                        features = extract_features(image_array)

                        X = pd.DataFrame(
                            [features],
                            columns=feature_columns
                        )

                        prediction = model.predict(X)[0]
                        probabilities = model.predict_proba(X)[0]

                        scores = {
                            str(label): float(score * 100)
                            for label, score in zip(
                                model.classes_,
                                probabilities
                            )
                        }

                        result = {
                            "filename": uploaded_file.name,
                            "prediction": str(prediction),
                            "scores": scores,
                            "features": features,
                            "time": datetime.now().strftime(
                                "%d-%m-%Y %H:%M:%S"
                            )
                        }

                        st.session_state.current_result = result
                        st.session_state.history.insert(0, result)

                        st.success("Analysis completed.")

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            result = st.session_state.current_result

            if result is not None:
                st.markdown("---")
                st.markdown(
                    '<div class="section-title">Classification result</div>',
                    unsafe_allow_html=True
                )

                result_col, score_col = st.columns([1, 1])

                with result_col:
                    st.markdown(f"""
                    <div class="panel">
                        <div class="result-label">PREDICTED CLASS</div>
                        <div class="result-value">{result["prediction"]}</div>
                        <div class="small-note" style="margin-top:8px;">
                            File: {result["filename"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with score_col:
                    confidence = max(result["scores"].values())

                    st.markdown(f"""
                    <div class="panel">
                        <div class="result-label">HIGHEST MODEL SCORE</div>
                        <div class="result-value">{confidence:.2f}%</div>
                        <div class="small-note" style="margin-top:8px;">
                            Model score, not guaranteed real-world accuracy.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("**Class scores**")

                score_df = pd.DataFrame({
                    "Class": list(result["scores"].keys()),
                    "Score (%)": list(result["scores"].values())
                }).set_index("Class")

                st.bar_chart(score_df)

                st.caption(
                    "Scores represent the model's predict_proba output. "
                    "They are not a guarantee that the classification is correct."
                )


# =====================================================
# IMAGE METRICS
# =====================================================

elif page == "Image Metrics":

    st.markdown(
        '<div class="main-title">Image Metrics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Explore the numerical features extracted from an image.</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.current_result

    if result is None:
        st.info("Analyze an image first to see its image metrics.")

    else:
        features = result["features"]

        st.markdown(
            '<div class="section-title">RGB channel averages</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        rgb_values = [
            ("Red", features["red_mean"]),
            ("Green", features["green_mean"]),
            ("Blue", features["blue_mean"])
        ]

        for col, (name, value) in zip([c1, c2, c3], rgb_values):
            with col:
                st.metric(name, f"{value:.2f}")

        st.markdown("---")

        st.markdown(
            '<div class="section-title">RGB histogram</div>',
            unsafe_allow_html=True
        )

        hist_data = {}

        for channel in ["red", "green", "blue"]:
            hist_data[channel.capitalize()] = [
                features[f"{channel}_hist_{i}"]
                for i in range(16)
            ]

        hist_df = pd.DataFrame(
            hist_data,
            index=[f"Bin {i+1}" for i in range(16)]
        )

        st.line_chart(hist_df)

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Image sharpness</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Variance of Laplacian",
            f'{features["sharpness"]:.2f}'
        )

        st.caption(
            "Sharpness is measured using the variance of the Laplacian. "
            "Its value depends on image content and resolution."
        )


# =====================================================
# PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Review images analyzed during this session.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:
        st.info("No prediction history yet. Analyze an image first.")

    else:
        history_data = []

        for item in st.session_state.history:
            history_data.append({
                "File": item["filename"],
                "Prediction": item["prediction"],
                "Highest score": (
                    f'{max(item["scores"].values()):.2f}%'
                ),
                "Time": item["time"]
            })

        st.dataframe(
            pd.DataFrame(history_data),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()


# =====================================================
# ABOUT MODEL
# =====================================================

elif page == "About Model":

    st.markdown(
        '<div class="main-title">About the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Technical overview of the image classification system.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <div class="metric-label">CLASSIFICATION ALGORITHM</div>
        <div style="font-family:Manrope; font-size:25px;
                    font-weight:800; color:#29372E; margin-top:8px;">
            Random Forest
        </div>
        <div style="color:#7B857C; font-size:14px; margin-top:8px;">
            An ensemble learning algorithm that combines multiple
            decision trees to classify digital image quality.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Feature extraction")

    st.markdown("""
    | Feature | Description | Count |
    |---|---|---:|
    | RGB Mean | Average intensity of each RGB channel | 3 |
    | RGB Histogram | 16-bin histogram for each RGB channel | 48 |
    | Sharpness | Variance of Laplacian | 1 |
    | **Total** | | **52** |
    """)

    st.markdown("### Classification categories")

    st.markdown("""
    - **Normal:** balanced image brightness and visible details.
    - **Dark:** relatively low image brightness.
    - **Overexposed:** relatively high image brightness.
    - **Blurry:** reduced image sharpness.
    """)

    st.markdown("### Model evaluation")

    st.write(
        "The model was evaluated using a separate test split "
        "from the prepared dataset."
    )

    st.metric("Test accuracy", "95.46%")

    st.caption(
        "This score reflects the prepared dataset and its synthetic "
        "image transformations. It should not be interpreted as "
        "guaranteed accuracy on all natural photographs."
    )

    st.markdown("---")

    st.markdown("**Project:** Digital Image Processing")

    st.markdown(
        "**Title:** Implementasi Algoritma Random Forest untuk "
        "Klasifikasi Kualitas Citra Digital Berdasarkan Fitur RGB, "
        "Histogram, dan Ketajaman Gambar."
    )
