
import os
import cv2
import joblib
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PIXEL LAB | Image Quality Analyzer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_project")

MODEL_PATH = os.path.join(DATASET_DIR, "random_forest_model.pkl")
FEATURE_PATH = os.path.join(DATASET_DIR, "daftar_fitur.pkl")

# =====================================================
# 2. DARK THEME CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b0b10;
    --panel: #14141c;
    --panel2: #191923;
    --border: #292936;
    --text: #f4f3fa;
    --muted: #9998aa;
    --purple: #8957ff;
    --purple2: #6336d9;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(11,11,16,0.95);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: var(--text);
}

p, label, li {
    color: #c4c2d0;
}

section[data-testid="stSidebar"] {
    background: #111119;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1px;
    color: white;
    margin-bottom: 0;
}

.brand span {
    color: #a17aff;
}

.brand-sub {
    color: #9290a3;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 3px;
}

.hero {
    background: linear-gradient(115deg, #191525, #171321 55%, #251746);
    border: 1px solid #39304e;
    border-radius: 20px;
    padding: 34px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #c5aaff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 43px;
    font-weight: 800;
    margin: 12px 0 5px 0;
}

.hero-title span {
    color: #a17aff;
}

.hero-desc {
    color: #b6b1c7;
    font-size: 15px;
    max-width: 600px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 21px;
    font-weight: 800;
    color: white;
    margin: 22px 0 13px 0;
}

.subtle {
    color: var(--muted);
    font-size: 13px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 22px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #191923, #12121a);
    border: 1px solid #302d3e;
    border-radius: 15px;
    padding: 19px;
    min-height: 112px;
}

.metric-label {
    color: #aaa7ba;
    font-size: 12px;
    margin-bottom: 9px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    color: white;
    font-size: 25px;
    font-weight: 800;
}

.metric-note {
    color: #9c83d8;
    font-size: 11px;
    margin-top: 4px;
}

.result-card {
    background: linear-gradient(135deg, #211a32, #17151f);
    border: 1px solid #554078;
    border-radius: 17px;
    padding: 23px;
}

.result-label {
    color: #b9a3ec;
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 700;
}

.result-value {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    margin: 7px 0;
}

.result-desc {
    color: #b9b4c8;
    font-size: 13px;
}

div.stButton > button {
    background: linear-gradient(100deg, #8957ff, #6538d5);
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 43px;
    font-weight: 700;
}

div.stButton > button:hover {
    border: 1px solid #b69aff;
    color: white;
    background: #7549e7;
}

[data-testid="stFileUploader"] {
    background: #171720;
    border: 1px dashed #65518e;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background: #171720;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #aaa7ba;
}

[data-testid="stMetricValue"] {
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

hr {
    border-color: #302d3b;
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
# 3. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, feature_names


if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan: {MODEL_PATH}")
    st.stop()

if not os.path.exists(FEATURE_PATH):
    st.error(f"Daftar fitur tidak ditemukan: {FEATURE_PATH}")
    st.stop()

try:
    model, feature_names = load_model()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()


# =====================================================
# 4. FEATURE EXTRACTION
# =====================================================

def extract_features(image):
    image = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = []

    # RGB Mean
    for channel in range(3):
        features.append(float(np.mean(rgb[:, :, channel])))

    # RGB Histogram: 16 bins per channel
    for channel in range(3):
        histogram = cv2.calcHist(
            [rgb], [channel], None, [16], [0, 256]
        )
        histogram = cv2.normalize(
            histogram, histogram
        ).flatten()

        features.extend(histogram.tolist())

    # Sharpness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    features.append(float(sharpness))

    return np.array(features).reshape(1, -1)


def predict_image(image):
    features = extract_features(image)

    if feature_names is not None and len(feature_names) == features.shape[1]:
        input_data = pd.DataFrame(
            features,
            columns=feature_names
        )
    else:
        input_data = features

    prediction = model.predict(input_data)[0]

    probabilities = None
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities)) * 100

    return prediction, confidence, probabilities


# =====================================================
# 5. IMAGE METRICS
# =====================================================

def calculate_metrics(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    red_mean = float(np.mean(rgb[:, :, 0]))
    green_mean = float(np.mean(rgb[:, :, 1]))
    blue_mean = float(np.mean(rgb[:, :, 2]))

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    return rgb, red_mean, green_mean, blue_mean, sharpness


def show_histogram(rgb):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#14141c")
    ax.set_facecolor("#14141c")

    colors = ["#ff555f", "#40d98a", "#4287ff"]
    labels = ["Red", "Green", "Blue"]

    for channel, color, label in zip(range(3), colors, labels):
        hist = cv2.calcHist(
            [rgb], [channel], None, [256], [0, 256]
        )
        ax.plot(hist, color=color, linewidth=1.5, label=label)

    ax.set_xlim([0, 255])
    ax.tick_params(colors="#aaa7ba")
    ax.grid(alpha=0.12)
    ax.legend(facecolor="#20202b", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#393544")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# 6. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:12px 4px 25px 4px;">
        <div class="brand">◈ PIXEL <span>LAB</span></div>
        <div class="brand-sub">IMAGE QUALITY ANALYZER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
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

    st.markdown("### About PIXEL LAB")
    st.caption(
        "A digital image quality classification application "
        "powered by the Random Forest algorithm."
    )

    st.markdown("---")
    st.caption("PIXEL LAB  •  Version 1.0.0")


# =====================================================
# 8. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">✦ SMART IMAGE ANALYSIS</div>
        <div class="hero-title">PIXEL <span>LAB</span></div>
        <div style="color:#c5b4ef;font-size:16px;font-weight:600;">
            Analyze • Understand • Improve
        </div>
        <div class="hero-desc">
            Analyze digital image quality using image processing
            and machine learning technology.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Overview</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        (col1, "TEST ACCURACY", "95.46%", "Held-out test dataset"),
        (col2, "TOTAL FEATURES", "52", "Extracted image features"),
        (col3, "ALGORITHM", "Random Forest", "Machine learning model"),
        (col4, "CLASS CATEGORIES", "4", "Image quality classes")
    ]

    for col, label, value, note in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explore PIXEL LAB</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="panel">
            <h3>◈ Analyze Image</h3>
            <p class="subtle">
                Upload an image and classify its quality
                using the trained Random Forest model.
            </p>
            <p>Supported formats: JPG, JPEG, PNG, JFIF</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <h3>▥ Image Metrics</h3>
            <p class="subtle">
                Explore RGB mean, RGB histogram, and image sharpness.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <h3>◷ Prediction History</h3>
            <p class="subtle">
                Review the images you have analyzed
                during this session.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <h3>ⓘ About Model</h3>
            <p class="subtle">
                Learn about the Random Forest algorithm,
                feature extraction, and model evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Classification Categories</div>',
                unsafe_allow_html=True)

    categories = st.columns(4)

    category_data = [
        ("NORMAL", "Normal image condition", "#42c98a"),
        ("DARK", "Underexposed image", "#6495ed"),
        ("OVEREXPOSED", "Excessive brightness", "#f0b64c"),
        ("BLURRY", "Low image sharpness", "#bd8aff")
    ]

    for col, (name, description, color) in zip(categories, category_data):
        with col:
            st.markdown(f"""
            <div class="panel">
                <div style="font-weight:800;color:{color};">{name}</div>
                <div class="subtle" style="margin-top:8px;">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# 9. ANALYZE IMAGE
# =====================================================

elif page == "Analyze Image":

    st.markdown("## Analyze Your Image")
    st.markdown(
        '<p class="subtle">Upload an image to evaluate its quality.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "UPLOAD IMAGE",
        type=["jpg", "jpeg", "png", "jfif"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        left, right = st.columns([0.85, 1.15], gap="large")

        with left:
            st.markdown('<div class="section-title">Image Preview</div>',
                        unsafe_allow_html=True)

            st.image(
                pil_image,
                caption=uploaded_file.name,
                width=320
            )

            st.caption(
                f"Resolution: {pil_image.width} × {pil_image.height}"
            )

        with right:
            st.markdown('<div class="section-title">Analysis</div>',
                        unsafe_allow_html=True)

            st.markdown("""
            <div class="panel">
                <p class="subtle">
                    The image will be processed using 52 extracted
                    features before being classified by Random Forest.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("✦  ANALYZE IMAGE", use_container_width=True):

                try:
                    with st.spinner("Analyzing image..."):
                        prediction, confidence, probabilities = predict_image(
                            image_bgr
                        )

                    st.session_state.last_result = {
                        "hash": file_hash,
                        "name": uploaded_file.name,
                        "prediction": str(prediction),
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "image": image_rgb
                    }

                    st.session_state.history.append({
                        "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                        "File": uploaded_file.name,
                        "Prediction": str(prediction),
                        "Score": (
                            f"{confidence:.2f}%"
                            if confidence is not None else "N/A"
                        )
                    })

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        result = st.session_state.last_result

        if result is not None and result["hash"] == file_hash:

            st.markdown("---")
            st.markdown('<div class="section-title">Analysis Result</div>',
                        unsafe_allow_html=True)

            result_col1, result_col2 = st.columns([0.8, 1.2], gap="large")

            with result_col1:
                st.image(
                    result["image"],
                    caption="Analyzed Image",
                    width=280
                )

            with result_col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED CLASS</div>
                    <div class="result-value">{result["prediction"]}</div>
                    <div class="result-desc">
                        Image quality classification result
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if result["confidence"] is not None:
                    st.metric(
                        "Model Score",
                        f'{result["confidence"]:.2f}%'
                    )

                    st.progress(
                        min(max(result["confidence"] / 100, 0), 1)
                    )

                    st.caption(
                        "Model score is the highest predicted class score, "
                        "not a guarantee of correctness."
                    )

            if result["probabilities"] is not None:
                st.markdown("### Class Scores")

                class_scores = pd.DataFrame({
                    "Class": model.classes_,
                    "Score (%)": result["probabilities"] * 100
                })

                st.bar_chart(
                    class_scores.set_index("Class"),
                    horizontal=True
                )


# =====================================================
# 10. IMAGE METRICS
# =====================================================

elif page == "Image Metrics":

    st.markdown("## Image Metrics")
    st.markdown(
        '<p class="subtle">Explore the technical characteristics of an image.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "UPLOAD IMAGE FOR METRICS",
        type=["jpg", "jpeg", "png", "jfif"],
        key="metrics_uploader"
    )

    if uploaded_file is not None:

        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        rgb, red, green, blue, sharpness = calculate_metrics(image_bgr)

        st.image(pil_image, width=300)

        st.markdown('<div class="section-title">RGB Mean & Sharpness</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Red Mean", f"{red:.2f}")
        c2.metric("Green Mean", f"{green:.2f}")
        c3.metric("Blue Mean", f"{blue:.2f}")
        c4.metric("Sharpness", f"{sharpness:.2f}")

        st.markdown('<div class="section-title">RGB Histogram</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="panel">Distribution of pixel intensity values '
            'across red, green, and blue channels.</div>',
            unsafe_allow_html=True
        )

        show_histogram(rgb)


# =====================================================
# 11. PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    st.markdown("## Prediction History")
    st.markdown(
        '<p class="subtle">Review your previous image analyses in this session.</p>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Total analyses: {len(history_df)}")

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    else:
        st.info("No prediction history yet. Analyze an image first.")


# =====================================================
# 12. ABOUT MODEL
# =====================================================

elif page == "About Model":

    st.markdown("## About the Model")
    st.markdown(
        '<p class="subtle">Understanding the technology behind PIXEL LAB.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">MACHINE LEARNING MODEL</div>
        <div class="hero-title">Random <span>Forest</span></div>
        <div class="hero-desc">
            An ensemble learning algorithm that combines multiple
            decision trees to classify digital image quality.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Feature Extraction")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown("""
        <div class="panel">
            <h3>RGB Mean</h3>
            <p class="subtle">
                Three features representing average red, green,
                and blue intensity.
            </p>
            <h3>RGB Histogram</h3>
            <p class="subtle">
                48 features representing color intensity distribution:
                16 bins for each RGB channel.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with feature_col2:
        st.markdown("""
        <div class="panel">
            <h3>Image Sharpness</h3>
            <p class="subtle">
                One feature calculated using Variance of Laplacian
                to represent image sharpness.
            </p>
            <h3>Total: 52 Features</h3>
            <p class="subtle">
                All extracted features are used as input
                for the Random Forest classifier.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Model Evaluation")

    c1, c2 = st.columns(2)

    c1.metric("Testing Accuracy", "95.46%")
    c2.metric("Number of Classes", "4")

    st.caption(
        "The reported accuracy is based on the held-out test dataset. "
        "It does not guarantee the same performance on every new image."
    )

    st.markdown("### Classification Categories")

    st.write("""
    - **Normal:** Normal image condition.
    - **Dark:** Underexposed image.
    - **Overexposed:** Excessively bright image.
    - **Blurry:** Image with low sharpness.
    """)
