"""
VitaminVision — Streamlit Application
AI-powered nutrient detection from food images.

Supports two modes:
  - LIVE MODE:  Uses a trained Keras model (models/my_model.h5)
  - DEMO MODE:  Simulates predictions when the model file is unavailable
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import textwrap
import random

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------                               
st.set_page_config(
    page_title="VitaminVision AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "my_model.h5")
IMG_SIZE = (224, 224)
LABELS = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]

VITAMIN_INFO = {
    "Vitamin A": {
        "icon": "🥕",
        "color": "#FF6B35",
        "benefits": "Essential for healthy vision, immune function, and skin cell renewal. Acts as an antioxidant protecting against cellular damage.",
        "sources": "Carrots, sweet potatoes, spinach, kale, liver, eggs",
        "daily_value": "900 µg RAE (men) / 700 µg RAE (women)",
    },
    "Vitamin B": {
        "icon": "🌾",
        "color": "#F7C948",
        "benefits": "Supports energy metabolism, nervous system function, and red blood cell production. Critical for brain health.",
        "sources": "Whole grains, eggs, dairy products, legumes, leafy greens",
        "daily_value": "Varies by subtype (B1–B12)",
    },
    "Vitamin C": {
        "icon": "🍊",
        "color": "#FF9F1C",
        "benefits": "Powerful antioxidant that boosts immune defense, promotes collagen synthesis, and enhances iron absorption.",
        "sources": "Citrus fruits, strawberries, bell peppers, broccoli, tomatoes",
        "daily_value": "90 mg (men) / 75 mg (women)",
    },
    "Vitamin D": {
        "icon": "☀️",
        "color": "#2EC4B6",
        "benefits": "Regulates calcium and phosphorus absorption for bone health. Supports immune modulation and mood regulation.",
        "sources": "Sunlight exposure, fatty fish, fortified milk, egg yolks, mushrooms",
        "daily_value": "15 µg (600 IU)",
    },
    "Vitamin E": {
        "icon": "🥜",
        "color": "#E71D36",
        "benefits": "Fat-soluble antioxidant that protects cell membranes from oxidative stress. Supports skin health and immune function.",
        "sources": "Almonds, sunflower seeds, spinach, avocados, vegetable oils",
        "daily_value": "15 mg",
    },
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    /* Global overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stMainBlockContainer {
        max-width: 800px;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Hero badge */
    .hero-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        background: rgba(127, 90, 240, 0.15);
        border: 1px solid rgba(127, 90, 240, 0.3);
        color: #a78bfa;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        margin: 16px 0;
    }

    /* Result card */
    .result-card {
        text-align: center;
        padding: 32px 24px;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        margin: 20px 0;
    }
    .result-icon {
        font-size: 64px;
        margin-bottom: 8px;
    }
    .result-vitamin {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .result-confidence {
        color: #2cb67d;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 20px;
    }

    /* Info grid */
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 20px 0;
        text-align: left;
    }
    .info-item {
        padding: 16px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .info-item.full-width {
        grid-column: 1 / -1;
    }
    .info-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a1b2;
        margin-bottom: 4px;
    }
    .info-value {
        font-size: 0.92rem;
        line-height: 1.5;
        color: #fffffe;
    }

    /* Demo badge */
    .demo-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        background: rgba(255, 159, 28, 0.15);
        border: 1px solid rgba(255, 159, 28, 0.3);
        color: #FF9F1C;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 20px 0;
    }
    .feature-item {
        text-align: center;
        padding: 24px 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: transform 0.2s;
    }
    .feature-item:hover {
        transform: translateY(-4px);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .feature-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #fffffe;
        margin-bottom: 4px;
    }
    .feature-desc {
        font-size: 0.82rem;
        color: #94a1b2;
        line-height: 1.4;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a1b2;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px;
    }
    .footer a { color: #a78bfa; text-decoration: none; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @media (max-width: 640px) {
        .info-grid { grid-template-columns: 1fr; }
        .feature-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the Keras model. Returns None if not found."""
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"EXCEPTION DURING MODEL LOAD: {e}")
            st.error(f"Error loading model: {e}")
            return None
    return None


def predict_live(image, model):
    """Run actual model prediction."""
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    predictions = model.predict(arr)
    idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions) * 100)
    return LABELS[idx], confidence


def predict_demo(image):
    """Simulate a prediction for demonstration purposes."""
    # Use image pixel statistics to generate a deterministic-looking result
    arr = np.array(image.resize((64, 64)))
    seed = int(np.sum(arr)) % len(LABELS)
    label = LABELS[seed]
    confidence = round(random.uniform(78.0, 97.5), 1)
    return label, confidence


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 VitaminVision")
    st.markdown("---")

    model = load_model()
    if model is not None:
        st.success("✅ Model loaded")
        demo_mode = False
    else:
        st.warning("⚠️ Model not found")
        st.caption(
            "Running in **demo mode**. Place `my_model.h5` in the "
            "`models/` directory to enable real predictions."
        )
        demo_mode = True

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "VitaminVision uses a **VGG19-based CNN** "
        "to classify food images by their dominant vitamin profile."
    )
    st.markdown(
        "**Tech Stack:** TensorFlow, Keras, Streamlit, Python"
    )
    st.markdown("---")
    st.caption("Built with ❤️ at VIT University")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Hero
st.markdown('<span class="hero-badge">🧬 Deep Learning Powered</span>', unsafe_allow_html=True)
st.markdown("# Discover Vitamins in Your Food")
st.markdown(
    "Upload a food image and let our AI identify its dominant vitamin profile — "
    "with confidence scores and nutritional insights."
)

if demo_mode:
    st.markdown(
        '<span class="demo-badge">⚡ DEMO MODE — Simulated Predictions</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# Upload Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📸 Upload Food Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, PNG, WEBP, BMP — Max 200 MB",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    analyze_btn = st.button("🔍 Analyze Nutrients", use_container_width=True, type="primary")

    if analyze_btn:
        with st.spinner("🧠 Running neural network analysis..."):
            import time
            time.sleep(1.5)  # Brief pause for UX

            if demo_mode:
                result, confidence = predict_demo(image)
            else:
                result, confidence = predict_live(image, model)

            info = VITAMIN_INFO[result]

        # Display Results
        st.markdown(textwrap.dedent(f"""
        <div class="result-card">
            <div class="result-icon">{info['icon']}</div>
            <div style="font-size: 0.85rem; color: #94a1b2; text-transform: uppercase; letter-spacing: 2px;">Predicted Nutrient</div>
            <div class="result-vitamin" style="color: {info['color']};">{result}</div>
            <div class="result-confidence">Confidence: {confidence}%</div>

            <div class="info-grid">
                <div class="info-item full-width">
                    <div class="info-label">Health Benefits</div>
                    <div class="info-value">{info['benefits']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Food Sources</div>
                    <div class="info-value">{info['sources']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Daily Recommended Value</div>
                    <div class="info-value">{info['daily_value']}</div>
                </div>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)

        if demo_mode:
            st.info(
                "💡 This is a **simulated prediction**. To get real results, "
                "retrain the model using the notebook in `notebooks/` and place "
                "the `.h5` file in `models/`."
            )

st.markdown('</div>', unsafe_allow_html=True)

# How It Works
st.markdown("---")
st.markdown("### How It Works")
st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <div class="feature-icon">📤</div>
        <div class="feature-title">1. Upload</div>
        <div class="feature-desc">Take a photo or upload an image of any food item</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">2. AI Analysis</div>
        <div class="feature-desc">VGG19 neural network processes and classifies the image</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">📊</div>
        <div class="feature-title">3. Results</div>
        <div class="feature-desc">Get the dominant vitamin, confidence score, and nutrition info</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2026 VitaminVision — Built with ❤️ and Deep Learning</p>
    <p>VIT University, Vellore, India</p>
</div>
""", unsafe_allow_html=True)
