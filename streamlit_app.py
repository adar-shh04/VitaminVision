"""
VitaminVision — Streamlit Frontend Client
AI-powered nutrient detection from food images.
Communicates with the FastAPI Backend.
"""

import streamlit as st
from PIL import Image
import requests
import io
import datetime

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
API_BASE_URL = "http://localhost:8000/api/v1"

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
# API Functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def check_backend_health():
    """Check if the backend is running and model is loaded."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

def fetch_history():
    """Fetch prediction history from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/history", timeout=5)
        if response.status_code == 200:
            return response.json().get("predictions", [])
    except Exception:
        return []
    return []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 VitaminVision")
    st.markdown("---")

    health_status = check_backend_health()
    if health_status:
        st.success("✅ Backend Online")
        if health_status.get("model_loaded"):
            st.info("🧠 Model Loaded")
            demo_mode = False
        else:
            st.warning("⚠️ Demo Mode")
            st.caption("Model not found on backend.")
            demo_mode = True
        backend_online = True
    else:
        st.error("❌ Backend Offline")
        st.caption("Please ensure the FastAPI backend is running on port 8000.")
        backend_online = False
        demo_mode = False

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "VitaminVision uses a **VGG19-based CNN** "
        "to classify food images by their dominant vitamin profile."
    )
    st.markdown(
        "**Tech Stack:** FastAPI, MongoDB, Streamlit, Keras"
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

tab1, tab2 = st.tabs(["🔍 Predict", "📚 History"])

with tab1:
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

        analyze_btn = st.button("🔍 Analyze Nutrients", use_container_width=True, type="primary", disabled=not backend_online)

        if analyze_btn:
            with st.spinner("🧠 Running neural network analysis..."):
                try:
                    # Send image to backend
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        result = data["predicted_vitamin"]
                        confidence = data["confidence"]
                        info = data["info"]
                        is_demo = data["is_demo_mode"]

                        # Display Results
                        html_content = f"""
<div class="result-card">
<div class="result-icon">{info['icon']}</div>
<div style="font-size: 0.85rem; color: #94a1b2; text-transform: uppercase; letter-spacing: 2px;">Predicted Nutrient</div>
<div class="result-vitamin" style="color: {info['color']};">{result}</div>
<div class="result-confidence">Confidence: {confidence:.2f}%</div>
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
"""
                        st.markdown(html_content, unsafe_allow_html=True)

                        if is_demo:
                            st.info(
                                "💡 This is a **simulated prediction**. To get real results, "
                                "place the `.h5` file in `models/`."
                            )
                    else:
                        st.error(f"Backend Error: {response.json().get('detail', 'Unknown Error')}")
                        
                except Exception as e:
                    st.error(f"Failed to communicate with backend: {e}")

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
            <div class="feature-desc">VGG19 neural network processes and classifies the image via FastAPI backend</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">3. Results</div>
            <div class="feature-desc">Get the dominant vitamin, confidence score, and nutrition info</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Prediction History")
    
    if not backend_online:
        st.warning("Cannot fetch history. Backend is offline.")
    else:
        with st.spinner("Fetching history..."):
            history = fetch_history()
            
            if not history:
                st.info("No prediction history found. Upload an image to make your first prediction!")
            else:
                for item in history:
                    date_str = item.get("created_at", "")
                    try:
                        # Format ISO date to readable string
                        if date_str:
                            dt = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            date_str = dt.strftime("%b %d, %Y - %I:%M %p")
                    except:
                        pass

                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0; color: #a78bfa; font-family: 'Outfit', sans-serif;">{item.get('predicted_vitamin', 'Unknown')}</h4>
                                <span style="font-size: 0.8rem; color: #94a1b2;">File: {item.get('filename', 'Unknown')}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #2cb67d; font-weight: bold; font-size: 1.1rem;">{item.get('confidence', 0):.1f}%</div>
                                <div style="font-size: 0.75rem; color: #94a1b2;">{date_str}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2026 VitaminVision — Built with ❤️ and Deep Learning</p>
    <p>VIT University, Vellore, India</p>
</div>
""", unsafe_allow_html=True)