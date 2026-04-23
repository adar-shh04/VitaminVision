import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VitaminVision AI",
    page_icon="🔬",
    layout="centered",
)

# --- PREMIUM STYLING ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background: linear-gradient(135deg, #7f5af0, #6c3fcf);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(127, 90, 240, 0.4);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #a78bfa !important;
        font-family: 'Outfit', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
LABELS = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]
VITAMIN_INFO = {
    "Vitamin A": {"icon": "🥕", "color": "#FF6B35", "benefits": "Vision & Immune support", "sources": "Carrots, Spinach"},
    "Vitamin B": {"icon": "🌾", "color": "#F7C948", "benefits": "Energy & Brain function", "sources": "Whole Grains, Eggs"},
    "Vitamin C": {"icon": "🍊", "color": "#FF9F1C", "benefits": "Immunity & Skin", "sources": "Citrus, Peppers"},
    "Vitamin D": {"icon": "☀️", "color": "#2EC4B6", "benefits": "Bone health & Immunity", "sources": "Sunlight, Fatty Fish"},
    "Vitamin E": {"icon": "🥜", "color": "#E71D36", "benefits": "Cell protection", "sources": "Nuts, Seeds"},
}

# --- MODEL LOADING ---
@st.cache_resource
def load_prediction_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "my_model.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_prediction_model()

# --- APP LAYOUT ---
st.title("🔬 VitaminVision AI")
st.markdown("### Discover the Nutrients in Your Food")

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Nutrients"):
            if model is not None:
                with st.spinner("Analyzing with Neural Network..."):
                    # Preprocessing
                    img = image.resize((224, 224))
                    img_array = np.array(img, dtype="float32") / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediction
                    predictions = model.predict(img_array)
                    idx = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    result = LABELS[idx]
                    info = VITAMIN_INFO[result]
                    
                    # Result Display
                    st.success(f"Analysis Complete!")
                    st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.2);">
                            <h1 style="font-size: 50px;">{info['icon']}</h1>
                            <h2 style="color: {info['color']};">{result}</h2>
                            <p>Confidence: <b>{confidence:.1f}%</b></p>
                            <hr style="border: 0.5px solid rgba(255,255,255,0.1);">
                            <p><b>Primary Benefit:</b> {info['benefits']}</p>
                            <p><b>Best Sources:</b> {info['sources']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Model file not found. Please upload 'my_model.h5' to the 'models' folder.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 VitaminVision • Powered by Deep Learning")
