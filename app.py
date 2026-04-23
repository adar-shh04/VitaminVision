"""
VitaminVision — Flask Application (Local Development)
Predicts vitamins & nutrients from food images using a deep learning model.

Run with: python app.py
Visit: http://127.0.0.1:5000
"""

import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "my_model.h5")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
IMG_SIZE = (224, 224)

LABELS = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]

VITAMIN_INFO = {
    "Vitamin A": {
        "icon": "🥕", "color": "#FF6B35",
        "benefits": "Essential for vision, immune function, and skin health.",
        "sources": "Carrots, sweet potatoes, spinach, kale, liver.",
        "daily_value": "900 µg RAE (men) / 700 µg RAE (women)",
    },
    "Vitamin B": {
        "icon": "🌾", "color": "#F7C948",
        "benefits": "Supports energy metabolism, brain function, and red blood cell production.",
        "sources": "Whole grains, eggs, dairy, legumes, leafy greens.",
        "daily_value": "Varies by B-vitamin subtype (B1–B12)",
    },
    "Vitamin C": {
        "icon": "🍊", "color": "#FF9F1C",
        "benefits": "Powerful antioxidant; boosts immunity, collagen synthesis, and iron absorption.",
        "sources": "Citrus fruits, strawberries, bell peppers, broccoli.",
        "daily_value": "90 mg (men) / 75 mg (women)",
    },
    "Vitamin D": {
        "icon": "☀️", "color": "#2EC4B6",
        "benefits": "Regulates calcium absorption, bone health, and immune modulation.",
        "sources": "Sunlight, fatty fish, fortified milk, egg yolks.",
        "daily_value": "15 µg (600 IU)",
    },
    "Vitamin E": {
        "icon": "🥜", "color": "#E71D36",
        "benefits": "Antioxidant that protects cells from oxidative damage; supports skin health.",
        "sources": "Nuts, seeds, spinach, vegetable oils.",
        "daily_value": "15 mg",
    },
}

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Load Model Once
# ---------------------------------------------------------------------------
model = None

def load_model():
    global model
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(MODEL_PATH)
        print(f"[✓] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] Model file not found at {MODEL_PATH}")
        print("    Place your trained model there to enable predictions.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")

load_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath: str) -> np.ndarray:
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("The prediction model is not loaded. Please contact the administrator.", "error")
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("No file was uploaded. Please select an image.", "error")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "error")
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    try:
        image_array = preprocess_image(filepath)
        predictions = model.predict(image_array)
        pred_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions) * 100)
        prediction_label = LABELS[pred_index]
        info = VITAMIN_INFO.get(prediction_label, {})
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template(
        "predict.html",
        prediction=prediction_label,
        confidence=round(confidence, 1),
        info=info,
    )

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
