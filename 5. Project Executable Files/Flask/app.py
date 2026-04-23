"""
VitaminVision – Flask Application
Predicts vitamins & nutrients from food images using a deep learning model.

Fixes applied:
  - Secure filename handling (path traversal prevention)
  - Model loaded once at startup (not per-request)
  - Proper Keras model loading instead of pickle
  - File-type validation
  - Relative paths (portable across machines)
  - Auto-creation of upload directory
  - Error handling with user-friendly messages
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

# Vitamin / nutrient labels the model was trained on
LABELS = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]

# Supplementary information for each prediction
VITAMIN_INFO = {
    "Vitamin A": {
        "icon": "🥕",
        "color": "#FF6B35",
        "benefits": "Essential for vision, immune function, and skin health.",
        "sources": "Carrots, sweet potatoes, spinach, kale, liver.",
        "daily_value": "900 µg RAE (men) / 700 µg RAE (women)",
    },
    "Vitamin B": {
        "icon": "🌾",
        "color": "#F7C948",
        "benefits": "Supports energy metabolism, brain function, and red blood cell production.",
        "sources": "Whole grains, eggs, dairy, legumes, leafy greens.",
        "daily_value": "Varies by B-vitamin subtype (B1–B12)",
    },
    "Vitamin C": {
        "icon": "🍊",
        "color": "#FF9F1C",
        "benefits": "Powerful antioxidant; boosts immunity, collagen synthesis, and iron absorption.",
        "sources": "Citrus fruits, strawberries, bell peppers, broccoli.",
        "daily_value": "90 mg (men) / 75 mg (women)",
    },
    "Vitamin D": {
        "icon": "☀️",
        "color": "#2EC4B6",
        "benefits": "Regulates calcium absorption, bone health, and immune modulation.",
        "sources": "Sunlight, fatty fish, fortified milk, egg yolks.",
        "daily_value": "15 µg (600 IU)",
    },
    "Vitamin E": {
        "icon": "🥜",
        "color": "#E71D36",
        "benefits": "Antioxidant that protects cells from oxidative damage; supports skin health.",
        "sources": "Nuts, seeds, spinach, vegetable oils.",
        "daily_value": "15 mg",
    },
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model ONCE at startup
# ---------------------------------------------------------------------------
model = None

def load_model():
    """Load the Keras model from disk. Called once at startup."""
    global model
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(MODEL_PATH)
        print(f"[✓] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] Model file not found at {MODEL_PATH}")
        print("    Place your trained model at the path above to enable predictions.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")

load_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in ALLOWED_EXTENSIONS."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath: str) -> np.ndarray:
    """Load an image, resize it, and return a batch-ready NumPy array."""
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0  # Normalize to [0, 1]
    return np.expand_dims(arr, axis=0)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page with the upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, run inference, and show results."""
    # --- guard: model must be loaded ----------------------------------------
    if model is None:
        flash("The prediction model is not loaded. Please contact the administrator.", "error")
        return redirect(url_for("index"))

    # --- guard: file must be present ----------------------------------------
    if "file" not in request.files:
        flash("No file was uploaded. Please select an image.", "error")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("No file selected. Please choose an image to upload.", "error")
        return redirect(url_for("index"))

    # --- guard: file type must be allowed -----------------------------------
    if not allowed_file(f.filename):
        flash(
            f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
            "error",
        )
        return redirect(url_for("index"))

    # --- save & predict -----------------------------------------------------
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
        # Clean up uploaded file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template(
        "predict.html",
        prediction=prediction_label,
        confidence=round(confidence, 1),
        info=info,
    )


@app.route("/about")
def about():
    """Simple about page (rendered inline from the template)."""
    return render_template("index.html", scroll_to="about")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=False for production; set True only during development
    app.run(debug=True, host="0.0.0.0", port=5000)
