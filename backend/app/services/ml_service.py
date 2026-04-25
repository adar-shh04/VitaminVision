import os
import numpy as np
from PIL import Image
import random
import logging

logger = logging.getLogger(__name__)

# Go up from backend/app/services to project root, then to models
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "my_model.h5"))
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

class MLService:
    def __init__(self):
        self.model = None
        self.is_demo_mode = True

    def load_model(self):
        """Load the Keras model. Singleton-like pattern managed by app startup."""
        if os.path.exists(MODEL_PATH):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.is_demo_mode = False
                logger.info("Successfully loaded ML model.")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.is_demo_mode = True
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Running in Demo Mode.")
            self.is_demo_mode = True

    def predict(self, image: Image.Image):
        """Run actual model prediction or simulated prediction."""
        if not self.is_demo_mode and self.model is not None:
            img = image.resize(IMG_SIZE)
            arr = np.array(img, dtype="float32") / 255.0
            arr = np.expand_dims(arr, axis=0)
            predictions = self.model.predict(arr)
            idx = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions) * 100)
            result_label = LABELS[idx]
        else:
            # Simulate prediction for demonstration purposes
            arr = np.array(image.resize((64, 64)))
            seed = int(np.sum(arr)) % len(LABELS)
            result_label = LABELS[seed]
            confidence = round(random.uniform(78.0, 97.5), 1)

        return {
            "predicted_vitamin": result_label,
            "confidence": confidence,
            "info": VITAMIN_INFO[result_label],
            "is_demo_mode": self.is_demo_mode
        }

ml_service = MLService()
