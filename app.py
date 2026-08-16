"""
AgriSense — Flask app.

Serves an upload UI and a /predict API endpoint.
Loads the trained model once at startup and runs inference
on uploaded leaf images.
"""

import json
import os
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_DIR = "models"
UPLOAD_DIR = os.path.join("static", "uploads")
IMG_SIZE = (224, 224)

app = Flask(__name__)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# Load model, class names and disease information
# --------------------------------------------------

model = None
class_names = []
disease_info = {}

try:
    model = load_model(
        os.path.join(MODEL_DIR, "agrisense_model.keras"),
        compile=False
    )

    with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
        class_names = json.load(f)

    print("[INFO] Model loaded successfully.")
    print(f"[INFO] Number of classes: {len(class_names)}")

except Exception as e:
    print(f"[ERROR] Model could not be loaded: {e}")


# Load disease information
if os.path.exists("disease_info.json"):
    try:
        with open("disease_info.json", "r") as f:
            disease_info = json.load(f)

        print("[INFO] Disease information loaded successfully.")

    except Exception as e:
        print(f"[ERROR] Could not load disease_info.json: {e}")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def prepare_image(path):
    """
    Prepare uploaded image for MobileNetV2 model.
    """

    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    return arr


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    print("[INFO] /predict request received.")

    # Check model
    if model is None:
        print("[ERROR] Model is not loaded.")

        return jsonify({
            "error": "Model not loaded. Please check the model file."
        }), 503

    # Check uploaded image
    if "image" not in request.files:
        print("[ERROR] No image received.")

        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        print("[ERROR] Empty filename.")

        return jsonify({
            "error": "No image selected."
        }), 400

    # Create unique filename
    filename = f"{uuid.uuid4().hex}_{file.filename}"

    save_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    try:

        # Save image
        file.save(save_path)

        print(f"[INFO] Image saved: {save_path}")

        # Prepare image
        x = prepare_image(save_path)

        print("[INFO] Image prepared.")

        # Run prediction
        print("[INFO] Starting prediction...")

        preds = model.predict(
            x,
            verbose=0
        )[0]

        print("[INFO] Prediction completed.")

        # Get highest probability
        top_idx = int(np.argmax(preds))

        confidence = float(preds[top_idx])

        # Check class index
        if top_idx >= len(class_names):
            raise ValueError(
                f"Prediction index {top_idx} is outside class list."
            )

        label = class_names[top_idx]

        print(
            f"[INFO] Prediction: {label} "
            f"({confidence * 100:.2f}%)"
        )

        # Get disease information
        info = disease_info.get(
            label,
            {
                "description": "No details available for this class yet.",
                "treatment": "Consult a local agricultural extension office."
            }
        )

        # Return JSON response
        response = {
            "class": label,
            "confidence": round(confidence * 100, 2),
            "description": info.get(
                "description",
                "No description available."
            ),
            "treatment": info.get(
                "treatment",
                "Consult a local agricultural extension office."
            ),
            "image_url": f"/{save_path}"
        }

        print("[INFO] Sending prediction response.")

        return jsonify(response), 200

    except Exception as e:

        print(f"[ERROR] Prediction failed: {e}")

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run application locally
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )